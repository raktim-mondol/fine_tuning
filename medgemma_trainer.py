"""
MedGemma Fine-tuning Implementation for Histopathology Classification
Based on official Google Health MedGemma implementation
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    set_seed
)
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

from config import Config
from data_utils import HistopathDataProcessor


class MedGemmaFineTuner:
    """
    Complete fine-tuning pipeline for MedGemma-4B-IT on histopathology data
    Based on official Google Health implementation
    """
    
    def __init__(self, config: Config):
        """
        Initialize the fine-tuning pipeline
        
        Args:
            config: Configuration object containing all parameters
        """
        self.config = config
        self.model = None
        self.processor = None
        self.datasets = None
        self.dataset_info = None
        self.trainer = None
        
        # Set random seeds for reproducibility
        set_seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        self.setup_environment()
    
    def setup_environment(self):
        """Setup authentication and verify GPU compatibility"""
        # Authenticate with Hugging Face
        if self.config.hf_token:
            login(token=self.config.hf_token)
            print("✅ Successfully authenticated with Hugging Face")
        else:
            print("⚠️ No HuggingFace token provided. Make sure you have access to the model.")
        
        # Verify GPU compatibility
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA is not available. GPU required for training.")
        
        device_capability = torch.cuda.get_device_capability()
        if device_capability[0] < 8:
            raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")
        else:
            print(f"✅ GPU supports bfloat16 (compute capability: {device_capability})")
        
        # Display GPU information
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️ Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    def load_model_and_processor(self):
        """Load MedGemma model and processor with quantization"""
        print(f"📥 Loading model: {self.config.model.model_id}")
        
        # Convert string dtype to torch dtype
        dtype_mapping = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32
        }
        torch_dtype = dtype_mapping.get(self.config.model.torch_dtype, torch.bfloat16)
        
        # Model loading configuration
        model_kwargs = {
            "attn_implementation": self.config.model.attn_implementation,
            "torch_dtype": torch_dtype,
            "device_map": self.config.model.device_map,
        }
        
        # Add quantization configuration
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=self.config.quantization.load_in_4bit,
            bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=torch_dtype,
        )
        
        try:
            # Load model and processor
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.config.model.model_id,
                **model_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(self.config.model.model_id)
            
            # Use right padding to avoid issues during training
            self.processor.tokenizer.padding_side = "right"
            
            print("✅ Model and processor loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model: {str(e)}")
    
    def prepare_datasets(self) -> Tuple[DatasetDict, Dict[str, Any]]:
        """Prepare histopathology datasets"""
        print("📁 Preparing histopathology datasets...")
        
        # Initialize data processor
        data_processor = HistopathDataProcessor(self.config.data)
        
        # Process datasets
        datasets, dataset_info = data_processor.process_dataset(self.config.data.data_path)
        
        self.datasets = datasets
        self.dataset_info = dataset_info
        
        return datasets, dataset_info
    
    def setup_lora(self) -> LoraConfig:
        """Configure LoRA for the model"""
        print("⚙️ Setting up LoRA configuration...")
        
        # Create LoRA configuration
        peft_config = LoraConfig(
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            r=self.config.lora.r,
            bias=self.config.lora.bias,
            target_modules=self.config.lora.target_modules,
            task_type=self.config.lora.task_type,
            modules_to_save=self.config.lora.modules_to_save,
        )
        
        print(f"📋 LoRA Configuration:")
        print(f"   Rank (r): {self.config.lora.r}")
        print(f"   Alpha: {self.config.lora.lora_alpha}")
        print(f"   Dropout: {self.config.lora.lora_dropout}")
        print(f"   Target modules: {self.config.lora.target_modules}")
        print(f"   Modules to save: {self.config.lora.modules_to_save}")
        
        return peft_config
    
    def create_collate_function(self):
        """Create custom collation function for multimodal training (official implementation)"""
        
        def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            """Custom collation function matching official implementation"""
            texts = []
            images = []
            
            for example in examples:
                images.append([example["image"].convert("RGB")])
                texts.append(self.processor.apply_chat_template(
                    example["messages"], add_generation_prompt=False, tokenize=False
                ).strip())

            # Tokenize the texts and process the images
            batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

            # The labels are the input_ids, with the padding and image tokens masked in
            # the loss computation
            labels = batch["input_ids"].clone()

            # Mask image tokens
            image_token_id = [
                self.processor.tokenizer.convert_tokens_to_ids(
                    self.processor.tokenizer.special_tokens_map["boi_token"]
                )
            ]
            # Mask tokens that are not used in the loss computation
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            labels[labels == image_token_id] = -100
            labels[labels == 262144] = -100

            batch["labels"] = labels
            return batch
        
        return collate_fn
    
    def setup_training_args(self) -> SFTConfig:
        """Setup training arguments using SFTConfig"""
        print("⚙️ Setting up training configuration...")
        
        args = SFTConfig(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            optim=self.config.training.optim,
            logging_steps=self.config.training.logging_steps,
            save_strategy=self.config.training.save_strategy,
            eval_strategy=self.config.training.eval_strategy,
            eval_steps=self.config.training.eval_steps,
            learning_rate=self.config.training.learning_rate,
            bf16=self.config.training.bf16,
            max_grad_norm=self.config.training.max_grad_norm,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            push_to_hub=self.config.training.push_to_hub,
            report_to=self.config.training.report_to,
            gradient_checkpointing_kwargs=self.config.training.gradient_checkpointing_kwargs,
            dataset_kwargs=self.config.training.dataset_kwargs,
            remove_unused_columns=self.config.training.remove_unused_columns,
            label_names=self.config.training.label_names,
        )
        
        print(f"📋 Training Configuration:")
        print(f"   Output directory: {args.output_dir}")
        print(f"   Epochs: {args.num_train_epochs}")
        print(f"   Batch size: {args.per_device_train_batch_size}")
        print(f"   Learning rate: {args.learning_rate}")
        print(f"   Gradient accumulation: {args.gradient_accumulation_steps}")
        
        return args
    
    def train(self) -> 'SFTTrainer':
        """Execute the complete training pipeline"""
        print("🚀 Starting MedGemma fine-tuning pipeline...")
        
        # Load model and processor
        self.load_model_and_processor()
        
        # Prepare datasets
        datasets, dataset_info = self.prepare_datasets()
        
        # Setup LoRA
        peft_config = self.setup_lora()
        
        # Create collate function
        collate_fn = self.create_collate_function()
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Create trainer
        print("🔧 Initializing SFTTrainer...")
        
        # Use subset of validation set for faster evaluation (like official notebook)
        eval_dataset = datasets["val"].shuffle().select(range(min(200, len(datasets["val"]))))
        
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            processing_class=self.processor,
            data_collator=collate_fn,
        )
        
        print("🎯 Starting training...")
        print(f"📊 Training samples: {len(datasets['train'])}")
        print(f"📊 Validation samples: {len(eval_dataset)}")
        
        # Start training
        train_result = self.trainer.train()
        
        print("✅ Training completed!")
        
        # Save the model
        print("💾 Saving final model...")
        self.trainer.save_model()
        
        # Save training metadata
        self.save_training_metadata(train_result, dataset_info)
        
        return self.trainer
    
    def save_training_metadata(self, train_result, dataset_info: Dict[str, Any]):
        """Save training metadata and results"""
        metadata = {
            "model_id": self.config.model.model_id,
            "training_config": self.config.training.__dict__,
            "lora_config": self.config.lora.__dict__,
            "dataset_info": dataset_info,
            "train_result": {
                "training_loss": train_result.training_loss,
                "metrics": train_result.metrics if hasattr(train_result, 'metrics') else {},
            },
            "final_model_path": self.config.training.output_dir
        }
        
        metadata_path = Path(self.config.training.output_dir) / "training_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"💾 Training metadata saved to {metadata_path}")
    
    def run_complete_pipeline(self) -> 'SFTTrainer':
        """
        Run the complete fine-tuning pipeline
        
        Returns:
            Trained SFTTrainer instance
        """
        try:
            trainer = self.train()
            print("🎉 Fine-tuning pipeline completed successfully!")
            return trainer
            
        except Exception as e:
            print(f"❌ Error during fine-tuning: {str(e)}")
            raise


def main():
    """Main function to run fine-tuning"""
    # Load configuration
    config = Config.from_yaml("config.yaml")
    
    # Validate configuration
    if not config.data.data_path:
        raise ValueError("❌ Please set data_path in config.yaml")
    
    if not config.hf_token:
        print("⚠️ Warning: No HuggingFace token provided. Make sure you have access to MedGemma.")
    
    # Initialize and run fine-tuning
    fine_tuner = MedGemmaFineTuner(config)
    trainer = fine_tuner.run_complete_pipeline()
    
    print("🎯 Fine-tuning completed! Model saved to:", config.training.output_dir)
    print("📊 You can now use the fine-tuned model for inference or further evaluation.")


if __name__ == "__main__":
    main()
