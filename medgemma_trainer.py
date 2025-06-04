"""
MedGemma Fine-tuning Implementation for Histopathology Classification
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
    TrainingArguments,
    EarlyStoppingCallback,
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
    Complete fine-tuning pipeline for MedGemma-4B-PT on histopathology data
    Handles multiple patches per patient scenario
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
            print(f"⚠️ GPU compute capability {device_capability} < 8.0")
            print("⚠️ Consider using fp16 instead of bfloat16 for older GPUs")
            # Adjust config for older GPUs
            self.config.training.bf16 = False
        else:
            print(f"✅ GPU supports bfloat16 (compute capability: {device_capability})")
        
        # Display GPU information
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️ Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    def load_model_and_processor(self):
        """Load MedGemma model and processor with optimal configurations"""
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
            "trust_remote_code": self.config.model.trust_remote_code
        }
        
        try:
            # Load model and processor
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.config.model.model_id,
                **model_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(self.config.model.model_id)
            
            # Configure tokenizer for training
            self.processor.tokenizer.padding_side = "right"
            
            # Add special tokens if needed
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            
            print("✅ Model and processor loaded successfully")
            print(f"📊 Model parameters: {self.model.num_parameters():,}")
            
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
        """Configure and apply LoRA to the model"""
        print("⚙️ Setting up LoRA configuration...")
        
        # Create LoRA configuration
        peft_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            target_modules=self.config.lora.target_modules,
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["lm_head", "embed_tokens"],
        )
        
        print(f"📋 LoRA Configuration:")
        print(f"   Rank (r): {self.config.lora.r}")
        print(f"   Alpha: {self.config.lora.lora_alpha}")
        print(f"   Dropout: {self.config.lora.lora_dropout}")
        print(f"   Target modules: {self.config.lora.target_modules}")
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"🔢 Parameter Statistics:")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        return peft_config
    
    def create_collate_function(self):
        """Create custom collation function for multimodal training"""
        
        def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            """Custom collation function for histopathology multimodal training"""
            # Extract images and prepare text
            images = []
            texts = []
            
            for example in examples:
                # Add image to batch
                images.append([example["image"]])
                
                # Apply chat template to create formatted conversation
                formatted_text = self.processor.apply_chat_template(
                    example["messages"],
                    add_generation_prompt=False,
                    tokenize=False
                ).strip()
                
                texts.append(formatted_text)
            
            # Process batch with processor
            batch = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.training.max_seq_length
            )
            
            # Create labels for loss computation
            labels = batch["input_ids"].clone()
            
            # Get special token IDs for masking
            pad_token_id = self.processor.tokenizer.pad_token_id
            
            # Mask tokens that shouldn't contribute to loss
            labels[labels == pad_token_id] = -100  # Mask padding tokens
            
            # Mask image tokens (model-specific - adjust as needed)
            labels[labels == 262144] = -100  # Common image token ID
            
            batch["labels"] = labels
            
            return batch
        
        return collate_fn
    
    def train(self) -> 'SFTTrainer':
        """Execute the fine-tuning process"""
        print("🚀 Starting model training...")
        
        if self.datasets is None:
            raise ValueError("Datasets not prepared. Call prepare_datasets() first.")
        
        # Calculate steps
        train_dataset_size = len(self.datasets["train"])
        steps_per_epoch = train_dataset_size // (
            self.config.training.per_device_train_batch_size * 
            self.config.training.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * self.config.training.num_epochs
        
        print(f"📊 Training Configuration:")
        print(f"   Dataset size: {train_dataset_size}")
        print(f"   Batch size: {self.config.training.per_device_train_batch_size}")
        print(f"   Gradient accumulation: {self.config.training.gradient_accumulation_steps}")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Total training steps: {total_steps}")
        print(f"   Learning rate: {self.config.training.learning_rate}")
        
        # Create output directory
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        
        # Training arguments using SFTConfig
        training_args = SFTConfig(
            # Output and logging
            output_dir=self.config.training.output_dir,
            run_name="medgemma-histpath-finetune",
            
            # Training hyperparameters
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            
            # Optimization
            learning_rate=self.config.training.learning_rate,
            optim="adamw_torch_fused",
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            
            # Learning rate scheduling
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            warmup_ratio=self.config.training.warmup_ratio,
            
            # Precision and performance
            bf16=self.config.training.bf16,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            remove_unused_columns=False,
            
            # Evaluation and saving
            evaluation_strategy=self.config.training.evaluation_strategy,
            save_strategy=self.config.training.save_strategy,
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            
            # Logging
            logging_dir=f"{self.config.training.output_dir}/logs",
            logging_strategy="steps",
            logging_steps=self.config.training.logging_steps,
            report_to=None,
            
            # Reproducibility
            seed=self.config.seed,
            data_seed=self.config.seed,
            
            # SFT-specific settings
            max_seq_length=self.config.training.max_seq_length,
            dataset_kwargs={"skip_prepare_dataset": True}
        )
        
        # Create collation function
        collate_fn = self.create_collate_function()
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets.get("val", self.datasets.get("validation")),
            data_collator=collate_fn,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.training.early_stopping_patience,
                    early_stopping_threshold=self.config.training.early_stopping_threshold
                )
            ]
        )
        
        print("🎯 Training starting...")
        
        try:
            # Start training
            train_result = self.trainer.train()
            
            print("✅ Training completed successfully!")
            print(f"📈 Final training loss: {train_result.training_loss:.4f}")
            
            # Save final model
            self.trainer.save_model(self.config.training.output_dir)
            self.processor.save_pretrained(self.config.training.output_dir)
            
            # Save training metadata
            self.save_training_metadata(train_result)
            
            print(f"💾 Model and metadata saved to {self.config.training.output_dir}")
            
            return self.trainer
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise e
    
    def save_training_metadata(self, train_result):
        """Save training metadata and configuration"""
        training_metadata = {
            "model_id": self.config.model.model_id,
            "lora_config": {
                "r": self.config.lora.r,
                "lora_alpha": self.config.lora.lora_alpha,
                "lora_dropout": self.config.lora.lora_dropout,
                "target_modules": self.config.lora.target_modules
            },
            "training_args": {
                "num_epochs": self.config.training.num_epochs,
                "batch_size": self.config.training.per_device_train_batch_size,
                "learning_rate": self.config.training.learning_rate,
                "final_loss": train_result.training_loss
            },
            "dataset_info": self.dataset_info,
            "config": {
                "seed": self.config.seed,
                "data_path": self.config.data.data_path
            }
        }
        
        metadata_path = Path(self.config.training.output_dir) / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(training_metadata, f, indent=2, default=str)
        
        # Also save the full config
        config_path = Path(self.config.training.output_dir) / "config.yaml"
        self.config.to_yaml(str(config_path))
    
    def run_complete_pipeline(self) -> 'SFTTrainer':
        """Run the complete fine-tuning pipeline"""
        print("🚀 Starting complete MedGemma fine-tuning pipeline...")
        
        # Step 1: Load model and processor
        self.load_model_and_processor()
        
        # Step 2: Prepare datasets
        self.prepare_datasets()
        
        # Step 3: Setup LoRA
        self.setup_lora()
        
        # Step 4: Train model
        trainer = self.train()
        
        print("🎉 Complete pipeline finished successfully!")
        
        return trainer


def main():
    """Main function to run the fine-tuning pipeline"""
    # Load configuration
    config = Config.from_yaml("config.yaml") if Path("config.yaml").exists() else Config(
        model=ModelConfig(),
        lora=LoRAConfig(),
        training=TrainingConfig(),
        data=DataConfig()
    )
    
    # Initialize fine-tuner
    fine_tuner = MedGemmaFineTuner(config)
    
    # Run complete pipeline
    trainer = fine_tuner.run_complete_pipeline()
    
    return fine_tuner, trainer


if __name__ == "__main__":
    from config import ModelConfig, LoRAConfig, TrainingConfig
    
    fine_tuner, trainer = main()
