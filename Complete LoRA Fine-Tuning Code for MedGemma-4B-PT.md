<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

## Complete LoRA Fine-Tuning Code for MedGemma-4B-PT on Histopathology Data

This comprehensive guide provides a complete implementation for fine-tuning `google/medgemma-4b-pt` using LoRA (Low-Rank Adaptation) for histopathology subtype classification[^2][^4].

---

### **1. Environment Setup and Dependencies**

```python
"""
Complete LoRA Fine-tuning Implementation for MedGemma-4B-PT
Designed for histopathology subtype classification tasks
"""

# Install required packages
"""
!pip install --upgrade --quiet transformers>=4.47.0 
!pip install --quiet bitsandbytes>=0.44.0 
!pip install --quiet datasets>=3.0.0 
!pip install --quiet evaluate>=0.4.0 
!pip install --quiet peft>=0.13.0 
!pip install --quiet trl>=0.11.0 
!pip install --quiet scikit-learn>=1.3.0
!pip install --quiet accelerate>=0.34.0
!pip install --quiet Pillow>=10.0.0
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Core ML libraries
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score, f1_score, classification_report
from huggingface_hub import login

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("✅ All dependencies imported successfully")
```


---

### **2. Authentication and GPU Verification**

```python
class MedGemmaFineTuner:
    """
    Complete fine-tuning pipeline for MedGemma-4B-PT on histopathology data
    """
    
    def __init__(self, hf_token: str):
        """
        Initialize the fine-tuning pipeline
        
        Args:
            hf_token (str): Hugging Face API token for model access
        """
        self.hf_token = hf_token
        self.model = None
        self.processor = None
        self.dataset = None
        self.setup_environment()
    
    def setup_environment(self):
        """Setup authentication and verify GPU compatibility"""
        # Authenticate with Hugging Face
        login(token=self.hf_token)
        print("✅ Successfully authenticated with Hugging Face")
        
        # Verify GPU compatibility for bfloat16
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA is not available. GPU required for training.")
            
        device_capability = torch.cuda.get_device_capability()
        if device_capability[^0] < 8:
            print(f"⚠️  GPU compute capability {device_capability} < 8.0")
            print("⚠️  Consider using fp16 instead of bfloat16 for older GPUs")
        else:
            print(f"✅ GPU supports bfloat16 (compute capability: {device_capability})")
        
        # Display GPU information
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️  Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")

# Initialize the fine-tuner
HF_TOKEN = "your_huggingface_token_here"  # Replace with your token
fine_tuner = MedGemmaFineTuner(HF_TOKEN)
```


---

### **3. Model and Processor Loading**

```python
def load_medgemma_model(self, model_id: str = "google/medgemma-4b-pt"):
    """
    Load MedGemma model and processor with optimal configurations
    
    Args:
        model_id (str): Model identifier from Hugging Face Hub
    """
    print(f"📥 Loading model: {model_id}")
    
    # Model loading configuration
    model_kwargs = {
        "attn_implementation": "eager",  # Use eager attention for stability
        "torch_dtype": torch.bfloat16,   # Use bfloat16 for efficiency
        "device_map": "auto",            # Automatic device mapping
        "trust_remote_code": True        # Allow custom model code
    }
    
    try:
        # Load model and processor
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Configure tokenizer for training
        self.processor.tokenizer.padding_side = "right"  # Right padding for training stability
        
        # Add special tokens if needed
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            
        print("✅ Model and processor loaded successfully")
        print(f"📊 Model parameters: {self.model.num_parameters():,}")
        
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {str(e)}")

# Add method to class
MedGemmaFineTuner.load_medgemma_model = load_medgemma_model

# Load the model
fine_tuner.load_medgemma_model()
```


---

### **4. Dataset Preparation for Histopathology Data**

```python
def prepare_histpath_dataset(self, 
                           data_path: str,
                           subtype_mapping: Dict[str, str],
                           train_split: float = 0.8,
                           val_split: float = 0.1) -> DatasetDict:
    """
    Prepare histopathology dataset for subtype classification
    
    Args:
        data_path (str): Path to dataset directory
        subtype_mapping (Dict): Mapping from folder names to subtype labels
        train_split (float): Training data fraction
        val_split (float): Validation data fraction
    
    Returns:
        DatasetDict: Formatted dataset ready for training
    """
    print("📁 Preparing histopathology dataset...")
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    # Collect all image files and labels
    all_data = []
    data_path = Path(data_path)
    
    for subtype_folder in data_path.iterdir():
        if subtype_folder.is_dir() and subtype_folder.name in subtype_mapping:
            subtype_label = subtype_mapping[subtype_folder.name]
            
            # Find all images in subtype folder
            image_files = [
                f for f in subtype_folder.iterdir() 
                if f.suffix.lower() in image_extensions
            ]
            
            print(f"📂 Found {len(image_files)} images for subtype: {subtype_label}")
            
            # Process each image
            for img_path in image_files:
                try:
                    # Load and verify image
                    image = Image.open(img_path).convert("RGB")
                    
                    # Create conversation format for instruction tuning
                    conversation = [
                        {
                            "role": "user", 
                            "content": f"<start_of_image> Classify the histopathology subtype in this image:"
                        },
                        {
                            "role": "assistant", 
                            "content": subtype_label
                        }
                    ]
                    
                    all_data.append({
                        "image": image,
                        "messages": conversation,
                        "subtype": subtype_label,
                        "image_path": str(img_path)
                    })
                    
                except Exception as e:
                    print(f"⚠️  Skipping corrupted image {img_path}: {e}")
                    continue
    
    print(f"✅ Total valid samples collected: {len(all_data)}")
    
    # Create label distribution
    label_counts = {}
    for item in all_data:
        label = item["subtype"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("📊 Label distribution:")
    for label, count in label_counts.items():
        print(f"   {label}: {count} samples")
    
    # Shuffle and split data
    np.random.shuffle(all_data)
    
    total_samples = len(all_data)
    train_end = int(total_samples * train_split)
    val_end = train_end + int(total_samples * val_split)
    
    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]
    
    print(f"📝 Dataset splits:")
    print(f"   Training: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples") 
    print(f"   Test: {len(test_data)} samples")
    
    # Convert to Hugging Face Dataset format
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data)
    })
    
    self.dataset = dataset_dict
    self.subtype_mapping = subtype_mapping
    
    return dataset_dict

# Add method to class
MedGemmaFineTuner.prepare_histpath_dataset = prepare_histpath_dataset

# Example usage - replace with your actual data path and mapping
SUBTYPE_MAPPING = {
    "adenocarcinoma": "adenocarcinoma",
    "squamous_cell": "squamous cell carcinoma", 
    "normal": "normal tissue",
    "inflammation": "inflammatory tissue"
    # Add your specific histopathology subtypes here
}

DATA_PATH = "/path/to/your/histpath/dataset"  # Replace with your data path

# Prepare dataset
# dataset = fine_tuner.prepare_histpath_dataset(DATA_PATH, SUBTYPE_MAPPING)
```


---

### **5. LoRA Configuration and Setup**

```python
def setup_lora_config(self, 
                     r: int = 16,
                     lora_alpha: int = 16, 
                     lora_dropout: float = 0.05,
                     target_modules: str = "all-linear") -> LoraConfig:
    """
    Configure LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    
    Args:
        r (int): Rank of adaptation matrices (lower = fewer parameters)
        lora_alpha (int): LoRA scaling parameter
        lora_dropout (float): Dropout rate for LoRA layers
        target_modules (str): Which modules to apply LoRA to
    
    Returns:
        LoraConfig: Configured LoRA setup
    """
    print("⚙️  Setting up LoRA configuration...")
    
    # LoRA configuration optimized for medical imaging
    peft_config = LoraConfig(
        r=r,                              # Rank of adaptation matrices
        lora_alpha=lora_alpha,            # Scaling parameter
        lora_dropout=lora_dropout,        # Dropout for regularization
        bias="none",                      # Don't adapt bias terms
        target_modules=target_modules,    # Apply to all linear layers
        task_type=TaskType.CAUSAL_LM,     # Causal language modeling task
        modules_to_save=[                 # Additional modules to save
            "lm_head",                    # Language model head
            "embed_tokens",               # Token embeddings
        ],
    )
    
    print(f"📋 LoRA Configuration:")
    print(f"   Rank (r): {r}")
    print(f"   Alpha: {lora_alpha}")
    print(f"   Dropout: {lora_dropout}")
    print(f"   Target modules: {target_modules}")
    
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

# Add method to class
MedGemmaFineTuner.setup_lora_config = setup_lora_config

# Setup LoRA
lora_config = fine_tuner.setup_lora_config(
    r=16,                    # Rank - higher means more parameters but better adaptation
    lora_alpha=16,           # Scaling factor
    lora_dropout=0.05,       # Dropout for regularization
    target_modules="all-linear"  # Apply to all linear layers
)
```


---

### **6. Custom Data Collation Function**

```python
def create_collate_function(self):
    """
    Create custom collation function for multimodal training
    Handles both images and text in conversation format
    
    Returns:
        Callable: Collation function for DataLoader
    """
    
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collation function for histopathology multimodal training
        
        Args:
            examples: List of dataset examples
            
        Returns:
            Dict containing batched tensors ready for training
        """
        # Extract images and prepare text
        images = []
        texts = []
        
        for example in examples:
            # Add image to batch (wrapped in list for processor)
            images.append([example["image"]])
            
            # Apply chat template to create formatted conversation
            formatted_text = self.processor.apply_chat_template(
                example["messages"], 
                add_generation_prompt=False,  # Don't add generation prompt for training
                tokenize=False                # Return string, not tokens
            ).strip()
            
            texts.append(formatted_text)
        
        # Process batch with processor
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,                    # Pad to same length
            truncation=True,                 # Truncate if too long
            max_length=512                   # Maximum sequence length
        )
        
        # Create labels for loss computation
        labels = batch["input_ids"].clone()
        
        # Get special token IDs for masking
        pad_token_id = self.processor.tokenizer.pad_token_id
        
        # Get image token ID (beginning of image token)
        special_tokens = self.processor.tokenizer.special_tokens_map
        if "boi_token" in special_tokens:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                special_tokens["boi_token"]
            )
        else:
            image_token_id = None
        
        # Mask tokens that shouldn't contribute to loss
        labels[labels == pad_token_id] = -100           # Mask padding tokens
        
        if image_token_id is not None:
            labels[labels == image_token_id] = -100     # Mask image tokens
            
        # Mask any other special image tokens (model-specific)
        labels[labels == 262144] = -100                 # Common image token ID
        
        batch["labels"] = labels
        
        return batch
    
    return collate_fn

# Add method to class
MedGemmaFineTuner.create_collate_function = create_collate_function

# Create collation function
collate_fn = fine_tuner.create_collate_function()
print("✅ Custom collation function created")
```


---

### **7. Training Configuration and Execution**

```python
def train_model(self,
               output_dir: str = "./medgemma-histpath-finetuned",
               num_epochs: int = 3,
               batch_size: int = 4,
               learning_rate: float = 2e-4,
               warmup_ratio: float = 0.1,
               save_strategy: str = "epoch") -> None:
    """
    Execute the fine-tuning process with comprehensive monitoring
    
    Args:
        output_dir (str): Directory to save model checkpoints
        num_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for optimizer
        warmup_ratio (float): Warmup ratio for learning rate scheduling
        save_strategy (str): When to save model checkpoints
    """
    print("🚀 Starting model training...")
    
    # Calculate steps
    train_dataset_size = len(self.dataset["train"])
    steps_per_epoch = train_dataset_size // batch_size
    total_steps = steps_per_epoch * num_epochs
    
    print(f"📊 Training Configuration:")
    print(f"   Dataset size: {train_dataset_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total training steps: {total_steps}")
    print(f"   Learning rate: {learning_rate}")
    
    # Training arguments using SFTConfig
    training_args = SFTConfig(
        # Output and logging
        output_dir=output_dir,
        run_name="medgemma-histpath-finetune",
        
        # Training hyperparameters
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,           # Accumulate gradients to simulate larger batch
        
        # Optimization
        learning_rate=learning_rate,
        optim="adamw_torch_fused",               # Efficient fused AdamW
        weight_decay=0.01,                       # L2 regularization
        max_grad_norm=1.0,                       # Gradient clipping
        
        # Learning rate scheduling
        lr_scheduler_type="cosine",              # Cosine annealing
        warmup_ratio=warmup_ratio,               # Warmup period
        
        # Precision and performance
        bf16=True,                               # Use bfloat16 for efficiency
        dataloader_num_workers=4,                # Parallel data loading
        remove_unused_columns=False,             # Keep all columns for multimodal
        
        # Evaluation and saving
        evaluation_strategy="epoch",             # Evaluate each epoch
        save_strategy=save_strategy,             # Save each epoch
        save_total_limit=3,                      # Keep only 3 best checkpoints
        load_best_model_at_end=True,             # Load best model after training
        metric_for_best_model="eval_loss",       # Use eval loss for best model selection
        greater_is_better=False,                 # Lower eval loss is better
        
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        report_to=None,                          # Disable wandb/tensorboard for now
        
        # Reproducibility
        seed=42,
        data_seed=42,
        
        # SFT-specific settings
        max_seq_length=512,                      # Maximum sequence length
        dataset_kwargs={"skip_prepare_dataset": True}  # We handle data preparation
    )
    
    # Create trainer with SFTTrainer
    trainer = SFTTrainer(
        model=self.model,
        args=training_args,
        train_dataset=self.dataset["train"],
        eval_dataset=self.dataset["validation"],
        data_collator=collate_fn,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,      # Stop if no improvement for 2 epochs
                early_stopping_threshold=0.01   # Minimum improvement threshold
            )
        ]
    )
    
    print("🎯 Training starting...")
    
    # Start training
    try:
        train_result = trainer.train()
        
        print("✅ Training completed successfully!")
        print(f"📈 Final training loss: {train_result.training_loss:.4f}")
        
        # Save final model
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)
        
        # Save training metadata
        training_metadata = {
            "model_id": "google/medgemma-4b-pt",
            "lora_config": {
                "r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "lora_dropout": lora_config.lora_dropout,
                "target_modules": lora_config.target_modules
            },
            "training_args": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "final_loss": train_result.training_loss
            },
            "dataset_info": {
                "total_samples": len(self.dataset["train"]),
                "num_classes": len(self.subtype_mapping),
                "class_mapping": self.subtype_mapping
            }
        }
        
        with open(f"{output_dir}/training_metadata.json", "w") as f:
            json.dump(training_metadata, f, indent=2)
        
        print(f"💾 Model and metadata saved to {output_dir}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise e

# Add method to class
MedGemmaFineTuner.train_model = train_model

# Start training (uncomment when ready)
# trainer = fine_tuner.train_model(
#     output_dir="./medgemma-histpath-finetuned",
#     num_epochs=3,
#     batch_size=4,
#     learning_rate=2e-4
# )
```


---

### **8. Model Evaluation and Inference**

```python
def evaluate_model(self, output_dir: str = "./medgemma-histpath-finetuned") -> Dict:
    """
    Comprehensive evaluation of the fine-tuned model
    
    Args:
        output_dir (str): Directory containing the fine-tuned model
        
    Returns:
        Dict: Evaluation metrics and results
    """
    print("📊 Evaluating fine-tuned model...")
    
    # Load fine-tuned model for evaluation
    eval_model = AutoModelForImageTextToText.from_pretrained(
        output_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    eval_processor = AutoProcessor.from_pretrained(output_dir)
    
    # Evaluation on test set
    test_dataset = self.dataset["test"]
    predictions = []
    ground_truth = []
    
    eval_model.eval()
    
    with torch.no_grad():
        for i, example in enumerate(test_dataset):
            if i % 50 == 0:
                print(f"   Evaluating sample {i}/{len(test_dataset)}")
            
            # Prepare input
            image = example["image"]
            input_text = "Classify the histopathology subtype in this image:"
            
            # Process input
            inputs = eval_processor(
                text=input_text,
                images=image,
                return_tensors="pt"
            ).to(eval_model.device)
            
            # Generate prediction
            outputs = eval_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,           # Deterministic generation
                temperature=0.1
            )
            
            # Decode prediction
            prediction = eval_processor.decode(
                outputs[^0][inputs["input_ids"].shape[^1]:], 
                skip_special_tokens=True
            ).strip()
            
            predictions.append(prediction)
            ground_truth.append(example["subtype"])
    
    # Calculate metrics
    # Create mapping for exact matches
    label_to_idx = {label: idx for idx, label in enumerate(set(ground_truth))}
    
    # Convert predictions to indices (handle partial matches)
    pred_indices = []
    true_indices = [label_to_idx[label] for label in ground_truth]
    
    for pred in predictions:
        # Find best matching label
        best_match = None
        best_score = 0
        
        for true_label in label_to_idx.keys():
            # Simple string matching (can be improved)
            if true_label.lower() in pred.lower():
                score = len(true_label) / len(pred) if pred else 0
                if score > best_score:
                    best_score = score
                    best_match = true_label
        
        if best_match:
            pred_indices.append(label_to_idx[best_match])
        else:
            pred_indices.append(-1)  # Unknown prediction
    
    # Calculate metrics
    accuracy = accuracy_score(true_indices, pred_indices)
    f1_macro = f1_score(true_indices, pred_indices, average='macro', zero_division=0)
    f1_weighted = f1_score(true_indices, pred_indices, average='weighted', zero_division=0)
    
    # Detailed classification report
    target_names = list(label_to_idx.keys())
    class_report = classification_report(
        true_indices, 
        pred_indices, 
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    
    evaluation_results = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": class_report,
        "num_test_samples": len(test_dataset),
        "predictions_sample": list(zip(ground_truth[:5], predictions[:5]))  # First 5 for inspection
    }
    
    print(f"✅ Evaluation completed!")
    print(f"📊 Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 (Macro): {f1_macro:.4f}")
    print(f"   F1 (Weighted): {f1_weighted:.4f}")
    
    return evaluation_results

def inference_single_image(self, 
                          image_path: str, 
                          model_dir: str = "./medgemma-histpath-finetuned") -> str:
    """
    Perform inference on a single histopathology image
    
    Args:
        image_path (str): Path to the histopathology image
        model_dir (str): Directory containing the fine-tuned model
        
    Returns:
        str: Predicted subtype classification
    """
    # Load model for inference
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    input_text = "Classify the histopathology subtype in this image:"
    
    # Prepare inputs
    inputs = processor(
        text=input_text,
        images=image,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate prediction
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.1
        )
    
    # Decode result
    prediction = processor.decode(
        outputs[^0][inputs["input_ids"].shape[^1]:], 
        skip_special_tokens=True
    ).strip()
    
    return prediction

# Add methods to class
MedGemmaFineTuner.evaluate_model = evaluate_model
MedGemmaFineTuner.inference_single_image = inference_single_image

print("✅ Evaluation and inference functions ready")
```


---

### **9. Complete Usage Example**

```python
def main():
    """
    Complete example workflow for fine-tuning MedGemma on histopathology data
    """
    # Initialize fine-tuner
    HF_TOKEN = "your_huggingface_token_here"  # Replace with your token
    fine_tuner = MedGemmaFineTuner(HF_TOKEN)
    
    # Load model
    fine_tuner.load_medgemma_model("google/medgemma-4b-pt")
    
    # Prepare dataset (replace with your data)
    SUBTYPE_MAPPING = {
        "adenocarcinoma": "adenocarcinoma",
        "squamous_cell": "squamous cell carcinoma",
        "normal": "normal tissue",
        "inflammation": "inflammatory tissue"
    }
    
    DATA_PATH = "/path/to/your/histpath/dataset"
    dataset = fine_tuner.prepare_histpath_dataset(DATA_PATH, SUBTYPE_MAPPING)
    
    # Setup LoRA
    fine_tuner.setup_lora_config(r=16, lora_alpha=16, lora_dropout=0.05)
    
    # Create collation function
    collate_fn = fine_tuner.create_collate_function()
    
    # Train model
    trainer = fine_tuner.train_model(
        output_dir="./medgemma-histpath-finetuned",
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4
    )
    
    # Evaluate model
    results = fine_tuner.evaluate_model("./medgemma-histpath-finetuned")
    
    # Test single inference
    prediction = fine_tuner.inference_single_image(
        "/path/to/test/image.png",
        "./medgemma-histpath-finetuned"
    )
    print(f"🔍 Single image prediction: {prediction}")
    
    return fine_tuner, results

# Run complete workflow
# fine_tuner, evaluation_results = main()
```

This comprehensive implementation provides a complete, production-ready pipeline for fine-tuning MedGemma-4B-PT on histopathology subtype classification tasks using LoRA[^2][^3]. The code includes extensive documentation, error handling, and monitoring capabilities, making it suitable for research and development purposes[^4].

<div style="text-align: center">⁂</div>

[^1]: https://huggingface.co/google/medgemma-4b-pt

[^2]: https://www.datacamp.com/tutorial/fine-tuning-medgemma

[^3]: https://arxiv.org/html/2411.14975v1

[^4]: https://developers.google.com/health-ai-developer-foundations/medgemma

[^5]: https://huggingface.co/google/medgemma-4b-it

[^6]: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma?pli=1\&inv=1\&invt=AbzAlg

[^7]: https://colab.research.google.com/github/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb

[^8]: https://github.com/google-health/medgemma

[^9]: https://www.youtube.com/watch?v=XDOSVh9jJiA

[^10]: https://www.youtube.com/watch?v=_xxGMSVLwU8

