"""
Configuration file for MedGemma histopathology fine-tuning
Based on official Google Health MedGemma implementation
"""

import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    model_id: str = "google/medgemma-4b-it"  # Updated to instruction-tuned version
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    device_map: str = "auto"
    trust_remote_code: bool = True

@dataclass
class QuantizationConfig:
    """Quantization configuration for 4-bit training"""
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_storage: str = "bfloat16"

@dataclass
class LoRAConfig:
    """LoRA configuration parameters"""
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: List[str] = None
    
    def __post_init__(self):
        if self.modules_to_save is None:
            self.modules_to_save = ["lm_head", "embed_tokens"]

@dataclass
class TrainingConfig:
    """Training configuration parameters using SFTConfig"""
    output_dir: str = "medgemma-4b-it-sft-lora-histopath"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch_fused"
    logging_steps: int = 50
    save_strategy: str = "epoch"
    eval_strategy: str = "steps"
    eval_steps: int = 50
    learning_rate: float = 2e-4
    bf16: bool = True
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"
    push_to_hub: bool = True
    report_to: str = "tensorboard"
    gradient_checkpointing_kwargs: Dict = None
    dataset_kwargs: Dict = None
    remove_unused_columns: bool = False
    label_names: List[str] = None
    
    def __post_init__(self):
        if self.gradient_checkpointing_kwargs is None:
            self.gradient_checkpointing_kwargs = {"use_reentrant": False}
        if self.dataset_kwargs is None:
            self.dataset_kwargs = {"skip_prepare_dataset": True}
        if self.label_names is None:
            self.label_names = ["labels"]

@dataclass
class DataConfig:
    """Data configuration parameters"""
    data_path: str = ""  # To be set by user
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    image_extensions: List[str] = None
    max_patches_per_patient: int = 10  # Limit patches per patient for memory
    min_patches_per_patient: int = 1   # Minimum patches required
    train_size: int = 9000  # Number of training samples (like official notebook)
    validation_size: int = 1000  # Number of validation samples
    
    def __post_init__(self):
        if self.image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']

@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig
    quantization: QuantizationConfig
    lora: LoRAConfig
    training: TrainingConfig
    data: DataConfig
    seed: int = 42
    hf_token: str = ""  # To be set by user
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            quantization=QuantizationConfig(**config_dict.get('quantization', {})),
            lora=LoRAConfig(**config_dict.get('lora', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            seed=config_dict.get('seed', 42),
            hf_token=config_dict.get('hf_token', "")
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': self.model.__dict__,
            'quantization': self.quantization.__dict__,
            'lora': self.lora.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'seed': self.seed,
            'hf_token': self.hf_token
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

# Default configuration
def get_default_config() -> Config:
    """Get default configuration for histopathology fine-tuning"""
    return Config(
        model=ModelConfig(),
        quantization=QuantizationConfig(),
        lora=LoRAConfig(),
        training=TrainingConfig(),
        data=DataConfig()
    )
