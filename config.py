"""
Configuration file for MedGemma histopathology fine-tuning
"""

import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    model_id: str = "google/medgemma-4b-pt"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    device_map: str = "auto"
    trust_remote_code: bool = True

@dataclass
class LoRAConfig:
    """LoRA configuration parameters"""
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    output_dir: str = "./medgemma-histpath-finetuned"
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    bf16: bool = True
    dataloader_num_workers: int = 4
    max_seq_length: int = 512
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.01

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
    
    def __post_init__(self):
        if self.image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']

@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig
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
        lora=LoRAConfig(),
        training=TrainingConfig(),
        data=DataConfig()
    )
