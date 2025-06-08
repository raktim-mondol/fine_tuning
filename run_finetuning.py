#!/usr/bin/env python3
"""
Main script to run MedGemma fine-tuning pipeline
Based on official Google Health MedGemma implementation
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config
from medgemma_trainer import MedGemmaFineTuner


def validate_config(config: Config) -> bool:
    """
    Validate configuration before starting training
    
    Args:
        config: Configuration object
        
    Returns:
        True if valid, False otherwise
    """
    errors = []
    
    # Check data path
    if not config.data.data_path:
        errors.append("❌ data.data_path is required")
    elif not Path(config.data.data_path).exists():
        errors.append(f"❌ Data path does not exist: {config.data.data_path}")
    
    # Check HuggingFace token
    if not config.hf_token and not os.getenv("HF_TOKEN"):
        errors.append("⚠️ No HuggingFace token provided. Set hf_token in config or HF_TOKEN environment variable")
    
    # Check output directory
    output_dir = Path(config.training.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"⚠️ Output directory {output_dir} already exists and is not empty")
        print("   Training will continue from existing checkpoint if available")
    
    # Check GPU requirements
    import torch
    if not torch.cuda.is_available():
        errors.append("❌ CUDA is not available. GPU is required for training")
    else:
        device_capability = torch.cuda.get_device_capability()
        if device_capability[0] < 8:
            errors.append("❌ GPU does not support bfloat16. Compute capability 8.0+ required")
    
    # Print errors
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  {error}")
        return False
    
    return True


def print_config_summary(config: Config):
    """Print a summary of the configuration"""
    print("\n" + "="*60)
    print("MEDGEMMA FINE-TUNING CONFIGURATION")
    print("="*60)
    
    print(f"Model:")
    print(f"  ID: {config.model.model_id}")
    print(f"  Precision: {config.model.torch_dtype}")
    print(f"  Quantization: 4-bit ({'enabled' if config.quantization.load_in_4bit else 'disabled'})")
    
    print(f"\nLoRA:")
    print(f"  Rank: {config.lora.r}")
    print(f"  Alpha: {config.lora.lora_alpha}")
    print(f"  Dropout: {config.lora.lora_dropout}")
    print(f"  Target modules: {config.lora.target_modules}")
    
    print(f"\nTraining:")
    print(f"  Output directory: {config.training.output_dir}")
    print(f"  Epochs: {config.training.num_train_epochs}")
    print(f"  Batch size: {config.training.per_device_train_batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    
    print(f"\nData:")
    print(f"  Path: {config.data.data_path}")
    print(f"  Train size: {config.data.train_size}")
    print(f"  Validation size: {config.data.validation_size}")
    print(f"  Splits: {config.data.train_split:.1f}/{config.data.val_split:.1f}/{config.data.test_split:.1f}")
    
    print("="*60 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Fine-tune MedGemma for histopathology classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with config file
  python run_finetuning.py --config config.yaml
  
  # Override data path
  python run_finetuning.py --config config.yaml --data_path /path/to/data
  
  # Override output directory
  python run_finetuning.py --config config.yaml --output_dir ./my_model
  
  # Set HuggingFace token
  python run_finetuning.py --config config.yaml --hf_token your_token_here
  
  # Quick test run with fewer epochs
  python run_finetuning.py --config config.yaml --epochs 1
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        help="Override data path from config"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        help="HuggingFace token (can also use HF_TOKEN environment variable)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override training batch size"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Override learning rate"
    )
    
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate configuration without training"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint if available"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Create a config.yaml file or specify --config path")
        return 1
    
    try:
        config = Config.from_yaml(str(config_path))
        print(f"✅ Loaded configuration from {config_path}")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1
    
    # Apply command line overrides
    if args.data_path:
        config.data.data_path = args.data_path
        print(f"🔄 Override data_path: {args.data_path}")
    
    if args.output_dir:
        config.training.output_dir = args.output_dir
        print(f"🔄 Override output_dir: {args.output_dir}")
    
    if args.hf_token:
        config.hf_token = args.hf_token
        print(f"🔄 Override HuggingFace token")
    elif os.getenv("HF_TOKEN"):
        config.hf_token = os.getenv("HF_TOKEN")
        print(f"🔄 Using HF_TOKEN from environment")
    
    if args.epochs:
        config.training.num_train_epochs = args.epochs
        print(f"🔄 Override epochs: {args.epochs}")
    
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
        config.training.per_device_eval_batch_size = args.batch_size
        print(f"🔄 Override batch_size: {args.batch_size}")
    
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
        print(f"🔄 Override learning_rate: {args.learning_rate}")
    
    # Print configuration summary
    print_config_summary(config)
    
    # Validate configuration
    if not validate_config(config):
        return 1
    
    if args.validate_only:
        print("✅ Configuration validation passed")
        return 0
    
    # Create output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the final configuration
    config_save_path = output_dir / "config.yaml"
    config.to_yaml(str(config_save_path))
    print(f"💾 Configuration saved to {config_save_path}")
    
    try:
        # Initialize fine-tuner
        print("🚀 Initializing MedGemma fine-tuner...")
        fine_tuner = MedGemmaFineTuner(config)
        
        # Run training
        print("🎯 Starting fine-tuning pipeline...")
        trainer = fine_tuner.run_complete_pipeline()
        
        print("\n🎉 Fine-tuning completed successfully!")
        print(f"📁 Model saved to: {config.training.output_dir}")
        print(f"📊 You can now use the model for inference or evaluation")
        
        # Print next steps
        print("\n📋 Next Steps:")
        print(f"  1. Evaluate the model:")
        print(f"     python -c \"from evaluation import MedGemmaEvaluator; evaluator = MedGemmaEvaluator('{config.training.output_dir}'); evaluator.generate_evaluation_report(test_dataset)\"")
        print(f"  2. Run inference:")
        print(f"     python inference.py --model_path {config.training.output_dir} --image_path your_image.jpg")
        print(f"  3. Check training logs:")
        print(f"     tensorboard --logdir {config.training.output_dir}/logs")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
