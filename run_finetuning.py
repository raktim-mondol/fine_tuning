#!/usr/bin/env python3
"""
Main script to run MedGemma fine-tuning for histopathology classification
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path to import local modules
sys.path.append(str(Path(__file__).parent))

from config import Config
from medgemma_trainer import MedGemmaFineTuner
from evaluation import MedGemmaEvaluator


def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune MedGemma for histopathology classification"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to histopathology dataset directory"
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="HuggingFace API token for model access"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./medgemma-histpath-finetuned",
        help="Output directory for fine-tuned model"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "both"],
        default="both",
        help="Mode: train, evaluate, or both"
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory containing fine-tuned model (for evaluation mode)"
    )
    
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    
    # Training parameters (can override config)
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--lora_r", type=int, help="LoRA rank")
    
    return parser.parse_args()


def load_and_update_config(args):
    """Load configuration and update with command line arguments"""
    # Load base config
    if Path(args.config).exists():
        config = Config.from_yaml(args.config)
        print(f"✅ Loaded configuration from {args.config}")
    else:
        print(f"⚠️ Config file {args.config} not found, using default configuration")
        from config import get_default_config
        config = get_default_config()
    
    # Update with command line arguments
    config.data.data_path = args.data_path
    config.hf_token = args.hf_token
    config.training.output_dir = args.output_dir
    
    # Override training parameters if provided
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
        config.training.per_device_eval_batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.lora_r:
        config.lora.r = args.lora_r
    
    return config


def validate_data_path(data_path: str):
    """Validate that the data path exists and has the expected structure"""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    if not data_path.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_path}")
    
    # Check for class directories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in: {data_path}")
    
    print(f"✅ Data validation passed: {len(class_dirs)} class directories found")
    for class_dir in class_dirs:
        image_files = [f for f in class_dir.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']]
        print(f"   {class_dir.name}: {len(image_files)} images")


def train_model(config: Config):
    """Train the MedGemma model"""
    print("🚀 Starting training...")
    
    # Initialize fine-tuner
    fine_tuner = MedGemmaFineTuner(config)
    
    # Run complete training pipeline
    trainer = fine_tuner.run_complete_pipeline()
    
    print("✅ Training completed successfully!")
    return trainer


def evaluate_model(model_dir: str, 
                  data_path: str, 
                  eval_output_dir: str,
                  config: Config):
    """Evaluate the fine-tuned model"""
    print("📊 Starting evaluation...")
    
    # Initialize evaluator
    evaluator = MedGemmaEvaluator(model_dir)
    
    # Prepare test dataset
    from data_utils import HistopathDataProcessor
    data_processor = HistopathDataProcessor(config.data)
    datasets, _ = data_processor.process_dataset(data_path)
    
    if "test" not in datasets:
        print("⚠️ No test split found, using validation split for evaluation")
        test_dataset = datasets.get("validation", datasets.get("val"))
    else:
        test_dataset = datasets["test"]
    
    if test_dataset is None:
        raise ValueError("No test or validation dataset available for evaluation")
    
    # Generate evaluation report
    results = evaluator.generate_evaluation_report(test_dataset, eval_output_dir)
    
    print("✅ Evaluation completed successfully!")
    return results


def main():
    """Main function"""
    print("🏥 MedGemma Histopathology Fine-tuning Pipeline")
    print("=" * 50)
    
    # Parse arguments
    args = setup_args()
    
    # Load and validate configuration
    config = load_and_update_config(args)
    
    # Validate data path
    validate_data_path(args.data_path)
    
    # Run based on mode
    if args.mode in ["train", "both"]:
        # Training
        trainer = train_model(config)
        
        # Save final config
        config.to_yaml(str(Path(config.training.output_dir) / "final_config.yaml"))
        
        if args.mode == "train":
            print("🎉 Training pipeline completed!")
            return
    
    if args.mode in ["evaluate", "both"]:
        # Evaluation
        model_dir = args.model_dir if args.model_dir else config.training.output_dir
        
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        results = evaluate_model(
            model_dir=model_dir,
            data_path=args.data_path,
            eval_output_dir=args.eval_output_dir,
            config=config
        )
    
    print("🎉 Complete pipeline finished successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
