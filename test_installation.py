"""
Quick test script to verify the installation and basic functionality
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'peft',
        'trl',
        'sklearn',
        'PIL',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'yaml'
    ]
    
    print("🔍 Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_torch_cuda():
    """Test PyTorch and CUDA availability"""
    import torch
    
    print("\n🔍 Testing PyTorch and CUDA...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1e9:.1f}GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
    else:
        print("⚠️  CUDA not available - training will be very slow!")

def test_config_loading():
    """Test configuration loading"""
    print("\n🔍 Testing configuration loading...")
    
    try:
        from config import Config, get_default_config
        
        # Test default config
        config = get_default_config()
        print("✅ Default configuration loaded")
        
        # Test YAML config if it exists
        if Path("config.yaml").exists():
            config = Config.from_yaml("config.yaml")
            print("✅ YAML configuration loaded")
        else:
            print("ℹ️  config.yaml not found, using defaults")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_data_utils():
    """Test data utilities"""
    print("\n🔍 Testing data utilities...")
    
    try:
        from data_utils import HistopathDataProcessor, create_example_dataset_structure
        from config import DataConfig
        
        # Create example dataset
        example_path = "./test_data"
        create_example_dataset_structure(example_path)
        print("✅ Example dataset structure created")
        
        # Test data processor
        data_config = DataConfig(data_path=example_path)
        processor = HistopathDataProcessor(data_config)
        
        dataset_info = processor.scan_dataset_directory(example_path)
        print("✅ Dataset scanning works")
        
        # Clean up
        import shutil
        shutil.rmtree(example_path, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Data utilities test failed: {e}")
        return False

def test_model_loading():
    """Test if we can load the processor (without the full model)"""
    print("\n🔍 Testing model components...")
    
    try:
        from transformers import AutoProcessor
        
        print("ℹ️  This will test if we can load the processor.")
        print("ℹ️  Full model loading requires HuggingFace authentication.")
        
        # This might fail without authentication, which is expected
        try:
            processor = AutoProcessor.from_pretrained("google/medgemma-4b-pt")
            print("✅ Model processor loaded successfully")
            return True
        except Exception as e:
            if "authentication" in str(e).lower() or "gated" in str(e).lower():
                print("ℹ️  Model access requires HuggingFace authentication (expected)")
                print("ℹ️  Run 'huggingface-cli login' with your token")
                return True
            else:
                print(f"❌ Unexpected error loading processor: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Model components test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 MedGemma Fine-tuning Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("PyTorch & CUDA", test_torch_cuda),
        ("Configuration", test_config_loading),
        ("Data Utilities", test_data_utils),
        ("Model Components", test_model_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your environment is ready for fine-tuning.")
        print("\n📋 Next steps:")
        print("1. Set up HuggingFace authentication: huggingface-cli login")
        print("2. Prepare your histopathology dataset")
        print("3. Update config.yaml with your settings")
        print("4. Run fine-tuning with run_finetuning.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("💡 Common fixes:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - Ensure CUDA is properly installed")
        print("   - Check your Python environment")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
