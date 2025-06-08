#!/bin/bash

# MedGemma Fine-tuning Setup Script for WSL/Linux
# This script sets up the environment and prepares for fine-tuning

echo "🏥 Setting up MedGemma Fine-tuning Environment"
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Installing pip..."
    sudo apt update
    sudo apt install python3-pip -y
fi

echo "✅ pip3 found: $(pip3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📥 Installing Python dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed successfully"
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Check if CUDA is available
echo "🔍 Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python3 -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name()}')"
    python3 -c "import torch; print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"
else
    echo "⚠️  CUDA not available. GPU training will not work."
fi

# Create example directory structure
echo "📁 Creating example dataset structure..."
mkdir -p example_data/{adenocarcinoma,squamous_cell,normal,inflammatory}

echo "📝 Creating example files..."
for class in adenocarcinoma squamous_cell normal inflammatory; do
    for patient in {001..003}; do
        for patch in {01..05}; do
            touch "example_data/${class}/patient_${patient}_patch_${patch}.jpg"
        done
    done
done

echo "✅ Example dataset structure created in example_data/"

# Check HuggingFace CLI
if ! command -v huggingface-cli &> /dev/null; then
    echo "📥 Installing HuggingFace CLI..."
    pip install huggingface_hub[cli]
fi

echo "✅ Environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Set up your HuggingFace token: huggingface-cli login"
echo "3. Prepare your histopathology dataset"
echo "4. Update config.yaml with your data path and settings"
echo "5. Run fine-tuning: python run_finetuning.py --data_path your_data --hf_token your_token"
echo ""
echo "🔧 For help, see README.md or run: python run_finetuning.py --help"
