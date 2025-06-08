# MedGemma Fine-tuning for Histopathology Classification

This repository provides a complete implementation for fine-tuning Google's MedGemma-4B-IT model on histopathology images, based on the **official Google Health MedGemma implementation**. The codebase supports tissue classification tasks using LoRA (Low-Rank Adaptation) with 4-bit quantization for efficient training.

## 🔥 Key Features

- **Official Implementation**: Based on Google Health's official MedGemma fine-tuning notebook
- **4-bit Quantization**: Memory-efficient training using QLoRA
- **Conversational Format**: Uses proper message format for instruction-tuned models
- **Patient-level Splitting**: Prevents data leakage in medical datasets
- **Comprehensive Evaluation**: Includes baseline comparison and detailed metrics
- **Production Ready**: Complete pipeline from training to inference

## 📋 Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with compute capability 8.0+ (A100, RTX 3090, RTX 4090, etc.)
- **Memory**: At least 40GB GPU memory recommended
- **Storage**: Sufficient space for dataset and model checkpoints

### Software Requirements
- Python 3.8+
- CUDA 11.8+ or 12.0+
- PyTorch 2.0+

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd medgemma_fine_tuning

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_installation.py
```

### 2. Setup HuggingFace Access

1. Create a HuggingFace account at [huggingface.co](https://huggingface.co)
2. Request access to MedGemma at [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
3. Generate a write access token at [HuggingFace Settings](https://huggingface.co/settings/tokens)
4. Set your token in `config.yaml` or as environment variable:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

### 3. Prepare Your Dataset

Organize your histopathology images in the following structure:

```
your_dataset/
├── adipose/
│   ├── patient001_patch001.jpg
│   ├── patient001_patch002.jpg
│   └── ...
├── background/
│   ├── patient002_patch001.jpg
│   └── ...
├── debris/
├── lymphocytes/
├── mucus/
├── smooth_muscle/
├── normal_colon_mucosa/
├── cancer_associated_stroma/
└── colorectal_adenocarcinoma_epithelium/
```

**Important**: Use patient-based naming (e.g., `patientID_patchID.extension`) to enable proper data splitting.

### 4. Configure Training

Edit `config.yaml` to set your dataset path:

```yaml
data:
  data_path: "/path/to/your/dataset"  # Set this to your dataset path

# Optional: Adjust training parameters
training:
  num_train_epochs: 1                # Number of epochs
  per_device_train_batch_size: 4     # Batch size per GPU
  learning_rate: 2e-4                # Learning rate
```

### 5. Run Training

```bash
# Basic training
python run_finetuning.py --config config.yaml

# With custom parameters
python run_finetuning.py \
  --config config.yaml \
  --data_path /path/to/your/dataset \
  --epochs 3 \
  --batch_size 4
```

### 6. Run Inference

```bash
# Single image
python inference.py \
  --model_path ./medgemma-4b-it-sft-lora-histopath \
  --image_path /path/to/image.jpg

# Batch inference on directory
python inference.py \
  --model_path ./medgemma-4b-it-sft-lora-histopath \
  --directory_path /path/to/images/ \
  --batch_size 8
```

## 📊 Evaluation

### Comprehensive Evaluation

```python
from evaluation import MedGemmaEvaluator

# Initialize evaluator
evaluator = MedGemmaEvaluator("./medgemma-4b-it-sft-lora-histopath")

# Generate complete evaluation report
evaluator.generate_evaluation_report(
    test_dataset,
    output_dir="./evaluation_results",
    include_baseline_comparison=True
)
```

### Baseline Comparison

The evaluation automatically compares your fine-tuned model with the baseline MedGemma-4B-IT model:

```
📈 Comparison Results:
   Fine-tuned - Accuracy: 0.945, F1: 0.944
   Baseline - Accuracy: 0.430, F1: 0.352
   Improvement - Accuracy: 0.515, F1: 0.592
```

## 🏗️ Architecture Overview

### Model Configuration
- **Base Model**: `google/medgemma-4b-it` (instruction-tuned version)
- **Quantization**: 4-bit with NF4 quantization type
- **LoRA**: Rank 16, Alpha 16, targeting all linear layers
- **Precision**: bfloat16 for training and inference

### Training Pipeline
1. **Data Loading**: Patient-based splitting to prevent leakage
2. **Preprocessing**: Conversational format with tissue classification prompt
3. **Model Setup**: Load with quantization and apply LoRA adapters
4. **Training**: SFT (Supervised Fine-Tuning) with custom collate function
5. **Evaluation**: Comprehensive metrics and baseline comparison

### Key Components

```
├── config.py              # Configuration management
├── config.yaml           # Training configuration
├── medgemma_trainer.py    # Main training pipeline
├── data_utils.py          # Data processing utilities
├── evaluation.py          # Evaluation and metrics
├── inference.py           # Inference pipeline
└── run_finetuning.py     # Main execution script
```

## 🔧 Configuration

### Model Configuration
```yaml
model:
  model_id: "google/medgemma-4b-it"
  torch_dtype: "bfloat16"
  attn_implementation: "eager"
```

### Quantization Configuration
```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"
```

### LoRA Configuration
```yaml
lora:
  r: 16
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: "all-linear"
  modules_to_save:
    - "lm_head"
    - "embed_tokens"
```

### Training Configuration
```yaml
training:
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  max_grad_norm: 0.3
  warmup_ratio: 0.03
  lr_scheduler_type: "linear"
```

## 📈 Performance

Based on the official implementation results:

| Metric | Baseline (MedGemma-4B-IT) | Fine-tuned | Improvement |
|--------|---------------------------|------------|-------------|
| Accuracy | 0.430 | 0.945 | +0.515 |
| F1 Score | 0.352 | 0.944 | +0.592 |

*Results on NCT-CRC-HE-100K histopathology dataset*

## 🛠️ Advanced Usage

### Custom Tissue Classes

Modify `data_utils.py` to define your own tissue classes:

```python
TISSUE_CLASSES = [
    "A: your_class_1",
    "B: your_class_2",
    # ... add your classes
]
```

### Custom Data Processing

Extend `HistopathDataProcessor` for custom data formats:

```python
class CustomDataProcessor(HistopathDataProcessor):
    def _extract_patient_id(self, filename: str) -> str:
        # Implement your patient ID extraction logic
        return patient_id
```

### Training Monitoring

Monitor training with TensorBoard:

```bash
tensorboard --logdir ./medgemma-4b-it-sft-lora-histopath/logs
```

## 🐛 Troubleshooting

### Common Issues

1. **GPU Memory Error**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable `gradient_checkpointing`

2. **HuggingFace Access Error**
   - Ensure you have access to MedGemma model
   - Verify your HF token has write permissions
   - Check token is correctly set

3. **Data Loading Error**
   - Verify dataset structure matches expected format
   - Check image file extensions are supported
   - Ensure patient ID extraction works correctly

### Performance Optimization

1. **Memory Optimization**
   ```yaml
   training:
     gradient_checkpointing: true
     dataloader_num_workers: 0  # Reduce if memory issues
   ```

2. **Speed Optimization**
   ```yaml
   training:
     optim: "adamw_torch_fused"  # Faster optimizer
     bf16: true                  # Use bfloat16
   ```

## 📚 References

- [MedGemma Official Repository](https://github.com/google-health/medgemma)
- [MedGemma Paper](https://arxiv.org/abs/2404.18814)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [NCT-CRC-HE-100K Dataset](https://zenodo.org/records/1214456)

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information

---

**Note**: This implementation is based on the official Google Health MedGemma fine-tuning notebook and follows the same architecture and training procedures for optimal compatibility and performance.