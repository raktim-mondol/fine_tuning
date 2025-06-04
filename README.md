# MedGemma Fine-tuning for Histopathology Classification

This repository contains a complete, reproducible pipeline for fine-tuning Google's MedGemma-4B-PT model on histopathology data using LoRA (Low-Rank Adaptation). The implementation is specifically designed to handle multiple patches per patient scenarios common in histopathology datasets.

## 🎯 Features

- **Patient-based data splitting** to prevent data leakage
- **Multiple patches per patient** handling with configurable limits
- **LoRA fine-tuning** for memory-efficient training
- **Comprehensive evaluation** at both patch and patient levels
- **Reproducible configuration** system with YAML files
- **Automatic mixed precision** training support
- **Early stopping** and checkpoint management
- **Detailed logging** and visualization

## 📁 Project Structure

```
fine_tuning/
├── config.py              # Configuration classes and utilities
├── config.yaml            # Main configuration file
├── data_utils.py           # Data preprocessing and loading
├── medgemma_trainer.py     # Main training pipeline
├── evaluation.py           # Evaluation utilities
├── run_finetuning.py       # Main execution script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the repository
cd fine_tuning

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your histopathology data in the following structure:

```
your_dataset/
├── class1/
│   ├── patient001_patch001.jpg
│   ├── patient001_patch002.jpg
│   ├── patient002_patch001.jpg
│   └── ...
├── class2/
│   ├── patient003_patch001.jpg
│   ├── patient003_patch002.jpg
│   └── ...
└── class3/
    └── ...
```

**Important**: The code extracts patient IDs from filenames automatically. The default pattern assumes `patientID_patchID.extension` format. Modify the `_extract_patient_id()` function in `data_utils.py` if your naming convention is different.

### 3. Configuration

Edit `config.yaml` to match your setup:

```yaml
# Update these required fields:
data:
  data_path: "/path/to/your/histopath/dataset"  # Your dataset path

hf_token: "your_huggingface_token_here"  # Your HF token for model access
```

### 4. Run Fine-tuning

```bash
python run_finetuning.py \
    --data_path /path/to/your/dataset \
    --hf_token your_hf_token \
    --mode both \
    --output_dir ./results/medgemma-finetuned
```

## 🔧 Configuration Options

### Model Configuration
```yaml
model:
  model_id: "google/medgemma-4b-pt"
  torch_dtype: "bfloat16"  # bfloat16, float16, float32
  attn_implementation: "eager"
```

### LoRA Configuration
```yaml
lora:
  r: 16                    # Rank (8, 16, 32, 64)
  lora_alpha: 16           # Scaling parameter
  lora_dropout: 0.05       # Dropout rate
  target_modules: "all-linear"
```

### Training Configuration
```yaml
training:
  num_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2e-4
  max_seq_length: 512
```

### Data Configuration
```yaml
data:
  train_split: 0.7         # 70% for training
  val_split: 0.15          # 15% for validation
  test_split: 0.15         # 15% for testing
  max_patches_per_patient: 10  # Limit patches per patient
  min_patches_per_patient: 1   # Minimum required patches
```

## 🎮 Usage Examples

### Training Only
```bash
python run_finetuning.py \
    --data_path ./histopath_data \
    --hf_token your_token \
    --mode train \
    --epochs 5 \
    --batch_size 8 \
    --learning_rate 1e-4
```

### Evaluation Only
```bash
python run_finetuning.py \
    --data_path ./histopath_data \
    --hf_token your_token \
    --mode evaluate \
    --model_dir ./results/medgemma-finetuned \
    --eval_output_dir ./evaluation_results
```

### Custom Configuration
```bash
python run_finetuning.py \
    --config custom_config.yaml \
    --data_path ./histopath_data \
    --hf_token your_token \
    --lora_r 32 \
    --epochs 10
```

## 📊 Evaluation Features

The evaluation pipeline provides:

### Patch-level Metrics
- Accuracy, Precision, Recall, F1-score
- Per-class performance metrics
- Confusion matrix visualization

### Patient-level Metrics
- Aggregated predictions using majority voting
- Patient-level accuracy and F1-scores
- Handles multiple patches per patient

### Visualization
- Confusion matrices (patch and patient level)
- Performance plots
- Detailed classification reports

## 🔬 Key Features for Histopathology

### Patient-based Splitting
```python
# Ensures no patient appears in both train and test sets
def create_patient_based_splits(self):
    # Groups patients by class for stratified splitting
    # Prevents data leakage between splits
```

### Multi-patch Handling
```python
# Configurable patch limits per patient
max_patches_per_patient: 10  # Memory management
min_patches_per_patient: 1   # Quality control
```

### Flexible Patient ID Extraction
```python
def _extract_patient_id(self, filename: str) -> str:
    # Supports various naming conventions:
    # "patient123_patch001.jpg" -> "patient123"
    # "P123_001.png" -> "P123"
    # "case_456_slide_1_patch_2.tif" -> "case_456"
```

## 💾 Output Files

After training and evaluation, you'll find:

```
results/
├── medgemma-finetuned/
│   ├── adapter_config.json        # LoRA configuration
│   ├── adapter_model.safetensors  # Fine-tuned weights
│   ├── training_metadata.json     # Training information
│   ├── config.yaml               # Used configuration
│   └── logs/                     # Training logs
└── evaluation_results/
    ├── evaluation_report.json     # Detailed metrics
    ├── evaluation_summary.txt     # Human-readable summary
    ├── confusion_matrix_patch.png # Patch-level confusion matrix
    └── confusion_matrix_patient.png # Patient-level confusion matrix
```

## 🚨 Requirements

- **GPU**: CUDA-compatible GPU with ≥8GB VRAM
- **Python**: 3.8+
- **CUDA**: 11.7+ (for optimal performance)
- **HuggingFace Token**: Required for MedGemma model access

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```yaml
   # Reduce batch size or max patches per patient
   per_device_train_batch_size: 2
   max_patches_per_patient: 5
   ```

2. **Model Access Issues**:
   ```bash
   # Ensure you have access to MedGemma model
   huggingface-cli login
   ```

3. **Data Path Issues**:
   ```python
   # Check data structure matches expected format
   # Verify image file extensions are supported
   ```

## 📈 Performance Tips

1. **For limited GPU memory**:
   - Use smaller batch sizes
   - Enable gradient checkpointing
   - Reduce max_patches_per_patient

2. **For faster training**:
   - Use larger batch sizes
   - Increase gradient_accumulation_steps
   - Use bfloat16 precision

3. **For better convergence**:
   - Adjust learning rate
   - Increase LoRA rank
   - Add more training epochs

## 🤝 Contributing

Feel free to:
- Report bugs or issues
- Suggest improvements
- Add support for new data formats
- Extend evaluation metrics

## 📄 License

This project follows the same license as the underlying MedGemma model. Please refer to Google's MedGemma licensing terms.

## 🙏 Acknowledgments

- Google Health AI for the MedGemma model
- Hugging Face for the transformers library
- Microsoft for the LoRA implementation

## 📚 References

- [MedGemma Model](https://huggingface.co/google/medgemma-4b-pt)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Transformers Library](https://huggingface.co/docs/transformers/)

---

For questions or support, please check the documentation or create an issue in the repository.