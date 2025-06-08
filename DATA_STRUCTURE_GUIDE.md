# Data Structure Guide for MedGemma Fine-tuning

This document provides detailed instructions on how to structure your histopathology dataset for successful fine-tuning with MedGemma. Following this guide ensures compatibility with the official implementation and optimal training results.

## 📁 Required Directory Structure

### Basic Structure
Your dataset must follow this exact directory structure:

```
your_dataset_name/
├── class_1/
│   ├── patient001_patch001.jpg
│   ├── patient001_patch002.jpg
│   ├── patient001_patch003.jpg
│   ├── patient002_patch001.jpg
│   ├── patient002_patch002.jpg
│   └── ...
├── class_2/
│   ├── patient003_patch001.jpg
│   ├── patient003_patch002.jpg
│   ├── patient004_patch001.jpg
│   └── ...
├── class_3/
│   └── ...
└── class_n/
    └── ...
```

### Official NCT-CRC-HE-100K Structure Example
For histopathology tissue classification (as used in the official notebook):

```
NCT-CRC-HE-100K/
├── adipose/
│   ├── patient001_patch001.tif
│   ├── patient001_patch002.tif
│   ├── patient002_patch001.tif
│   └── ...
├── background/
│   ├── patient003_patch001.tif
│   ├── patient003_patch002.tif
│   └── ...
├── debris/
│   └── ...
├── lymphocytes/
│   └── ...
├── mucus/
│   └── ...
├── smooth_muscle/
│   └── ...
├── normal_colon_mucosa/
│   └── ...
├── cancer_associated_stroma/
│   └── ...
└── colorectal_adenocarcinoma_epithelium/
    └── ...
```

## 🏷️ File Naming Convention

### Critical Requirements

1. **Patient-based naming**: Files must include patient identifiers to enable proper data splitting
2. **Consistent format**: Use a consistent naming pattern across all files
3. **Unique patient IDs**: Each patient should have a unique identifier

### Supported Naming Patterns

#### Pattern 1: PatientID_PatchID.extension (Recommended)
```
patient001_patch001.jpg
patient001_patch002.jpg
patient002_patch001.jpg
P123_001.png
P123_002.png
```

#### Pattern 2: PatientID_SlideID_PatchID.extension
```
patient001_slide01_patch001.tif
patient001_slide01_patch002.tif
patient001_slide02_patch001.tif
```

#### Pattern 3: CaseID_Additional_Info.extension
```
case_001_H&E_patch_01.jpg
case_001_H&E_patch_02.jpg
case_002_H&E_patch_01.jpg
```

### Patient ID Extraction Logic

The system automatically extracts patient IDs using these rules:

1. **Underscore separation**: Takes the first part before underscore
   - `patient123_patch001.jpg` → Patient ID: `patient123`
   - `P001_slide1_patch2.tif` → Patient ID: `P001`

2. **Alphanumeric pattern**: Extracts letters followed by numbers
   - `case456patch1.png` → Patient ID: `case456`

3. **Custom extraction**: You can modify the extraction logic in `data_utils.py`:

```python
def _extract_patient_id(self, filename: str) -> str:
    """
    Customize this function for your naming convention
    """
    # Example for custom pattern: "TCGA-XX-XXXX-XX_patch_N.jpg"
    if filename.startswith("TCGA"):
        return filename.split("_")[0]  # Returns "TCGA-XX-XXXX-XX"
    
    # Default pattern
    name_without_ext = Path(filename).stem
    if '_' in name_without_ext:
        return name_without_ext.split('_')[0]
    
    return name_without_ext
```

## 🖼️ Image Requirements

### Supported Formats
- **JPEG**: `.jpg`, `.jpeg`
- **PNG**: `.png`
- **TIFF**: `.tiff`, `.tif`
- **BMP**: `.bmp`

### Image Specifications
- **Color space**: RGB (images will be automatically converted)
- **Size**: Any size (will be processed by the model's image processor)
- **Quality**: High-quality images recommended for better results
- **Bit depth**: 8-bit or 16-bit (will be normalized)

### Recommended Specifications
- **Resolution**: 224x224 pixels or higher
- **Format**: TIFF for medical images (lossless compression)
- **Color**: RGB color images
- **File size**: < 50MB per image for efficient processing

## 📊 Dataset Size Guidelines

### Minimum Requirements
- **Classes**: At least 2 classes
- **Patients per class**: Minimum 5 patients per class
- **Patches per patient**: At least 1 patch per patient
- **Total images**: Minimum 100 images for meaningful training

### Recommended for Good Results
- **Patients per class**: 50+ patients per class
- **Patches per patient**: 5-20 patches per patient
- **Total images**: 1000+ images
- **Validation set**: 10-20% of total data

### Official Notebook Configuration
- **Training samples**: 9,000 images
- **Validation samples**: 1,000 images
- **Classes**: 9 tissue types
- **Format**: 224x224 pixel patches

## 🔧 Configuration Parameters

### Data Configuration in config.yaml

```yaml
data:
  data_path: "/path/to/your/dataset"           # Root directory path
  train_split: 0.7                             # 70% for training
  val_split: 0.15                              # 15% for validation  
  test_split: 0.15                             # 15% for testing
  max_patches_per_patient: 10                  # Limit patches per patient
  min_patches_per_patient: 1                   # Minimum patches required
  train_size: 9000                             # Max training samples
  validation_size: 1000                        # Max validation samples
  image_extensions:                            # Supported formats
    - ".jpg"
    - ".jpeg"
    - ".png"
    - ".tiff"
    - ".tif"
    - ".bmp"
```

### Key Parameters Explained

- **max_patches_per_patient**: Prevents memory issues and class imbalance
- **min_patches_per_patient**: Ensures data quality
- **train_size/validation_size**: Limits dataset size for faster training
- **Patient-based splitting**: Ensures no patient appears in multiple splits

## 🏥 Medical Dataset Considerations

### Patient Privacy
- **De-identification**: Ensure all patient identifiers are anonymized
- **HIPAA compliance**: Follow medical data privacy regulations
- **Ethical approval**: Obtain necessary institutional approvals

### Data Quality
- **Consistent staining**: Use consistent H&E staining protocols
- **Image quality**: Ensure good focus and proper exposure
- **Annotation accuracy**: Verify pathologist annotations
- **Artifact removal**: Remove or mark imaging artifacts

### Class Balance
- **Balanced distribution**: Aim for similar numbers of patients per class
- **Clinical relevance**: Include clinically relevant tissue types
- **Rare classes**: Consider data augmentation for rare tissue types

## 📝 Step-by-Step Setup Guide

### Step 1: Organize Your Data

1. Create the main dataset directory:
```bash
mkdir my_histopathology_dataset
cd my_histopathology_dataset
```

2. Create class subdirectories:
```bash
mkdir adenocarcinoma squamous_cell normal inflammatory
```

3. Copy images to appropriate class folders with proper naming:
```bash
# Example for adenocarcinoma class
cp /source/patient001_patch001.jpg adenocarcinoma/
cp /source/patient001_patch002.jpg adenocarcinoma/
cp /source/patient002_patch001.jpg adenocarcinoma/
```

### Step 2: Verify Data Structure

Use the provided verification script:

```python
from data_utils import HistopathDataProcessor
from config import DataConfig

# Initialize processor
config = DataConfig(data_path="./my_histopathology_dataset")
processor = HistopathDataProcessor(config)

# Scan and validate dataset
dataset_info = processor.scan_dataset_directory("./my_histopathology_dataset")

# Print summary
print(f"Total images: {dataset_info['total_images']}")
print(f"Total patients: {dataset_info['total_patients']}")
print(f"Number of classes: {dataset_info['num_classes']}")

for class_name, stats in dataset_info['class_stats'].items():
    print(f"{class_name}: {stats['num_images']} images, {stats['num_patients']} patients")
```

### Step 3: Test Data Loading

```python
# Test dataset creation
datasets, info = processor.process_dataset("./my_histopathology_dataset")

print("Dataset splits created:")
for split_name, dataset in datasets.items():
    print(f"  {split_name}: {len(dataset)} samples")

# Check a sample
sample = datasets["train"][0]
print(f"Sample keys: {sample.keys()}")
print(f"Image shape: {sample['image'].size}")
print(f"Messages format: {sample['messages']}")
```

## ⚠️ Common Issues and Solutions

### Issue 1: Patient ID Extraction Fails
**Problem**: Patients not properly separated between splits

**Solution**: Customize the `_extract_patient_id` function:
```python
def _extract_patient_id(self, filename: str) -> str:
    # Debug: Print filename to understand pattern
    print(f"Processing filename: {filename}")
    
    # Your custom logic here
    if "TCGA" in filename:
        return filename.split("_")[0]
    
    # Default fallback
    return filename.split("_")[0] if "_" in filename else filename.split(".")[0]
```

### Issue 2: Inconsistent Image Formats
**Problem**: Mixed image formats causing loading errors

**Solution**: Convert all images to a consistent format:
```bash
# Convert all images to JPEG
find . -name "*.tif" -exec convert {} {}.jpg \;
find . -name "*.png" -exec convert {} {}.jpg \;
```

### Issue 3: Class Imbalance
**Problem**: Uneven distribution of patients across classes

**Solution**: 
1. Balance patient numbers across classes
2. Use data augmentation for underrepresented classes
3. Adjust `max_patches_per_patient` per class

### Issue 4: Memory Issues
**Problem**: Too many large images causing memory errors

**Solution**:
1. Reduce `max_patches_per_patient`
2. Resize images to smaller dimensions
3. Use smaller batch sizes

## 📋 Validation Checklist

Before starting training, verify:

- [ ] Directory structure follows the required format
- [ ] All images are in supported formats
- [ ] File naming includes patient identifiers
- [ ] Patient IDs are correctly extracted
- [ ] Each class has sufficient patients (>5 recommended)
- [ ] No patient appears in multiple classes
- [ ] Images load correctly without errors
- [ ] Dataset splits are created successfully
- [ ] Configuration file points to correct data path

## 🔍 Example Datasets

### Small Test Dataset (for testing)
```
test_dataset/
├── class_a/
│   ├── patient001_patch001.jpg  # 3 patients
│   ├── patient001_patch002.jpg
│   ├── patient002_patch001.jpg
│   └── patient003_patch001.jpg
└── class_b/
    ├── patient004_patch001.jpg  # 3 patients
    ├── patient005_patch001.jpg
    └── patient006_patch001.jpg
```

### Production Dataset (recommended)
```
production_dataset/
├── adenocarcinoma/              # 100+ patients
├── squamous_cell_carcinoma/     # 100+ patients  
├── normal_tissue/               # 100+ patients
├── inflammatory/                # 50+ patients
└── necrosis/                    # 50+ patients
```

## 🚀 Quick Start Script

Create this script to set up your dataset structure:

```python
#!/usr/bin/env python3
"""
Quick setup script for MedGemma dataset structure
"""

import os
from pathlib import Path

def create_dataset_structure(base_path, class_names):
    """Create the required directory structure"""
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)
    
    for class_name in class_names:
        class_dir = base_path / class_name
        class_dir.mkdir(exist_ok=True)
        print(f"Created directory: {class_dir}")
    
    print(f"\nDataset structure created at: {base_path}")
    print("Now copy your images to the appropriate class directories")
    print("Remember to use patient-based naming: patientID_patchID.extension")

# Example usage
if __name__ == "__main__":
    # Define your class names
    classes = [
        "adenocarcinoma",
        "squamous_cell_carcinoma", 
        "normal_tissue",
        "inflammatory",
        "necrosis"
    ]
    
    # Create structure
    create_dataset_structure("./my_histopathology_dataset", classes)
```

## 📞 Support

If you encounter issues with data structure:

1. **Check the troubleshooting section** in this guide
2. **Verify your naming convention** matches supported patterns
3. **Test with a small subset** before processing the full dataset
4. **Review the example datasets** for reference

For additional help, refer to the main README.md or create an issue with:
- Your directory structure
- Sample filenames
- Error messages
- Dataset statistics

---

**Note**: This data structure guide is based on the official Google Health MedGemma implementation and ensures compatibility with the fine-tuning pipeline. 