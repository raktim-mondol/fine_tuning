#!/usr/bin/env python3
"""
Example Dataset Creator for MedGemma Fine-tuning

This script creates an example dataset structure with dummy data
to help researchers understand the required format.

Usage:
    python create_example_dataset.py --output_dir ./example_dataset
    python create_example_dataset.py --output_dir ./example_dataset --classes adenocarcinoma normal inflammatory
"""

import argparse
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_dummy_image(size=(224, 224), text="Sample", color_scheme="random"):
    """
    Create a dummy histopathology-like image
    
    Args:
        size: Image dimensions
        text: Text to overlay
        color_scheme: Color scheme for the image
    
    Returns:
        PIL Image
    """
    # Create base image with tissue-like colors
    if color_scheme == "pink":  # H&E staining - nuclei
        base_color = (255, 182, 193)  # Light pink
        accent_color = (139, 69, 19)  # Brown for nuclei
    elif color_scheme == "purple":  # H&E staining - cytoplasm
        base_color = (221, 160, 221)  # Plum
        accent_color = (75, 0, 130)  # Indigo
    elif color_scheme == "blue":  # Special staining
        base_color = (173, 216, 230)  # Light blue
        accent_color = (0, 0, 139)  # Dark blue
    else:  # Random tissue-like colors
        base_colors = [
            (255, 182, 193),  # Light pink
            (221, 160, 221),  # Plum
            (173, 216, 230),  # Light blue
            (240, 230, 140),  # Khaki
            (255, 218, 185),  # Peach
        ]
        base_color = random.choice(base_colors)
        accent_color = tuple(max(0, c - 100) for c in base_color)
    
    # Create image
    img = Image.new('RGB', size, base_color)
    draw = ImageDraw.Draw(img)
    
    # Add some texture (simulating tissue structures)
    for _ in range(random.randint(20, 50)):
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        radius = random.randint(2, 8)
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                    fill=accent_color, outline=None)
    
    # Add some lines (simulating fibers)
    for _ in range(random.randint(5, 15)):
        x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
        x2, y2 = random.randint(0, size[0]), random.randint(0, size[1])
        draw.line([x1, y1, x2, y2], fill=accent_color, width=random.randint(1, 3))
    
    # Add text overlay
    try:
        # Try to use a default font
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Position text in bottom right
        text_x = size[0] - text_width - 10
        text_y = size[1] - text_height - 10
        
        # Add background for text
        draw.rectangle([text_x-2, text_y-2, text_x+text_width+2, text_y+text_height+2], 
                      fill=(255, 255, 255, 128))
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    except:
        # Fallback if font loading fails
        draw.text((10, 10), text, fill=(0, 0, 0))
    
    return img


def create_example_dataset(output_dir, class_names, num_patients_per_class=10, patches_per_patient=5):
    """
    Create example dataset with proper structure
    
    Args:
        output_dir: Output directory path
        class_names: List of class names
        num_patients_per_class: Number of patients per class
        patches_per_patient: Number of patches per patient
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🏗️ Creating example dataset at: {output_dir}")
    print(f"📊 Configuration:")
    print(f"   Classes: {len(class_names)}")
    print(f"   Patients per class: {num_patients_per_class}")
    print(f"   Patches per patient: {patches_per_patient}")
    print(f"   Total images: {len(class_names) * num_patients_per_class * patches_per_patient}")
    
    # Color schemes for different classes
    color_schemes = ["pink", "purple", "blue", "random"]
    
    total_images = 0
    
    for class_idx, class_name in enumerate(class_names):
        print(f"\n📁 Creating class: {class_name}")
        
        # Create class directory
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Select color scheme for this class
        color_scheme = color_schemes[class_idx % len(color_schemes)]
        
        for patient_idx in range(1, num_patients_per_class + 1):
            patient_id = f"patient{patient_idx:03d}"
            
            for patch_idx in range(1, patches_per_patient + 1):
                # Create filename following the convention
                filename = f"{patient_id}_patch{patch_idx:03d}.jpg"
                filepath = class_dir / filename
                
                # Create dummy image
                image_text = f"{class_name[:8]}\n{patient_id}\nP{patch_idx}"
                img = create_dummy_image(
                    size=(224, 224),
                    text=image_text,
                    color_scheme=color_scheme
                )
                
                # Save image
                img.save(filepath, "JPEG", quality=95)
                total_images += 1
                
                if patch_idx == 1:  # Print progress for first patch of each patient
                    print(f"   Created {patient_id}: {patches_per_patient} patches")
    
    print(f"\n✅ Example dataset created successfully!")
    print(f"📊 Summary:")
    print(f"   Total images: {total_images}")
    print(f"   Directory structure:")
    
    # Print directory tree
    for class_name in class_names:
        class_dir = output_dir / class_name
        image_count = len(list(class_dir.glob("*.jpg")))
        print(f"     {class_name}/: {image_count} images")
    
    # Create a README file
    readme_content = f"""# Example Dataset for MedGemma Fine-tuning

This is an example dataset created for testing the MedGemma fine-tuning pipeline.

## Dataset Information
- **Classes**: {len(class_names)} ({', '.join(class_names)})
- **Patients per class**: {num_patients_per_class}
- **Patches per patient**: {patches_per_patient}
- **Total images**: {total_images}
- **Image format**: JPEG (224x224 pixels)

## File Naming Convention
Files follow the pattern: `patientID_patchID.jpg`
- Example: `patient001_patch001.jpg`
- Patient IDs: patient001 to patient{num_patients_per_class:03d}
- Patch IDs: patch001 to patch{patches_per_patient:03d}

## Usage
This dataset can be used to test the MedGemma fine-tuning pipeline:

```bash
# Validate the dataset
python validate_dataset.py --data_path {output_dir}

# Run training
python run_finetuning.py --config config.yaml --data_path {output_dir}
```

## Note
This is synthetic data created for testing purposes only.
Replace with real histopathology data for actual research.
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\n📄 README.md created at: {readme_path}")
    
    # Create a sample config file
    config_content = f"""# Sample configuration for example dataset
data:
  data_path: "{output_dir}"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  max_patches_per_patient: {patches_per_patient}
  min_patches_per_patient: 1
  train_size: {int(total_images * 0.7)}
  validation_size: {int(total_images * 0.15)}

# Use default settings for other parameters
model:
  model_id: "google/medgemma-4b-it"

training:
  num_train_epochs: 1  # Reduced for testing
  per_device_train_batch_size: 2  # Small for testing

hf_token: ""  # Add your HuggingFace token here
"""
    
    config_path = output_dir / "sample_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"⚙️ Sample config created at: {config_path}")
    
    print(f"\n🚀 Next steps:")
    print(f"1. Validate the dataset:")
    print(f"   python validate_dataset.py --data_path {output_dir}")
    print(f"2. Test training (after setting HF token):")
    print(f"   python run_finetuning.py --config {config_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Create example dataset for MedGemma fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create basic example with default classes
  python create_example_dataset.py --output_dir ./example_dataset
  
  # Create with custom classes
  python create_example_dataset.py --output_dir ./my_example \\
    --classes adenocarcinoma squamous_cell normal inflammatory
  
  # Create larger dataset
  python create_example_dataset.py --output_dir ./large_example \\
    --patients 20 --patches 10
        """
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the example dataset"
    )
    
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["adenocarcinoma", "normal", "inflammatory"],
        help="List of class names (default: adenocarcinoma normal inflammatory)"
    )
    
    parser.add_argument(
        "--patients",
        type=int,
        default=10,
        help="Number of patients per class (default: 10)"
    )
    
    parser.add_argument(
        "--patches",
        type=int,
        default=5,
        help="Number of patches per patient (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.patients < 1:
        print("❌ Number of patients must be at least 1")
        return 1
    
    if args.patches < 1:
        print("❌ Number of patches must be at least 1")
        return 1
    
    if len(args.classes) < 2:
        print("❌ At least 2 classes are required")
        return 1
    
    # Create example dataset
    try:
        create_example_dataset(
            output_dir=args.output_dir,
            class_names=args.classes,
            num_patients_per_class=args.patients,
            patches_per_patient=args.patches
        )
        return 0
    except Exception as e:
        print(f"❌ Error creating example dataset: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 