"""
Data preprocessing and loading utilities for histopathology images
Based on official Google Health MedGemma implementation
Handles multiple patches per patient scenario with conversational format
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import torch
from datasets import Dataset, DatasetDict

from config import DataConfig

# Tissue classification classes (can be customized based on your dataset)
TISSUE_CLASSES = [
    "A: adipose",
    "B: background", 
    "C: debris",
    "D: lymphocytes",
    "E: mucus",
    "F: smooth muscle",
    "G: normal colon mucosa",
    "H: cancer-associated stroma",
    "I: colorectal adenocarcinoma epithelium"
]

# Create the classification prompt
options = "\n".join(TISSUE_CLASSES)
CLASSIFICATION_PROMPT = f"What is the most likely tissue type shown in the histopathology image?\n{options}"


class HistopathDataProcessor:
    """
    Data processor for histopathology images with multiple patches per patient
    Uses conversational format compatible with MedGemma
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.image_extensions = set(config.image_extensions)
        self.patient_data = defaultdict(list)
        self.label_mapping = {}
        self.tissue_classes = TISSUE_CLASSES  # Can be customized
        
    def scan_dataset_directory(self, data_path: str) -> Dict[str, Any]:
        """
        Scan dataset directory to understand structure and collect metadata
        
        Expected structure:
        data_path/
        ├── class1/
        │   ├── patient1_patch1.jpg
        │   ├── patient1_patch2.jpg
        │   ├── patient2_patch1.jpg
        │   └── ...
        ├── class2/
        │   ├── patient3_patch1.jpg
        │   └── ...
        
        Args:
            data_path: Root directory containing class folders
            
        Returns:
            Dictionary with dataset statistics
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        print(f"📁 Scanning dataset directory: {data_path}")
        
        # Collect data by class and patient
        class_stats = {}
        total_images = 0
        total_patients = 0
        
        for class_dir in data_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            class_images = []
            class_patients = set()
            
            # Find all image files
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in self.image_extensions:
                    # Extract patient ID from filename
                    # Assumes format: patientID_patchID.ext or similar
                    patient_id = self._extract_patient_id(img_file.name)
                    
                    class_images.append({
                        'image_path': str(img_file),
                        'patient_id': patient_id,
                        'class': class_name
                    })
                    
                    class_patients.add(patient_id)
                    self.patient_data[patient_id].append({
                        'image_path': str(img_file),
                        'class': class_name
                    })
            
            class_stats[class_name] = {
                'num_images': len(class_images),
                'num_patients': len(class_patients),
                'patients': list(class_patients)
            }
            
            total_images += len(class_images)
            total_patients += len(class_patients)
            
            print(f"   {class_name}: {len(class_images)} images, {len(class_patients)} patients")
        
        # Create label mapping - map class names to tissue class indices
        self.label_mapping = self._create_label_mapping(list(class_stats.keys()))
        
        dataset_info = {
            'total_images': total_images,
            'total_patients': total_patients,
            'num_classes': len(class_stats),
            'class_stats': class_stats,
            'label_mapping': self.label_mapping,
            'tissue_classes': self.tissue_classes
        }
        
        print(f"✅ Dataset scan complete: {total_images} images, {total_patients} patients, {len(class_stats)} classes")
        
        return dataset_info
    
    def _create_label_mapping(self, class_names: List[str]) -> Dict[str, int]:
        """
        Create mapping from class names to tissue class indices
        You may need to customize this based on your dataset's class names
        """
        # Simple mapping - you should customize this based on your actual class names
        # For now, we'll create a direct mapping
        return {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
    
    def _extract_patient_id(self, filename: str) -> str:
        """
        Extract patient ID from filename
        
        This is a simple implementation - modify based on your naming convention
        Examples:
        - "patient123_patch001.jpg" -> "patient123"
        - "P123_001.png" -> "P123"
        - "case_456_slide_1_patch_2.tif" -> "case_456"
        """
        # Remove extension
        name_without_ext = Path(filename).stem
        
        # Common patterns to extract patient ID
        # Pattern 1: patient_id followed by underscore
        if '_' in name_without_ext:
            parts = name_without_ext.split('_')
            # Take first part as patient ID
            return parts[0]
        
        # Pattern 2: If no underscore, use first part before any numbers at the end
        import re
        match = re.match(r'^([A-Za-z]+\d*)', name_without_ext)
        if match:
            return match.group(1)
        
        # Fallback: use entire filename without extension
        return name_without_ext
    
    def create_patient_based_splits(self, 
                                  train_ratio: float = None,
                                  val_ratio: float = None,
                                  test_ratio: float = None) -> Dict[str, List[str]]:
        """
        Create train/val/test splits based on patients to avoid data leakage
        
        Args:
            train_ratio: Training split ratio
            val_ratio: Validation split ratio  
            test_ratio: Test split ratio
            
        Returns:
            Dictionary with patient IDs for each split
        """
        if train_ratio is None:
            train_ratio = self.config.train_split
        if val_ratio is None:
            val_ratio = self.config.val_split
        if test_ratio is None:
            test_ratio = self.config.test_split
            
        # Normalize ratios
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
        
        # Group patients by class for stratified splitting
        patients_by_class = defaultdict(list)
        for patient_id, patches in self.patient_data.items():
            # Use the class of the first patch (assuming all patches have same class)
            patient_class = patches[0]['class']
            patients_by_class[patient_class].append(patient_id)
        
        # Split patients within each class
        splits = {'train': [], 'val': [], 'test': []}
        
        for class_name, patients in patients_by_class.items():
            # Shuffle patients
            random.shuffle(patients)
            
            n_patients = len(patients)
            n_train = int(n_patients * train_ratio)
            n_val = int(n_patients * val_ratio)
            
            # Split patients
            train_patients = patients[:n_train]
            val_patients = patients[n_train:n_train + n_val]
            test_patients = patients[n_train + n_val:]
            
            splits['train'].extend(train_patients)
            splits['val'].extend(val_patients)
            splits['test'].extend(test_patients)
            
            print(f"   {class_name}: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test patients")
        
        print(f"📊 Patient splits: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
        
        return splits
    
    def format_data_for_conversation(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format data sample into conversational format for MedGemma
        
        Args:
            example: Data sample with image_path, class, etc.
            
        Returns:
            Formatted example with messages field
        """
        # Get the label index for this class
        label_idx = self.label_mapping.get(example["class"], 0)
        
        # Create conversational format
        example["messages"] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text", 
                        "text": CLASSIFICATION_PROMPT,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": self.tissue_classes[label_idx] if label_idx < len(self.tissue_classes) else f"Class {example['class']}",
                    },
                ],
            },
        ]
        
        # Add label for training
        example["label"] = label_idx
        
        return example

    def create_datasets(self, patient_splits: Dict[str, List[str]]) -> DatasetDict:
        """
        Create HuggingFace datasets from patient splits with conversational format
        
        Args:
            patient_splits: Dictionary with patient IDs for each split
            
        Returns:
            DatasetDict with train/val/test splits
        """
        datasets = {}
        
        for split_name, patient_ids in patient_splits.items():
            if not patient_ids:
                continue
                
            # Collect all samples for this split
            split_samples = []
            
            for patient_id in patient_ids:
                patient_patches = self.patient_data[patient_id]
                
                # Limit patches per patient if specified
                if len(patient_patches) > self.config.max_patches_per_patient:
                    patient_patches = random.sample(patient_patches, self.config.max_patches_per_patient)
                
                # Skip patients with too few patches
                if len(patient_patches) < self.config.min_patches_per_patient:
                    continue
                
                # Add each patch as a sample
                for patch_info in patient_patches:
                    # Load image
                    try:
                        image = Image.open(patch_info['image_path']).convert("RGB")
                        
                        sample = {
                            'image': image,
                            'image_path': patch_info['image_path'],
                            'patient_id': patient_id,
                            'class': patch_info['class']
                        }
                        
                        # Format for conversation
                        sample = self.format_data_for_conversation(sample)
                        split_samples.append(sample)
                        
                    except Exception as e:
                        print(f"⚠️ Error loading image {patch_info['image_path']}: {e}")
                        continue
            
            # Limit dataset size if specified (like in official notebook)
            if split_name == 'train' and hasattr(self.config, 'train_size') and self.config.train_size > 0:
                if len(split_samples) > self.config.train_size:
                    split_samples = random.sample(split_samples, self.config.train_size)
            elif split_name == 'val' and hasattr(self.config, 'validation_size') and self.config.validation_size > 0:
                if len(split_samples) > self.config.validation_size:
                    split_samples = random.sample(split_samples, self.config.validation_size)
            
            # Create dataset
            if split_samples:
                datasets[split_name] = Dataset.from_list(split_samples)
                print(f"✅ Created {split_name} dataset: {len(split_samples)} samples")
            else:
                print(f"⚠️ No samples found for {split_name} split")
        
        return DatasetDict(datasets)

    def process_dataset(self, data_path: str) -> Tuple[DatasetDict, Dict[str, Any]]:
        """
        Complete dataset processing pipeline
        
        Args:
            data_path: Path to dataset directory
            
        Returns:
            Tuple of (DatasetDict, dataset_info)
        """
        print("🔄 Starting dataset processing...")
        
        # Scan directory structure
        dataset_info = self.scan_dataset_directory(data_path)
        
        # Create patient-based splits
        patient_splits = self.create_patient_based_splits()
        
        # Create datasets
        datasets = self.create_datasets(patient_splits)
        
        # Add split info to dataset_info
        dataset_info['patient_splits'] = patient_splits
        dataset_info['dataset_sizes'] = {split: len(ds) for split, ds in datasets.items()}
        
        print("✅ Dataset processing completed!")
        print(f"📊 Final dataset sizes: {dataset_info['dataset_sizes']}")
        
        return datasets, dataset_info

    def save_dataset_info(self, dataset_info: Dict[str, Any], output_path: str):
        """Save dataset information to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert sets to lists for JSON serialization
        serializable_info = json.loads(json.dumps(dataset_info, default=str))
        
        with open(output_path, 'w') as f:
            json.dump(serializable_info, f, indent=2)
        
        print(f"💾 Dataset info saved to {output_path}")


def create_example_dataset_structure(base_path: str):
    """
    Create an example dataset structure for testing
    This creates dummy directories - you should replace with your actual data
    """
    base_path = Path(base_path)
    
    # Example tissue types (customize based on your data)
    tissue_types = [
        "adipose",
        "background", 
        "debris",
        "lymphocytes",
        "mucus",
        "smooth_muscle",
        "normal_colon_mucosa",
        "cancer_associated_stroma",
        "colorectal_adenocarcinoma_epithelium"
    ]
    
    for tissue_type in tissue_types:
        tissue_dir = base_path / tissue_type
        tissue_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {tissue_dir}")
    
    print(f"✅ Example dataset structure created at {base_path}")
    print("📝 Please place your histopathology images in the appropriate tissue type folders")
    print("📝 Image naming convention: patientID_patchID.extension (e.g., P001_001.jpg)")


if __name__ == "__main__":
    # Example usage
    from config import DataConfig
    
    config = DataConfig(data_path="./example_dataset")
    processor = HistopathDataProcessor(config)
    
    # Create example structure
    create_example_dataset_structure("./example_dataset")
