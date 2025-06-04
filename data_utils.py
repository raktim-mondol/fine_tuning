"""
Data preprocessing and loading utilities for histopathology images
Handles multiple patches per patient scenario
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


class HistopathDataProcessor:
    """
    Data processor for histopathology images with multiple patches per patient
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.image_extensions = set(config.image_extensions)
        self.patient_data = defaultdict(list)
        self.label_mapping = {}
        
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
        
        # Create label mapping
        self.label_mapping = {class_name: idx for idx, class_name in enumerate(sorted(class_stats.keys()))}
        
        dataset_info = {
            'total_images': total_images,
            'total_patients': total_patients,
            'num_classes': len(class_stats),
            'class_stats': class_stats,
            'label_mapping': self.label_mapping
        }
        
        print(f"✅ Dataset scan complete: {total_images} images, {total_patients} patients, {len(class_stats)} classes")
        
        return dataset_info
    
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
        
        return splits
    
    def create_datasets(self, patient_splits: Dict[str, List[str]]) -> DatasetDict:
        """
        Create HuggingFace datasets from patient splits
        
        Args:
            patient_splits: Dictionary with patient IDs for each split
            
        Returns:
            DatasetDict with train/val/test datasets
        """
        datasets = {}
        
        for split_name, patient_ids in patient_splits.items():
            if not patient_ids:  # Skip empty splits
                continue
                
            split_data = []
            
            for patient_id in patient_ids:
                patient_patches = self.patient_data[patient_id]
                
                # Limit patches per patient if specified
                if len(patient_patches) > self.config.max_patches_per_patient:
                    patient_patches = random.sample(patient_patches, self.config.max_patches_per_patient)
                elif len(patient_patches) < self.config.min_patches_per_patient:
                    # Skip patients with insufficient patches
                    continue
                
                for patch_info in patient_patches:
                    try:
                        # Load and verify image
                        image = Image.open(patch_info['image_path']).convert("RGB")
                        
                        # Create conversation format for instruction tuning
                        conversation = [
                            {
                                "role": "user",
                                "content": "Classify the histopathology subtype in this image:"
                            },
                            {
                                "role": "assistant",
                                "content": patch_info['class']
                            }
                        ]
                        
                        split_data.append({
                            "image": image,
                            "messages": conversation,
                            "subtype": patch_info['class'],
                            "patient_id": patient_id,
                            "image_path": patch_info['image_path'],
                            "label": self.label_mapping[patch_info['class']]
                        })
                        
                    except Exception as e:
                        print(f"⚠️ Skipping corrupted image {patch_info['image_path']}: {e}")
                        continue
            
            if split_data:
                datasets[split_name] = Dataset.from_list(split_data)
                print(f"📊 {split_name}: {len(split_data)} samples from {len(patient_ids)} patients")
        
        return DatasetDict(datasets)
    
    def process_dataset(self, data_path: str) -> Tuple[DatasetDict, Dict[str, Any]]:
        """
        Complete dataset processing pipeline
        
        Args:
            data_path: Root directory containing class folders
            
        Returns:
            Tuple of (DatasetDict, dataset_info)
        """
        print("🔄 Starting dataset processing...")
        
        # Step 1: Scan directory
        dataset_info = self.scan_dataset_directory(data_path)
        
        # Step 2: Create patient-based splits
        patient_splits = self.create_patient_based_splits()
        
        # Step 3: Create datasets
        datasets = self.create_datasets(patient_splits)
        
        # Add split info to dataset_info
        dataset_info['splits'] = {
            split_name: len(patient_ids) for split_name, patient_ids in patient_splits.items()
        }
        dataset_info['dataset_splits'] = {
            split_name: len(dataset) for split_name, dataset in datasets.items()
        }
        
        print("✅ Dataset processing complete!")
        print(f"📈 Final dataset sizes: {dataset_info['dataset_splits']}")
        
        return datasets, dataset_info
    
    def save_dataset_info(self, dataset_info: Dict[str, Any], output_path: str):
        """Save dataset information to JSON file"""
        with open(output_path, 'w') as f:
            # Convert any non-serializable objects
            serializable_info = {}
            for key, value in dataset_info.items():
                if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                    serializable_info[key] = value
                else:
                    serializable_info[key] = str(value)
            
            json.dump(serializable_info, f, indent=2)
        print(f"💾 Dataset info saved to {output_path}")


def create_example_dataset_structure(base_path: str):
    """
    Create an example dataset structure for testing
    This is just for demonstration - replace with your actual data
    """
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)
    
    # Create class directories
    classes = ['adenocarcinoma', 'squamous_cell', 'normal', 'inflammatory']
    
    for class_name in classes:
        class_dir = base_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create some dummy patient files (you would place your actual images here)
        for patient_id in range(1, 6):  # 5 patients per class
            for patch_id in range(1, 4):  # 3 patches per patient
                filename = f"patient_{patient_id:03d}_patch_{patch_id:02d}.jpg"
                dummy_file = class_dir / filename
                if not dummy_file.exists():
                    dummy_file.touch()
    
    print(f"📁 Example dataset structure created at: {base_path}")
    print("Replace dummy files with your actual histopathology images")


if __name__ == "__main__":
    # Example usage
    from config import DataConfig
    
    # Create example dataset structure
    create_example_dataset_structure("./example_histopath_data")
    
    # Initialize processor
    data_config = DataConfig(data_path="./example_histopath_data")
    processor = HistopathDataProcessor(data_config)
    
    # Process dataset
    datasets, info = processor.process_dataset(data_config.data_path)
    
    # Save dataset info
    processor.save_dataset_info(info, "./dataset_info.json")
