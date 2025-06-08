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
        # Correctly initialize patient_data here for each scan
        self.patient_data = defaultdict(list)

        current_patients_in_scan = set()

        for class_dir in data_path.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            class_images_count = 0
            class_patients_in_this_class_dir = set()

            # Find all image files
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in self.image_extensions:
                    patient_id = self._extract_patient_id(img_file.name)

                    self.patient_data[patient_id].append({
                        'image_path': str(img_file),
                        'class': class_name
                    })
                    class_images_count +=1
                    class_patients_in_this_class_dir.add(patient_id)
                    current_patients_in_scan.add(patient_id) # Add to overall set for this scan

            class_stats[class_name] = {
                'num_images': class_images_count,
                'num_patients': len(class_patients_in_this_class_dir),
                'patients': list(class_patients_in_this_class_dir)
            }
            
            total_images += len(class_images)
            total_patients += len(class_patients)
            
            print(f"   {class_name}: {len(class_images)} images, {len(class_patients)} patients")
        
        # Create label mapping
        self.label_mapping = {class_name: idx for idx, class_name in enumerate(sorted(class_stats.keys()))}
        
        dataset_info = {
            'total_images': total_images,
            'total_patients': total_patients, # Use the count of unique patient IDs
            'num_classes': len(class_stats),
            'class_stats': class_stats,
            'label_mapping': self.label_mapping,
            'tissue_classes': self.tissue_classes
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
        # Assumes format like: {patient_id}_patch_{patch_number} or {patient_id}_slide_{slide_number}_patch_{patch_number}
        # We want to capture the part before "_patch_" or "_slide_" if they exist, otherwise the whole stem.

        # More specific extraction:
        # Try to split by "_patch_" first, as it's a common delimiter for patches.
        if "_patch_" in name_without_ext:
            return name_without_ext.split("_patch_")[0]

        # If not, try to split by "_slide_" (if slides are an intermediate step before patches)
        if "_slide_" in name_without_ext:
            return name_without_ext.split("_slide_")[0]

        # If still underscores are present, assume the part before the last underscore is the patient_id
        # e.g. patient_123_extra_info -> patient_123_extra
        # This might need adjustment if patient IDs themselves contain many underscores and patch info doesn't use "_patch_"
        if '_' in name_without_ext:
            return name_without_ext.rsplit('_', 1)[0]

        # Fallback: if no common delimiters or underscores, use the whole name stem.
        # This covers cases like "patient123.jpg"
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

        # Normalize ratios if they don't sum to 1
        current_total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 < current_total_ratio < 1.01): # Allow for small float inaccuracies
            print(f"⚠️ Ratios do not sum to 1. Normalizing: train={train_ratio}, val={val_ratio}, test={test_ratio}")
            train_ratio /= current_total_ratio
            val_ratio /= current_total_ratio
            test_ratio /= current_total_ratio

        # Group patients by class for stratified splitting
        patients_by_class = defaultdict(list)
        # Ensure patient_data is populated from the most recent scan
        if not self.patient_data:
            raise ValueError("patient_data is empty. Run scan_dataset_directory first.")

        for patient_id, patches in self.patient_data.items():
            if not patches: continue # Skip if patient has no patches listed
            patient_class = patches[0]['class']
            patients_by_class[patient_class].append(patient_id)

        splits = {'train': [], 'val': [], 'test': []}

        for class_name, patients in patients_by_class.items():
            random.shuffle(patients)

            n_patients_in_class = len(patients)

            n_train = int(np.floor(n_patients_in_class * train_ratio))
            n_val = int(np.floor(n_patients_in_class * val_ratio))
            n_test = int(np.floor(n_patients_in_class * test_ratio))

            # Distribute remainder (due to floor)
            remainder = n_patients_in_class - (n_train + n_val + n_test)

            # Prioritize splits that have a ratio > 0 but got 0 patients
            if remainder > 0 and val_ratio > 0 and n_val == 0:
                n_val += 1
                remainder -= 1
            if remainder > 0 and test_ratio > 0 and n_test == 0:
                n_test += 1
                remainder -= 1

            # Distribute any further remainder, typically to train or largest specified split
            if remainder > 0:
                if train_ratio >= val_ratio and train_ratio >= test_ratio: # Train is largest or equal
                    n_train += remainder
                elif val_ratio >= test_ratio: # Val is largest or equal (and larger than train)
                    n_val += remainder
                else: # Test is largest
                    n_test += remainder

            # Correct potential over-assignment from remainder logic if a split had 0 ratio
            if train_ratio == 0 and n_train > 0:
                if val_ratio > 0:
                    n_val += n_train
                elif test_ratio > 0:
                    n_test += n_train
                else: # This case should ideally not happen if total ratio is 1
                    if n_patients_in_class > 0:
                        n_train = n_patients_in_class # assign all to train if others are 0
                if train_ratio == 0:
                    n_train = 0 # ensure it's zero if ratio is zero

            if val_ratio == 0 and n_val > 0:
                if train_ratio > 0:
                    n_train += n_val
                elif test_ratio > 0:
                    n_test += n_val
                else:
                    if n_patients_in_class > 0:
                        n_val = n_patients_in_class
                if val_ratio == 0:
                    n_val = 0

            if test_ratio == 0 and n_test > 0:
                if train_ratio > 0:
                    n_train += n_test
                elif val_ratio > 0:
                    n_val += n_test
                else:
                    if n_patients_in_class > 0:
                        n_test = n_patients_in_class
                if test_ratio == 0:
                    n_test = 0

            # Ensure sum matches n_patients_in_class after adjustments
            # This is a final guard. If logic above is perfect, this might not be strictly needed.
            final_sum = n_train + n_val + n_test
            if final_sum != n_patients_in_class:
                # If sum is less, add deficit to the largest ratio split
                if final_sum < n_patients_in_class:
                    deficit = n_patients_in_class - final_sum
                    if train_ratio >= val_ratio and train_ratio >= test_ratio:
                        n_train += deficit
                    elif val_ratio >= test_ratio:
                        n_val += deficit
                    else:
                        n_test += deficit
                # If sum is more, remove surplus from smallest non-zero ratio split that has items
                elif final_sum > n_patients_in_class:
                    surplus = final_sum - n_patients_in_class
                    # Try removing from smallest positive ratio split first
                    if test_ratio > 0 and n_test >= surplus:
                        n_test -= surplus
                    elif val_ratio > 0 and n_val >= surplus:
                        n_val -= surplus
                    elif train_ratio > 0 and n_train >= surplus:
                        n_train -= surplus
                    else: # fallback if all are zero ratio but have counts (should not happen)
                        if n_test >= surplus:
                            n_test -= surplus
                        elif n_val >= surplus:
                            n_val -= surplus
                        else:
                            n_train -= surplus


            train_patients = patients[:n_train]
            val_patients = patients[n_train : n_train + n_val]
            test_patients = patients[n_train + n_val : n_train + n_val + n_test]

            # Ensure all patients are assigned if counts don't sum up, assign to the largest
            if len(train_patients) + len(val_patients) + len(test_patients) < n_patients_in_class:
                # This indicates an issue in count distribution
                # As a fallback, assign remaining to train
                remaining_start_idx = n_train + n_val + n_test
                if train_ratio > 0: # only add to train if it's supposed to have data
                    train_patients.extend(patients[remaining_start_idx:])
                elif val_ratio > 0:
                    val_patients.extend(patients[remaining_start_idx:])
                elif test_ratio > 0:
                    test_patients.extend(patients[remaining_start_idx:])
                # If all ratios are 0, this is an edge case (e.g. 1 patient, ratios 0,0,0)
                # The initial normalize ratios should prevent sum of ratios being 0 if total > 0

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
            DatasetDict with train/val/test splits
        """
        datasets = {}

        for split_name, patient_ids in patient_splits.items():
            if not patient_ids:  # Skip empty splits
                continue
                
            split_data = []
            
            for patient_id in patient_ids:
                if patient_id not in self.patient_data:
                    print(f"⚠️ Patient ID {patient_id} from splits not found in self.patient_data. Skipping.")
                    continue
                patient_patches = self.patient_data[patient_id]

                # Limit patches per patient if specified
                if len(patient_patches) > self.config.max_patches_per_patient:
                    patient_patches = random.sample(patient_patches, self.config.max_patches_per_patient)
                elif len(patient_patches) < self.config.min_patches_per_patient:
                    # Skip patients with insufficient patches
                    continue
                
                for patch_info in patient_patches:
                    # Load image
                    try:
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
