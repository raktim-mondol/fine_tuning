#!/usr/bin/env python3
"""
Dataset Validation Script for MedGemma Fine-tuning

This script validates your dataset structure and provides detailed feedback
to ensure compatibility with the MedGemma fine-tuning pipeline.

Usage:
    python validate_dataset.py --data_path /path/to/your/dataset
    python validate_dataset.py --data_path /path/to/your/dataset --fix_issues
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import json

from PIL import Image
from data_utils import HistopathDataProcessor
from config import DataConfig


class DatasetValidator:
    """Comprehensive dataset validation for MedGemma fine-tuning"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.issues = []
        self.warnings = []
        self.stats = {}
        
        # Supported image extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
    def validate(self, fix_issues: bool = False) -> bool:
        """
        Run comprehensive validation
        
        Args:
            fix_issues: Whether to attempt automatic fixes
            
        Returns:
            True if validation passes, False otherwise
        """
        print("🔍 Starting dataset validation...")
        print(f"📁 Dataset path: {self.data_path}")
        print("=" * 60)
        
        # Check basic structure
        if not self._validate_basic_structure():
            return False
        
        # Scan dataset
        if not self._scan_dataset():
            return False
        
        # Validate naming convention
        self._validate_naming_convention()
        
        # Validate patient distribution
        self._validate_patient_distribution()
        
        # Validate image files
        self._validate_images()
        
        # Check data splits
        self._validate_data_splits()
        
        # Generate report
        self._generate_report()
        
        # Apply fixes if requested
        if fix_issues and self.issues:
            self._apply_fixes()
        
        # Return validation result
        return len(self.issues) == 0
    
    def _validate_basic_structure(self) -> bool:
        """Validate basic directory structure"""
        print("📋 Checking basic structure...")
        
        if not self.data_path.exists():
            self.issues.append(f"Dataset path does not exist: {self.data_path}")
            return False
        
        if not self.data_path.is_dir():
            self.issues.append(f"Dataset path is not a directory: {self.data_path}")
            return False
        
        # Check for class directories
        class_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        if len(class_dirs) == 0:
            self.issues.append("No class directories found")
            return False
        
        if len(class_dirs) < 2:
            self.issues.append(f"Only {len(class_dirs)} class found. Minimum 2 classes required")
            return False
        
        print(f"✅ Found {len(class_dirs)} class directories")
        return True
    
    def _scan_dataset(self) -> bool:
        """Scan dataset and collect statistics"""
        print("📊 Scanning dataset...")
        
        try:
            config = DataConfig(data_path=str(self.data_path))
            processor = HistopathDataProcessor(config)
            dataset_info = processor.scan_dataset_directory(str(self.data_path))
            
            self.stats = dataset_info
            print(f"✅ Dataset scan completed")
            print(f"   Total images: {dataset_info['total_images']}")
            print(f"   Total patients: {dataset_info['total_patients']}")
            print(f"   Classes: {dataset_info['num_classes']}")
            
            return True
            
        except Exception as e:
            self.issues.append(f"Failed to scan dataset: {str(e)}")
            return False
    
    def _validate_naming_convention(self):
        """Validate file naming convention"""
        print("🏷️ Validating naming convention...")
        
        patient_files = defaultdict(list)
        invalid_names = []
        
        for class_name, class_stats in self.stats['class_stats'].items():
            class_dir = self.data_path / class_name
            
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in self.supported_extensions:
                    filename = img_file.name
                    
                    # Try to extract patient ID
                    try:
                        config = DataConfig()
                        processor = HistopathDataProcessor(config)
                        patient_id = processor._extract_patient_id(filename)
                        
                        if patient_id == filename.split('.')[0]:  # No extraction happened
                            if '_' not in filename:
                                invalid_names.append(filename)
                        
                        patient_files[patient_id].append(filename)
                        
                    except Exception as e:
                        invalid_names.append(f"{filename} (error: {e})")
        
        # Report naming issues
        if invalid_names:
            self.warnings.append(f"Files with unclear patient IDs: {len(invalid_names)}")
            if len(invalid_names) <= 10:
                for name in invalid_names:
                    self.warnings.append(f"  - {name}")
            else:
                for name in invalid_names[:5]:
                    self.warnings.append(f"  - {name}")
                self.warnings.append(f"  ... and {len(invalid_names) - 5} more")
        
        # Check for patient ID consistency
        single_file_patients = [pid for pid, files in patient_files.items() if len(files) == 1]
        if len(single_file_patients) > len(patient_files) * 0.5:
            self.warnings.append(f"Many patients ({len(single_file_patients)}) have only one image")
        
        print(f"✅ Naming convention check completed")
        print(f"   Unique patients detected: {len(patient_files)}")
        print(f"   Average images per patient: {sum(len(files) for files in patient_files.values()) / len(patient_files):.1f}")
    
    def _validate_patient_distribution(self):
        """Validate patient distribution across classes"""
        print("👥 Validating patient distribution...")
        
        class_patient_counts = {}
        patient_class_mapping = {}
        
        for class_name, class_stats in self.stats['class_stats'].items():
            class_patient_counts[class_name] = class_stats['num_patients']
            
            # Check for patients appearing in multiple classes
            for patient in class_stats['patients']:
                if patient in patient_class_mapping:
                    self.issues.append(f"Patient {patient} appears in multiple classes: {patient_class_mapping[patient]} and {class_name}")
                else:
                    patient_class_mapping[patient] = class_name
        
        # Check class balance
        min_patients = min(class_patient_counts.values())
        max_patients = max(class_patient_counts.values())
        
        if min_patients < 5:
            self.issues.append(f"Some classes have very few patients (minimum: {min_patients}). Recommended: 5+ patients per class")
        
        if max_patients / min_patients > 10:
            self.warnings.append(f"Significant class imbalance detected (ratio: {max_patients/min_patients:.1f}:1)")
        
        # Print distribution
        print("   Patient distribution by class:")
        for class_name, count in sorted(class_patient_counts.items()):
            print(f"     {class_name}: {count} patients")
    
    def _validate_images(self):
        """Validate image files"""
        print("🖼️ Validating image files...")
        
        corrupted_images = []
        unsupported_formats = []
        large_images = []
        
        total_images = 0
        total_size = 0
        
        for class_name in self.stats['class_stats'].keys():
            class_dir = self.data_path / class_name
            
            for img_file in class_dir.iterdir():
                if img_file.is_file():
                    total_images += 1
                    file_size = img_file.stat().st_size
                    total_size += file_size
                    
                    # Check file extension
                    if img_file.suffix.lower() not in self.supported_extensions:
                        unsupported_formats.append(str(img_file))
                        continue
                    
                    # Check if file is too large
                    if file_size > 50 * 1024 * 1024:  # 50MB
                        large_images.append(f"{img_file.name} ({file_size / 1024 / 1024:.1f}MB)")
                    
                    # Try to load image
                    try:
                        with Image.open(img_file) as img:
                            # Check image properties
                            if img.mode not in ['RGB', 'RGBA', 'L']:
                                self.warnings.append(f"Unusual color mode in {img_file.name}: {img.mode}")
                            
                            if min(img.size) < 32:
                                self.warnings.append(f"Very small image: {img_file.name} ({img.size})")
                    
                    except Exception as e:
                        corrupted_images.append(f"{img_file.name}: {str(e)}")
        
        # Report image issues
        if corrupted_images:
            self.issues.append(f"Corrupted images found: {len(corrupted_images)}")
            for img in corrupted_images[:5]:
                self.issues.append(f"  - {img}")
            if len(corrupted_images) > 5:
                self.issues.append(f"  ... and {len(corrupted_images) - 5} more")
        
        if unsupported_formats:
            self.warnings.append(f"Unsupported image formats: {len(unsupported_formats)}")
        
        if large_images:
            self.warnings.append(f"Large images (>50MB): {len(large_images)}")
            for img in large_images[:3]:
                self.warnings.append(f"  - {img}")
        
        print(f"✅ Image validation completed")
        print(f"   Total images: {total_images}")
        print(f"   Average file size: {total_size / total_images / 1024 / 1024:.1f}MB")
    
    def _validate_data_splits(self):
        """Validate data splitting capability"""
        print("📊 Validating data splits...")
        
        try:
            config = DataConfig(data_path=str(self.data_path))
            processor = HistopathDataProcessor(config)
            datasets, info = processor.process_dataset(str(self.data_path))
            
            print(f"✅ Data splits created successfully:")
            for split_name, dataset in datasets.items():
                print(f"     {split_name}: {len(dataset)} samples")
            
            # Check split sizes
            total_samples = sum(len(ds) for ds in datasets.values())
            if total_samples < 100:
                self.warnings.append(f"Small dataset size: {total_samples} samples. Recommended: 1000+ for good results")
            
            # Check if validation set is reasonable
            if 'val' in datasets and len(datasets['val']) < 50:
                self.warnings.append("Small validation set. Consider increasing validation split ratio")
            
        except Exception as e:
            self.issues.append(f"Failed to create data splits: {str(e)}")
    
    def _generate_report(self):
        """Generate validation report"""
        print("\n" + "=" * 60)
        print("📋 VALIDATION REPORT")
        print("=" * 60)
        
        # Summary
        if self.issues:
            print(f"❌ VALIDATION FAILED - {len(self.issues)} critical issues found")
        else:
            print("✅ VALIDATION PASSED - Dataset is ready for training")
        
        if self.warnings:
            print(f"⚠️ {len(self.warnings)} warnings found")
        
        # Critical issues
        if self.issues:
            print("\n🚨 CRITICAL ISSUES (must be fixed):")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        
        # Warnings
        if self.warnings:
            print("\n⚠️ WARNINGS (recommended to address):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        # Dataset statistics
        if self.stats:
            print("\n📊 DATASET STATISTICS:")
            print(f"  Total images: {self.stats['total_images']}")
            print(f"  Total patients: {self.stats['total_patients']}")
            print(f"  Number of classes: {self.stats['num_classes']}")
            print(f"  Average images per patient: {self.stats['total_images'] / self.stats['total_patients']:.1f}")
            
            print("\n  Class distribution:")
            for class_name, stats in self.stats['class_stats'].items():
                print(f"    {class_name}: {stats['num_images']} images, {stats['num_patients']} patients")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if not self.issues and not self.warnings:
            print("  Your dataset is well-structured and ready for training!")
        else:
            if self.issues:
                print("  1. Fix all critical issues before training")
            if self.warnings:
                print("  2. Consider addressing warnings for better results")
            print("  3. Test with a small subset first")
            print("  4. Monitor training metrics closely")
        
        print("\n📖 For detailed guidance, see DATA_STRUCTURE_GUIDE.md")
    
    def _apply_fixes(self):
        """Apply automatic fixes where possible"""
        print("\n🔧 Attempting automatic fixes...")
        
        fixes_applied = 0
        
        # Example fixes could include:
        # - Converting image formats
        # - Renaming files to follow convention
        # - Moving misplaced files
        
        # For now, just provide guidance
        print("⚠️ Automatic fixes not implemented yet.")
        print("Please manually address the issues listed above.")
        print("Refer to DATA_STRUCTURE_GUIDE.md for detailed instructions.")
    
    def save_report(self, output_path: str):
        """Save validation report to file"""
        report = {
            "validation_passed": len(self.issues) == 0,
            "critical_issues": self.issues,
            "warnings": self.warnings,
            "statistics": self.stats,
            "timestamp": str(Path().cwd()),
            "dataset_path": str(self.data_path)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Validation report saved to: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Validate dataset structure for MedGemma fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python validate_dataset.py --data_path ./my_dataset
  
  # Validation with fix attempts
  python validate_dataset.py --data_path ./my_dataset --fix_issues
  
  # Save detailed report
  python validate_dataset.py --data_path ./my_dataset --report validation_report.json
        """
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--fix_issues",
        action="store_true",
        help="Attempt to automatically fix issues"
    )
    
    parser.add_argument(
        "--report",
        type=str,
        help="Save detailed report to JSON file"
    )
    
    args = parser.parse_args()
    
    # Validate dataset
    validator = DatasetValidator(args.data_path)
    validation_passed = validator.validate(fix_issues=args.fix_issues)
    
    # Save report if requested
    if args.report:
        validator.save_report(args.report)
    
    # Exit with appropriate code
    if validation_passed:
        print("\n🎉 Dataset validation successful! You can proceed with training.")
        sys.exit(0)
    else:
        print("\n❌ Dataset validation failed. Please fix the issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main() 