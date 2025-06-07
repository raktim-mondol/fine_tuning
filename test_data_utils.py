import unittest
import tempfile
import shutil
import json
from pathlib import Path
from PIL import Image
from data_utils import HistopathDataProcessor, create_example_dataset_structure
from config import DataConfig

class TestDataUtils(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.data_path = Path(self.test_dir) / "example_histopath_data"

        # Create example dataset structure
        create_example_dataset_structure(str(self.data_path))

        # Initialize DataConfig (adjust parameters as needed for testing)
        self.data_config = DataConfig(
            data_path=str(self.data_path),
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            max_patches_per_patient=10, # Allow all patches for testing
            min_patches_per_patient=1
        )
        self.processor = HistopathDataProcessor(self.data_config)

    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)

    def test_create_example_dataset_structure(self):
        # Check if class directories are created
        classes = ['adenocarcinoma', 'squamous_cell', 'normal', 'inflammatory']
        for class_name in classes:
            self.assertTrue((self.data_path / class_name).exists())
            # Check for dummy files
            self.assertTrue(len(list((self.data_path / class_name).iterdir())) > 0)

    def test_scan_dataset_directory(self):
        dataset_info = self.processor.scan_dataset_directory(str(self.data_path))

        self.assertEqual(dataset_info['num_classes'], 4)
        self.assertEqual(dataset_info['total_images'], 4 * 5 * 3) # 4 classes * 5 patients * 3 patches
        self.assertEqual(dataset_info['total_patients'], 4 * 5)

        expected_labels = {'adenocarcinoma': 0, 'inflammatory': 1, 'normal': 2, 'squamous_cell': 3}
        self.assertEqual(dataset_info['label_mapping'], expected_labels)

        for class_name in ['adenocarcinoma', 'squamous_cell', 'normal', 'inflammatory']:
            self.assertEqual(dataset_info['class_stats'][class_name]['num_images'], 5 * 3)
            self.assertEqual(dataset_info['class_stats'][class_name]['num_patients'], 5)

    def test_extract_patient_id(self):
        # Test various filename formats
        self.assertEqual(self.processor._extract_patient_id("patient123_patch001.jpg"), "patient123")
        self.assertEqual(self.processor._extract_patient_id("P123_001.png"), "P123")
        self.assertEqual(self.processor._extract_patient_id("case_456_slide_1_patch_2.tif"), "case_456_slide_1") # Should split on _patch_
        self.assertEqual(self.processor._extract_patient_id("abc_def_ghi.jpeg"), "abc_def") # Adjusted expectation based on rsplit
        self.assertEqual(self.processor._extract_patient_id("TestID007.bmp"), "TestID007") # No underscore

    def test_create_patient_based_splits(self):
        # First, scan the directory to populate patient_data
        self.processor.scan_dataset_directory(str(self.data_path))
        patient_splits = self.processor.create_patient_based_splits()

        self.assertIn('train', patient_splits)
        self.assertIn('val', patient_splits)
        self.assertIn('test', patient_splits)

        all_patients_in_splits = set(patient_splits['train'] + patient_splits['val'] + patient_splits['test'])

        # Check that all patients from scan are in the splits
        original_patients = set(self.processor.patient_data.keys())
        self.assertEqual(original_patients, all_patients_in_splits)

        # Check that splits are mutually exclusive (no patient ID in more than one split)
        self.assertTrue(set(patient_splits['train']).isdisjoint(set(patient_splits['val'])))
        self.assertTrue(set(patient_splits['train']).isdisjoint(set(patient_splits['test'])))
        self.assertTrue(set(patient_splits['val']).isdisjoint(set(patient_splits['test'])))

        # Check approximate ratios (can vary due to small number of patients per class)
        total_patients = len(original_patients)
        # print(f"Total patients: {total_patients}")
        # print(f"Train: {len(patient_splits['train'])}, Val: {len(patient_splits['val'])}, Test: {len(patient_splits['test'])}")
        # Expected counts based on 5 patients per class, 4 classes = 20 patients total
        # Train: 0.7 * 5 = 3.5 -> 3 per class (12 total)
        # Val: 0.15 * 5 = 0.75 -> 1 per class (4 total)
        # Test: 0.15 * 5 = 0.75 -> 1 per class (4 total)
        # These counts are approximate due to integer rounding within each class split
        self.assertAlmostEqual(len(patient_splits['train']) / total_patients, self.data_config.train_split, delta=0.2)
        self.assertAlmostEqual(len(patient_splits['val']) / total_patients, self.data_config.val_split, delta=0.2)
        self.assertAlmostEqual(len(patient_splits['test']) / total_patients, self.data_config.test_split, delta=0.2)


    def test_create_datasets(self):
        # Need to scan and create splits first
        self.processor.scan_dataset_directory(str(self.data_path))
        patient_splits = self.processor.create_patient_based_splits()

        # For this test, create some dummy image files as create_example_dataset_structure only touches them
        for class_name in self.processor.label_mapping.keys():
            class_dir = self.data_path / class_name
            for patient_id_key in self.processor.patient_data: # patient_data keys are like 'patient_001' etc.
                # Find files belonging to this patient_id_key
                # Example: patient_id_key = 'patient_001', filename 'patient_001_patch_01.jpg'
                matching_files = [f for f in class_dir.glob(f"{patient_id_key}*.jpg")]
                for img_file_path in matching_files:
                    try:
                        img = Image.new('RGB', (60, 30), color = 'red') # Create a small dummy image
                        img.save(img_file_path)
                    except Exception as e:
                        print(f"Error creating dummy image {img_file_path}: {e}")


        datasets = self.processor.create_datasets(patient_splits)

        self.assertIn('train', datasets)
        self.assertIn('val', datasets)
        self.assertIn('test', datasets)

        total_images_in_datasets = sum(len(ds) for ds in datasets.values())
        # Expected total images: 4 classes * 5 patients * 3 patches = 60
        # This should match total_images from scan_dataset_directory if max_patches_per_patient is high enough
        # and min_patches_per_patient is low enough
        self.assertEqual(total_images_in_datasets, 4 * 5 * 3)

        # Check structure of one sample
        if len(datasets['train']) > 0:
            sample = datasets['train'][0]
            self.assertIn('image', sample)
            self.assertIn('messages', sample)
            self.assertIn('subtype', sample)
            self.assertIn('patient_id', sample)
            self.assertIn('image_path', sample)
            self.assertIn('label', sample)
            self.assertIsInstance(sample['image'], Image.Image)
            self.assertEqual(len(sample['messages']), 2) # User and assistant turn
            self.assertEqual(sample['messages'][0]['role'], 'user')
            self.assertEqual(sample['messages'][1]['role'], 'assistant')
            self.assertEqual(sample['messages'][1]['content'], sample['subtype'])

    def test_process_dataset(self):
        # Create dummy images for process_dataset to load
        # Iterate over files created by create_example_dataset_structure
        for class_dir in self.data_path.iterdir():
            if not class_dir.is_dir():
                continue
            for img_file_path in class_dir.iterdir():
                if img_file_path.suffix.lower() not in self.processor.image_extensions:
                    continue
                try:
                    # Ensure parent directory exists (it should from setUp)
                    img_file_path.parent.mkdir(parents=True, exist_ok=True)
                    img = Image.new('RGB', (60, 30), color = 'blue')
                    img.save(img_file_path)
                except Exception as e:
                    print(f"Error creating dummy image {img_file_path}: {e}")

        datasets, dataset_info = self.processor.process_dataset(str(self.data_path))

        self.assertIsNotNone(datasets)
        self.assertIsNotNone(dataset_info)

        self.assertIn('train', datasets)
        self.assertIn('val', datasets)
        self.assertIn('test', datasets)

        self.assertEqual(dataset_info['num_classes'], 4)
        self.assertEqual(dataset_info['total_images'], 4 * 5 * 3)

        # Check that split info is added
        self.assertIn('splits', dataset_info) # patient counts per split
        self.assertIn('dataset_splits', dataset_info) # image counts per split

    def test_save_dataset_info(self):
        dataset_info = {"test_key": "test_value", "num": 123}
        output_path = self.data_path / "dataset_info.json"
        self.processor.save_dataset_info(dataset_info, str(output_path))

        self.assertTrue(output_path.exists())
        with open(output_path, 'r') as f:
            loaded_info = json.load(f)
        self.assertEqual(loaded_info, dataset_info)

if __name__ == '__main__':
    unittest.main()
