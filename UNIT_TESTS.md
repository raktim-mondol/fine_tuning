# Unit Tests for MedGemma Fine-Tuning

This document provides an overview of the unit tests included in this project. These tests are designed to ensure the core components of the data processing and model training pipeline are functioning correctly. For researchers looking to fine-tune the MedGemma model, understanding these tests can be beneficial for verifying custom modifications or extending the testing framework.

## 1. Data Processing Tests (`test_data_utils.py`)

The tests in `test_data_utils.py` focus on the functionalities provided by `data_utils.py`, which is responsible for preparing the histopathology image datasets.

The test class `TestDataUtils(unittest.TestCase)` uses:
- **`setUp(self)`**: This method runs before each test. It creates a temporary directory structure (`self.test_dir`) that mimics a real dataset, populated with dummy class folders and files using `create_example_dataset_structure`. This provides a consistent and isolated environment for each test.
- **`tearDown(self)`**: This method runs after each test. It cleans up by removing the temporary directory and its contents, ensuring no test artifacts are left behind.

Key tests within `TestDataUtils` include:

### `test_create_example_dataset_structure(self)`
- **Purpose**: Verifies that the `create_example_dataset_structure` function correctly builds a sample directory layout.
- **Coverage**: Checks for the creation of predefined class directories (e.g., 'adenocarcinoma', 'squamous_cell') and ensures that dummy image files are created within them.
- **Assertions**: Asserts that class directories exist and contain files.

### `test_scan_dataset_directory(self)`
- **Purpose**: Tests the `HistopathDataProcessor.scan_dataset_directory` method, which is responsible for analyzing the dataset structure.
- **Coverage**: Checks if the method accurately counts the total number of classes, images, and unique patients. It also verifies the generated label mapping (class name to integer ID) and statistics per class (number of images and patients).
- **Assertions**: Compares expected counts and mappings with the results from the scan. For the example dataset, it expects 4 classes, 60 total images (4 classes * 5 patients/class * 3 patches/patient), and 20 unique patients.

### `test_extract_patient_id(self)`
- **Purpose**: Validates the `_extract_patient_id` helper method within `HistopathDataProcessor`.
- **Coverage**: Tests the method's ability to correctly parse patient identifiers from various filename formats (e.g., "patient123_patch001.jpg", "P123_001.png").
- **Assertions**: Checks that the extracted patient ID matches the expected ID for given filenames.

### `test_create_patient_based_splits(self)`
- **Purpose**: Ensures the `HistopathDataProcessor.create_patient_based_splits` method correctly divides patients into training, validation, and test sets.
- **Coverage**: Verifies that all unique patients from the initial scan are assigned to one of the splits. It also checks that the splits are mutually exclusive (no patient ID appears in more than one split) and that the proportion of patients in each split is approximately close to the configured ratios (e.g., 70% train, 15% val, 15% test), especially considering the small number of sample patients per class in the test setup.
- **Assertions**: Uses set operations to check for completeness and mutual exclusivity of patient IDs in splits. `assertAlmostEqual` is used for ratio checks, allowing for small deviations due to integer rounding when splitting few patients.

### `test_create_datasets(self)`
- **Purpose**: Tests the `HistopathDataProcessor.create_datasets` method, which constructs Hugging Face `Dataset` objects from the patient splits.
- **Coverage**: Verifies that 'train', 'val', and 'test' datasets are created. It checks if the total number of images across these datasets matches the expected count from the initial scan (60 images). It also inspects the structure of a sample item from a dataset, ensuring it contains the necessary fields like 'image' (as a PIL.Image object), 'messages' (in conversation format), 'subtype', 'patient_id', 'image_path', and 'label'.
- **Assertions**: Checks for the presence of all splits in the output `DatasetDict`. Asserts the total image count and the structure and content of a sample data point. Dummy PIL images are created during this test's setup phase because the `setUp` method only `touch`es files, which `PIL.Image.open` cannot handle.

### `test_process_dataset(self)`
- **Purpose**: Validates the end-to-end `HistopathDataProcessor.process_dataset` method.
- **Coverage**: This is an integration test within `data_utils.py` that calls `scan_dataset_directory`, `create_patient_based_splits`, and `create_datasets`. It ensures the final `DatasetDict` and `dataset_info` are correctly generated.
- **Assertions**: Checks that datasets for all splits are present and that `dataset_info` contains expected overall statistics (number of classes, total images) and split information. Dummy images are also created for this test.

### `test_save_dataset_info(self)`
- **Purpose**: Verifies that the `HistopathDataProcessor.save_dataset_info` method correctly saves dataset metadata to a JSON file.
- **Coverage**: Checks if the specified output JSON file is created and if its content matches the input dictionary.
- **Assertions**: Asserts file existence and compares the loaded JSON content with the original dictionary.

```markdown

## 2. Model Training Tests (`test_medgemma_trainer.py`)

The tests in `test_medgemma_trainer.py` are designed to verify the components of the `MedGemmaFineTuner` class from `medgemma_trainer.py`. Due to the resource-intensive nature of actual model loading and training, these tests heavily rely on **mocking** (using `unittest.mock.patch` and `unittest.mock.MagicMock`) to isolate and test individual methods without external dependencies or long execution times.

The test class `TestMedGemmaFineTuner(unittest.TestCase)` includes:
- **`setUp(self)`**: Initializes a temporary directory for outputs and a minimal, valid `Config` object. This config is used by the `MedGemmaFineTuner` instance in tests. Dummy data directories are also created as expected by the configuration.
- **`tearDown(self)`**: Cleans up the temporary directory after each test.

Key tests within `TestMedGemmaFineTuner` include:

### `test_init_and_setup_environment_*`
- **Purpose**: These tests verify the `__init__` method and the subsequent `setup_environment` call under different conditions.
- **Coverage**:
    - `test_init_and_setup_environment_cuda_available_high_capability`: Checks behavior when CUDA is available and GPU capability is high (>=8.0). Verifies Hugging Face login is attempted and `bf16` remains enabled.
    - `test_init_and_setup_environment_cuda_available_low_capability`: Checks behavior with CUDA but low GPU capability (<8.0). Verifies `bf16` is disabled.
    - `test_init_and_setup_environment_no_cuda`: Ensures a `RuntimeError` is raised if CUDA is not available.
- **Mocking**: `torch.cuda` functions (e.g., `is_available`, `get_device_capability`) and `huggingface_hub.login` are mocked to simulate different hardware and authentication states.
- **Assertions**: Check for mock calls (e.g., `mock_hf_login.assert_called_once_with(...)`) and changes in the configuration (e.g., `fine_tuner.config.training.bf16`).

### `test_load_model_and_processor(self)`
- **Purpose**: Tests the `load_model_and_processor` method.
- **Coverage**: Ensures that `AutoModelForImageTextToText.from_pretrained` and `AutoProcessor.from_pretrained` are called with the correct parameters from the configuration. It also checks that the processor's tokenizer is appropriately configured (padding side, pad token).
- **Mocking**: The actual Hugging Face model/processor loading functions are mocked to return `MagicMock` objects. `setup_environment` is also mocked to isolate this test.
- **Assertions**: Verifies that the mocked loading functions are called with expected arguments. Checks attributes of the `fine_tuner.model` and `fine_tuner.processor` after the call.

### `test_prepare_datasets(self)`
- **Purpose**: Validates the `prepare_datasets` method.
- **Coverage**: Checks that an instance of `HistopathDataProcessor` is created and its `process_dataset` method is called correctly.
- **Mocking**: `HistopathDataProcessor` is mocked. `setup_environment` is mocked.
- **Assertions**: Ensures the mocked `HistopathDataProcessor` and its methods are called as expected and that the `fine_tuner.datasets` and `fine_tuner.dataset_info` attributes are populated with the (mocked) results.

### `test_setup_lora(self)`
- **Purpose**: Tests the `setup_lora` method for LoRA configuration.
- **Coverage**: Verifies that `get_peft_model` (from PEFT library) is called with the correct LoRA configuration derived from the main `Config` object. It also checks that the model's trainable parameter statistics are calculated (though the calculation itself relies on mocked parameter counts).
- **Mocking**: `get_peft_model` is mocked. The `fine_tuner.model` itself is a mock, and its `parameters()` method is also mocked to return mock parameters with `numel` attributes for the statistics calculation. `setup_environment` is mocked.
- **Assertions**: Checks that `get_peft_model` is called with the correct base model and a `LoraConfig` object matching the test setup. Ensures `fine_tuner.model` is updated to the (mocked) PEFT model.

### `test_create_collate_function(self)`
- **Purpose**: Validates the custom `create_collate_function` used by the SFTTrainer.
- **Coverage**: Tests the collate function with a sample batch of data. It ensures that the function correctly processes images and text messages, applies the chat template, and generates labels suitable for training (masking padding and image tokens).
- **Mocking**: The `fine_tuner.processor` and its `tokenizer` are mocked. The `apply_chat_template` method and the processor call itself are mocked to return predefined structures. `setup_environment` is mocked.
- **Assertions**: Checks the structure of the output batch, particularly the `input_ids` and `labels`, ensuring labels are correctly masked.

### `test_train_method_calls(self)`
- **Purpose**: Tests the high-level structure and sequence of operations within the `train` method, without actually running any training.
- **Coverage**: Verifies that `SFTTrainer` is initialized with the correct arguments (model, training arguments, datasets, collate function, callbacks). It checks that `trainer.train()`, `trainer.save_model()`, `processor.save_pretrained()`, and `save_training_metadata()` are called.
- **Mocking**: `SFTTrainer`, `os.makedirs`, and several internal methods of `MedGemmaFineTuner` (`setup_environment`, `load_model_and_processor`, `prepare_datasets`, `setup_lora`, `create_collate_function`, `save_training_metadata`) are mocked. The `fine_tuner.datasets` attribute is populated with mock datasets having a `__len__` method.
- **Assertions**: Focuses on call order and arguments passed to mocked objects (e.g., `MockSFTTrainer.assert_called_once()`).

### `test_save_training_metadata(self)`
- **Purpose**: Tests the `save_training_metadata` method.
- **Coverage**: Ensures that training metadata (including parts of the configuration and training results) is saved to a JSON file, and the main configuration is saved to a YAML file.
- **Mocking**: `builtins.open`, `json.dump`, and the `config.to_yaml` method are mocked. `setup_environment` is mocked.
- **Assertions**: Verifies that files are opened for writing at the correct paths and that `json.dump` and `config.to_yaml` are called with the expected data.

```markdown

## 3. Guidance for Researchers

These unit tests form a foundational layer for ensuring the reliability of the MedGemma fine-tuning codebase. Researchers can leverage and extend these tests in several ways:

### Running the Tests

To run the unit tests, navigate to the root directory of the project in your terminal and execute the following commands:

- **Run Data Utils Tests:**
  ```bash
  python -m unittest test_data_utils.py
  ```

- **Run Trainer Tests:**
  ```bash
  python -m unittest test_medgemma_trainer.py
  ```

- **Run All Tests (if you have multiple test files and want a consolidated run, ensure they follow `test_*.py` pattern):**
  ```bash
  python -m unittest discover
  ```
  (This command discovers and runs all tests in files named `test_*.py` in the current directory and subdirectories.)

Before running, ensure all dependencies listed in `requirements.txt` are installed in your Python environment.

### Extending the Tests

If you modify the existing codebase or add new features, it's highly recommended to update or add corresponding unit tests. Here are some scenarios:

- **Modifying Data Processing Logic (`data_utils.py`):**
    - If you change the patient ID extraction logic in `_extract_patient_id`, update `test_extract_patient_id` with new filename examples.
    - If you alter how datasets are split (e.g., new stratification methods), adjust `test_create_patient_based_splits` to reflect these changes.
    - If the structure of the data expected by `scan_dataset_directory` changes, modify the example dataset created in `setUp` and update assertions in `test_scan_dataset_directory`.

- **Modifying Training Logic (`medgemma_trainer.py`):**
    - If you introduce new configuration parameters in `Config` that affect the trainer, add tests to verify their handling (similar to how `bf16` based on GPU capability is tested).
    - If you change the LoRA setup (e.g., different target modules, different PEFT configurations), update `test_setup_lora` to assert the new configuration.
    - If you modify the `create_collate_function`, ensure `test_create_collate_function` still passes or update its assertions for the new batch structure or label masking logic.
    - For new helper methods within `MedGemmaFineTuner`, create new dedicated test methods using appropriate mocks.

- **Adding New Functionality:**
    - For any new Python module or significant function, create a corresponding `test_your_module_name.py` file and populate it with relevant test cases.

### Importance of Unit Testing in Research

- **Reproducibility**: Ensures that your code behaves as expected across different setups and over time.
- **Reliability**: Catches bugs early, especially when refactoring or adding new features to complex pipelines.
- **Collaboration**: Makes it easier for others (and your future self) to understand and modify the code with confidence.
- **Faster Iteration**: Allows for quick verification of specific components without needing to run the full, time-consuming fine-tuning pipeline for every small change.

By maintaining and extending these unit tests, researchers can contribute to a more robust and reliable fine-tuning workflow for MedGemma.
```
