import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import shutil
from pathlib import Path
import os
import json
import torch # Import torch here

# Make sure config classes are imported before MedGemmaFineTuner
from config import Config, ModelConfig, LoRAConfig, TrainingConfig, DataConfig
from medgemma_trainer import MedGemmaFineTuner


class TestMedGemmaFineTuner(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.test_dir) / "output"
        self.data_dir = Path(self.test_dir) / "data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal valid config
        self.config = Config(
            hf_token="dummy_token",
            seed=42,
            model=ModelConfig(
                model_id="google/medgemma-4b-pt", # A real model ID for processor loading
                torch_dtype='bfloat16',
                attn_implementation='flash_attention_2',
                device_map='auto',
                trust_remote_code=True
            ),
            lora=LoRAConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "v_proj"]
            ),
            training=TrainingConfig(
                output_dir=str(self.output_dir),
                num_epochs=1,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=5e-5,
                max_seq_length=512,
                bf16=True, # Will be overridden if CUDA capability is low in tests
                early_stopping_patience=3
            ),
            data=DataConfig(
                data_path=str(self.data_dir),
                image_extensions=[".jpg", ".png"],
                train_split=0.7,
                val_split=0.15,
                test_split=0.15
            )
        )

        # Create dummy data structure for DataConfig
        (self.data_dir / "classA").mkdir()
        (self.data_dir / "classA" / "patient1_patch1.jpg").touch()


    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('medgemma_trainer.login')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_capability')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    def test_init_and_setup_environment_cuda_available_high_capability(
        self, mock_get_device_properties, mock_get_device_name, mock_get_device_capability, mock_is_available, mock_hf_login
    ):
        mock_is_available.return_value = True
        mock_get_device_capability.return_value = (8, 0) # High capability
        mock_get_device_name.return_value = "Test GPU"
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1e9
        mock_get_device_properties.return_value = mock_props

        fine_tuner = MedGemmaFineTuner(self.config)

        mock_hf_login.assert_called_once_with(token="dummy_token")
        self.assertTrue(fine_tuner.config.training.bf16) # Should remain true


    @patch('medgemma_trainer.login')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_capability')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    def test_init_and_setup_environment_cuda_available_low_capability(
        self, mock_get_device_properties, mock_get_device_name, mock_get_device_capability, mock_is_available, mock_hf_login
    ):
        mock_is_available.return_value = True
        mock_get_device_capability.return_value = (7, 0) # Low capability
        mock_get_device_name.return_value = "Old Test GPU"
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1e9
        mock_get_device_properties.return_value = mock_props

        original_bf16_setting = self.config.training.bf16
        fine_tuner = MedGemmaFineTuner(self.config)

        mock_hf_login.assert_called_once_with(token="dummy_token")
        if original_bf16_setting: # bf16 should be forced to False for low capability
             self.assertFalse(fine_tuner.config.training.bf16)


    @patch('medgemma_trainer.login')
    @patch('torch.cuda.is_available')
    def test_init_and_setup_environment_no_cuda(self, mock_is_available, mock_hf_login):
        mock_is_available.return_value = False
        with self.assertRaises(RuntimeError) as context:
            MedGemmaFineTuner(self.config)
        self.assertIn("CUDA is not available", str(context.exception))
        mock_hf_login.assert_called_once_with(token="dummy_token")


    @patch('medgemma_trainer.AutoModelForImageTextToText.from_pretrained')
    @patch('medgemma_trainer.AutoProcessor.from_pretrained')
    @patch.object(MedGemmaFineTuner, 'setup_environment') # Mock setup_environment to isolate this test
    def test_load_model_and_processor(self, mock_setup_env, mock_processor_load, mock_model_load):
        mock_model = MagicMock()
        mock_model.num_parameters.return_value = 1_000_000
        mock_model_load.return_value = mock_model

        mock_proc = MagicMock()
        mock_proc.tokenizer = MagicMock()
        mock_proc.tokenizer.pad_token = None # Test the pad_token setting logic
        mock_proc.tokenizer.eos_token = "<eos>"
        mock_processor_load.return_value = mock_proc

        fine_tuner = MedGemmaFineTuner(self.config) # setup_environment is mocked
        fine_tuner.load_model_and_processor()

        mock_model_load.assert_called_once_with(
            self.config.model.model_id,
            attn_implementation=self.config.model.attn_implementation,
            torch_dtype=torch.bfloat16, # Based on config string
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code
        )
        mock_processor_load.assert_called_once_with(self.config.model.model_id)
        self.assertIsNotNone(fine_tuner.model)
        self.assertIsNotNone(fine_tuner.processor)
        self.assertEqual(fine_tuner.processor.tokenizer.padding_side, "right")
        self.assertEqual(fine_tuner.processor.tokenizer.pad_token, "<eos>")


    @patch('medgemma_trainer.HistopathDataProcessor')
    @patch.object(MedGemmaFineTuner, 'setup_environment')
    def test_prepare_datasets(self, mock_setup_env, MockHistopathDataProcessor):
        mock_data_processor_instance = MagicMock()
        mock_datasets = MagicMock(spec=dict) # Make it behave like a dict
        mock_dataset_info = {"info": "test_info"}
        mock_data_processor_instance.process_dataset.return_value = (mock_datasets, mock_dataset_info)
        MockHistopathDataProcessor.return_value = mock_data_processor_instance

        fine_tuner = MedGemmaFineTuner(self.config)
        datasets, dataset_info = fine_tuner.prepare_datasets()

        MockHistopathDataProcessor.assert_called_once_with(self.config.data)
        mock_data_processor_instance.process_dataset.assert_called_once_with(self.config.data.data_path)
        self.assertEqual(datasets, mock_datasets)
        self.assertEqual(dataset_info, mock_dataset_info)
        self.assertEqual(fine_tuner.datasets, mock_datasets)
        self.assertEqual(fine_tuner.dataset_info, mock_dataset_info)


    @patch('medgemma_trainer.get_peft_model')
    @patch.object(MedGemmaFineTuner, 'setup_environment')
    def test_setup_lora(self, mock_setup_env, mock_get_peft_model):
        fine_tuner = MedGemmaFineTuner(self.config)
        fine_tuner.model = MagicMock() # Mock the model attribute
        mock_param_trainable = MagicMock()
        mock_param_trainable.requires_grad = True
        mock_param_trainable.numel.return_value = 100

        mock_param_frozen = MagicMock()
        mock_param_frozen.requires_grad = False
        mock_param_frozen.numel.return_value = 900

        fine_tuner.model.parameters.return_value = [mock_param_trainable, mock_param_frozen]

        mock_peft_model = MagicMock()
        # Mock the parameters method for the peft model as well
        mock_peft_param_trainable = MagicMock()
        mock_peft_param_trainable.requires_grad = True
        mock_peft_param_trainable.numel.return_value = 50 # e.g. LoRA params
        mock_peft_param_frozen = MagicMock()
        mock_peft_param_frozen.requires_grad = False
        mock_peft_param_frozen.numel.return_value = 900 # Original frozen params
        mock_peft_model.parameters.return_value = [mock_peft_param_trainable, mock_param_frozen, mock_param_trainable] # some could be duplicated by name

        mock_get_peft_model.return_value = mock_peft_model

        original_model = fine_tuner.model # Capture model before it's replaced by setup_lora
        peft_config_result = fine_tuner.setup_lora()

        mock_get_peft_model.assert_called_once()
        args, kwargs = mock_get_peft_model.call_args
        self.assertEqual(args[0], original_model) # Compare with the model before replacement
        lora_config_arg = args[1]

        self.assertEqual(lora_config_arg.r, self.config.lora.r)
        self.assertEqual(lora_config_arg.lora_alpha, self.config.lora.lora_alpha)
        self.assertEqual(set(lora_config_arg.target_modules), set(self.config.lora.target_modules)) # Compare as sets

        self.assertEqual(fine_tuner.model, mock_peft_model) # Model should be updated to peft model

    @patch.object(MedGemmaFineTuner, 'setup_environment')
    def test_create_collate_function(self, mock_setup_env):
        fine_tuner = MedGemmaFineTuner(self.config)

        # Mock processor and tokenizer for collate_fn
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.pad_token_id = 0
        mock_processor.tokenizer.image_token_id = 262144 # Ensure consistency with test's expected image token
        mock_processor.apply_chat_template.side_effect = lambda x, **kwargs: " ".join(m['content'] for m in x) # Simple mock

        # Example input batch structure
        input_ids = torch.tensor([[101, 1, 2, 3, 102, 262144, 0, 0], [101, 4, 5, 102, 262144, 262144, 0, 0]])
        mock_processor.return_value = { # Output of processor call
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids)
        }
        fine_tuner.processor = mock_processor

        collate_fn = fine_tuner.create_collate_function()

        example_batch = [
            {"image": "dummy_image_1", "messages": [{"role": "user", "content": "text1"}, {"role": "assistant", "content": "resp1"}]},
            {"image": "dummy_image_2", "messages": [{"role": "user", "content": "text2"}, {"role": "assistant", "content": "resp2"}]}
        ]

        processed_batch = collate_fn(example_batch)

        self.assertIn("input_ids", processed_batch)
        self.assertIn("labels", processed_batch)

        expected_labels = input_ids.clone()
        expected_labels[expected_labels == 0] = -100 # pad_token_id
        expected_labels[expected_labels == 262144] = -100 # image token

        self.assertTrue(torch.equal(processed_batch["labels"], expected_labels))
        mock_processor.assert_called_once() # Check processor was called
        # Check apply_chat_template calls
        self.assertEqual(mock_processor.apply_chat_template.call_count, len(example_batch))


    @patch('medgemma_trainer.SFTTrainer')
    @patch.object(MedGemmaFineTuner, 'setup_environment')
    @patch.object(MedGemmaFineTuner, 'load_model_and_processor') # Further mock these
    @patch.object(MedGemmaFineTuner, 'prepare_datasets')
    @patch.object(MedGemmaFineTuner, 'setup_lora')
    @patch.object(MedGemmaFineTuner, 'create_collate_function')
    @patch.object(MedGemmaFineTuner, 'save_training_metadata')
    @patch('os.makedirs') # Mock os.makedirs
    def test_train_method_calls(self, mock_makedirs, mock_save_meta, mock_collate_fn, mock_setup_lora,
                                mock_prepare_datasets, mock_load_model, mock_setup_env, MockSFTTrainer):
        fine_tuner = MedGemmaFineTuner(self.config)

        # Mock attributes that train() relies on
        fine_tuner.model = MagicMock()
        fine_tuner.processor = MagicMock()
        fine_tuner.datasets = {
            "train": MagicMock(),
            "val": MagicMock()
        }
        # Directly mock __len__ on the MagicMock instance
        fine_tuner.datasets["train"].__len__ = MagicMock(return_value=10)
        fine_tuner.datasets["val"].__len__ = MagicMock(return_value=5)

        # Ensure bf16 is False for this test to avoid ValueError from SFTConfig
        # as setup_environment (which might disable it) is mocked.
        fine_tuner.config.training.bf16 = False

        mock_trainer_instance = MockSFTTrainer.return_value
        mock_train_result = MagicMock()
        mock_train_result.training_loss = 0.123
        mock_trainer_instance.train.return_value = mock_train_result

        mock_collate_fn.return_value = lambda x: x # Simple pass-through collator

        trainer_returned = fine_tuner.train()

        mock_makedirs.assert_any_call(self.config.training.output_dir, exist_ok=True)
        MockSFTTrainer.assert_called_once() # Check SFTTrainer was initialized
        mock_trainer_instance.train.assert_called_once() # Check train was called on SFTTrainer
        mock_trainer_instance.save_model.assert_called_once_with(self.config.training.output_dir)
        fine_tuner.processor.save_pretrained.assert_called_once_with(self.config.training.output_dir)
        mock_save_meta.assert_called_once_with(mock_train_result)
        self.assertEqual(trainer_returned, mock_trainer_instance)


    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch.object(Config, 'to_yaml') # Mock the to_yaml method of the Config instance
    @patch.object(MedGemmaFineTuner, 'setup_environment')
    def test_save_training_metadata(self, mock_setup_env, mock_config_to_yaml, mock_json_dump, mock_file_open):
        fine_tuner = MedGemmaFineTuner(self.config)
        fine_tuner.dataset_info = {"key": "value"} # Mock dataset_info

        mock_train_result = MagicMock()
        mock_train_result.training_loss = 0.555

        fine_tuner.save_training_metadata(mock_train_result)

        expected_metadata_path = Path(self.config.training.output_dir) / "training_metadata.json"
        expected_config_path = Path(self.config.training.output_dir) / "config.yaml"

        mock_file_open.assert_any_call(expected_metadata_path, "w")
        mock_json_dump.assert_called_once()
        args, _ = mock_json_dump.call_args
        saved_data = args[0]

        self.assertEqual(saved_data['model_id'], self.config.model.model_id)
        self.assertEqual(saved_data['lora_config']['r'], self.config.lora.r)
        self.assertEqual(saved_data['training_args']['final_loss'], 0.555)
        self.assertEqual(saved_data['dataset_info'], {"key": "value"})

        mock_config_to_yaml.assert_called_once_with(str(expected_config_path))


if __name__ == '__main__':
    unittest.main()
