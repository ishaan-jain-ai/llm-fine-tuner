"""
Unit tests for configuration module
"""
import unittest
import tempfile
import os
from pathlib import Path

# Add src to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import FineTunerConfig, ModelConfig, TrainingConfig, DataConfig, LoggingConfig


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = ModelConfig()
        
        self.assertEqual(config.model_name, "microsoft/DialoGPT-medium")
        self.assertEqual(config.model_type, "causal")
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.padding_side, "right")
        self.assertFalse(config.trust_remote_code)
    
    def test_custom_values(self):
        """Test custom configuration values"""
        config = ModelConfig(
            model_name="gpt2",
            model_type="causal",
            max_length=1024
        )
        
        self.assertEqual(config.model_name, "gpt2")
        self.assertEqual(config.model_type, "causal")
        self.assertEqual(config.max_length, 1024)


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig class"""
    
    def test_default_values(self):
        """Test default training configuration values"""
        config = TrainingConfig()
        
        self.assertEqual(config.num_epochs, 3)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertTrue(config.use_lora)
        self.assertFalse(config.use_qlora)
    
    def test_lora_configuration(self):
        """Test LoRA configuration"""
        config = TrainingConfig(
            use_lora=True,
            lora_r=32,
            lora_alpha=64
        )
        
        self.assertTrue(config.use_lora)
        self.assertEqual(config.lora_r, 32)
        self.assertEqual(config.lora_alpha, 64)


class TestDataConfig(unittest.TestCase):
    """Test DataConfig class"""
    
    def test_default_values(self):
        """Test default data configuration values"""
        config = DataConfig()
        
        self.assertEqual(config.text_column, "text")
        self.assertEqual(config.max_seq_length, 512)
        self.assertEqual(config.validation_split_percentage, 5)
    
    def test_custom_values(self):
        """Test custom data configuration values"""
        config = DataConfig(
            train_file="train.json",
            validation_file="val.json",
            text_column="content"
        )
        
        self.assertEqual(config.train_file, "train.json")
        self.assertEqual(config.validation_file, "val.json")
        self.assertEqual(config.text_column, "content")


class TestFineTunerConfig(unittest.TestCase):
    """Test FineTunerConfig class"""
    
    def test_default_configuration(self):
        """Test default fine tuner configuration"""
        config = FineTunerConfig()
        
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.logging, LoggingConfig)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test QLoRA validation
        config = FineTunerConfig()
        config.training.use_qlora = True
        config.training.bf16 = False
        
        with self.assertRaises(ValueError):
            config.__post_init__()
        
        # Test LoRA + QLoRA conflict
        config.training.bf16 = True
        config.training.use_lora = True
        
        with self.assertRaises(ValueError):
            config.__post_init__()
    
    def test_save_and_load(self):
        """Test configuration save and load functionality"""
        config = FineTunerConfig()
        config.model.model_name = "test-model"
        config.training.learning_rate = 1e-4
        
        # Save configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save(f.name)
            temp_file = f.name
        
        try:
            # Load configuration
            loaded_config = FineTunerConfig.load(temp_file)
            
            # Verify loaded values
            self.assertEqual(loaded_config.model.model_name, "test-model")
            self.assertEqual(loaded_config.training.learning_rate, 1e-4)
            
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_to_dict(self):
        """Test configuration to dictionary conversion"""
        config = FineTunerConfig()
        config.model.model_name = "custom-model"
        
        config_dict = config.to_dict()
        
        self.assertIn('model', config_dict)
        self.assertIn('training', config_dict)
        self.assertIn('data', config_dict)
        self.assertIn('logging', config_dict)
        
        self.assertEqual(config_dict['model']['model_name'], "custom-model")


class TestDefaultConfigs(unittest.TestCase):
    """Test default configuration presets"""
    
    def test_gpt2_preset(self):
        """Test GPT-2 preset configuration"""
        from config import DEFAULT_CONFIGS
        
        gpt2_config = DEFAULT_CONFIGS["gpt2"]
        
        self.assertEqual(gpt2_config.model.model_name, "gpt2")
        self.assertEqual(gpt2_config.model.model_type, "causal")
        self.assertEqual(gpt2_config.training.learning_rate, 5e-5)
        self.assertTrue(gpt2_config.training.use_lora)
    
    def test_llama_preset(self):
        """Test LLaMA preset configuration"""
        from config import DEFAULT_CONFIGS
        
        llama_config = DEFAULT_CONFIGS["llama"]
        
        self.assertEqual(llama_config.model.model_name, "meta-llama/Llama-2-7b-hf")
        self.assertTrue(llama_config.training.use_qlora)
        self.assertTrue(llama_config.training.bf16)
    
    def test_t5_preset(self):
        """Test T5 preset configuration"""
        from config import DEFAULT_CONFIGS
        
        t5_config = DEFAULT_CONFIGS["t5"]
        
        self.assertEqual(t5_config.model.model_name, "t5-base")
        self.assertEqual(t5_config.model.model_type, "seq2seq")
        self.assertFalse(t5_config.training.use_lora)


if __name__ == '__main__':
    unittest.main()
