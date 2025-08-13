"""
Configuration module for LLM Fine Tuner Agent
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the base model"""
    model_name: str = "microsoft/DialoGPT-medium"
    model_type: str = "causal"  # causal, seq2seq
    max_length: int = 512
    padding_side: str = "right"
    trust_remote_code: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # Basic training params
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # LoRA params
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # QLoRA params
    use_qlora: bool = False
    bf16: bool = True
    fp16: bool = False
    
    # Evaluation
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    
    # Checkpointing
    output_dir: str = "./outputs"
    save_total_limit: Optional[int] = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"


@dataclass
class DataConfig:
    """Configuration for data processing"""
    train_file: str = ""
    validation_file: str = ""
    test_file: str = ""
    text_column: str = "text"
    max_seq_length: int = 512
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    validation_split_percentage: int = 5


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring"""
    use_wandb: bool = False
    use_tensorboard: bool = True
    log_level: str = "INFO"
    project_name: str = "llm-fine-tuner"
    run_name: Optional[str] = None


@dataclass
class FineTunerConfig:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.training.use_qlora and not self.training.bf16:
            raise ValueError("QLoRA requires bf16=True")
        
        if self.training.use_lora and self.training.use_qlora:
            raise ValueError("Cannot use both LoRA and QLoRA simultaneously")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FineTunerConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'logging': self.logging.__dict__
        }
    
    def save(self, filepath: str):
        """Save config to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'FineTunerConfig':
        """Load config from file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Default configurations for common models
DEFAULT_CONFIGS = {
    "gpt2": FineTunerConfig(
        model=ModelConfig(
            model_name="gpt2",
            model_type="causal",
            max_length=512
        ),
        training=TrainingConfig(
            learning_rate=5e-5,
            batch_size=2,
            lora_r=8,
            lora_alpha=16
        )
    ),
    "llama": FineTunerConfig(
        model=ModelConfig(
            model_name="meta-llama/Llama-2-7b-hf",
            model_type="causal",
            max_length=2048
        ),
        training=TrainingConfig(
            use_qlora=True,
            bf16=True,
            learning_rate=2e-4,
            batch_size=1,
            lora_r=64,
            lora_alpha=16
        )
    ),
    "t5": FineTunerConfig(
        model=ModelConfig(
            model_name="t5-base",
            model_type="seq2seq",
            max_length=512
        ),
        training=TrainingConfig(
            learning_rate=3e-5,
            batch_size=8,
            use_lora=False
        )
    )
}
