"""
Model management module for LLM Fine Tuner Agent
"""
import os
import torch
from typing import Optional, Dict, Any, Tuple
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model,
    prepare_model_for_kbit_training
)
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading, configuration, and PEFT adapters"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        
    def _get_device(self) -> torch.device:
        """Get the appropriate device for training"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        return device
    
    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load the base model and tokenizer"""
        logger.info(f"Loading model: {self.config.model.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=self.config.model.trust_remote_code,
            padding_side=self.config.model.padding_side
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token = "</s>"
        
        # Load model
        if self.config.model.model_type == "causal":
            self.model = self._load_causal_model()
        else:  # seq2seq
            self.model = self._load_seq2seq_model()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        logger.info(f"Model loaded successfully on {self.device}")
        return self.model, self.tokenizer
    
    def _load_causal_model(self):
        """Load causal language model"""
        if self.config.training.use_qlora:
            return self._load_qlora_model()
        else:
            return self._load_standard_model()
    
    def _load_seq2seq_model(self):
        """Load sequence-to-sequence model"""
        return AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=torch.float16 if self.config.training.fp16 else torch.float32
        )
    
    def _load_standard_model(self):
        """Load standard model without quantization"""
        return AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=torch.float16 if self.config.training.fp16 else torch.float32
        )
    
    def _load_qlora_model(self):
        """Load model with QLoRA quantization"""
        if not self.config.training.bf16:
            raise ValueError("QLoRA requires bf16=True")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=self.config.model.trust_remote_code,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        return model
    
    def setup_peft(self) -> Any:
        """Setup PEFT adapters (LoRA/QLoRA)"""
        if not self.config.training.use_lora and not self.config.training.use_qlora:
            logger.info("No PEFT adapters configured, using full fine-tuning")
            return self.model
        
        logger.info("Setting up PEFT adapters")
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.training.lora_r,
            lora_alpha=self.config.training.lora_alpha,
            lora_dropout=self.config.training.lora_dropout,
            target_modules=self.config.training.target_modules,
            bias="none"
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        return self.model
    
    def get_training_arguments(self) -> TrainingArguments:
        """Get training arguments for the trainer"""
        return TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=self.config.training.warmup_steps,
            max_grad_norm=self.config.training.max_grad_norm,
            evaluation_strategy="steps",
            eval_steps=self.config.training.eval_steps,
            save_strategy="steps",
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=False,  # For loss metrics
            logging_steps=self.config.training.logging_steps,
            logging_dir=f"{self.config.training.output_dir}/logs",
            report_to=["tensorboard"] if self.config.logging.use_tensorboard else [],
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            bf16=self.config.training.bf16,
            fp16=self.config.training.fp16,
            dataloader_num_workers=self.config.data.preprocessing_num_workers,
            group_by_length=True,  # Group similar length sequences
            ddp_find_unused_parameters=False,
        )
    
    def save_model(self, output_dir: str, save_tokenizer: bool = True):
        """Save the fine-tuned model"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save PEFT model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
            logger.info(f"Model saved to {output_dir}")
        
        # Save tokenizer
        if save_tokenizer and self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Tokenizer saved to {output_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint"""
        if os.path.isdir(checkpoint_path):
            # Load PEFT model
            if hasattr(self.model, 'load_state_dict'):
                checkpoint = torch.load(
                    os.path.join(checkpoint_path, "pytorch_model.bin"),
                    map_location=self.device
                )
                self.model.load_state_dict(checkpoint)
                logger.info(f"Checkpoint loaded from {checkpoint_path}")
            else:
                logger.warning("Model doesn't support loading checkpoints")
        else:
            raise ValueError(f"Checkpoint path must be a directory: {checkpoint_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            'model_name': self.config.model.model_name,
            'model_type': self.config.model.model_type,
            'device': str(self.device),
            'dtype': str(next(self.model.parameters()).dtype) if self.model else None,
            'num_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0,
        }
        
        if hasattr(self.model, 'config'):
            info['config'] = {
                'vocab_size': getattr(self.model.config, 'vocab_size', 'N/A'),
                'hidden_size': getattr(self.model.config, 'hidden_size', 'N/A'),
                'num_layers': getattr(self.model.config, 'num_hidden_layers', 'N/A'),
                'max_position_embeddings': getattr(self.model.config, 'max_position_embeddings', 'N/A'),
            }
        
        return info
    
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model resources cleaned up")


class ModelEvaluator:
    """Handles model evaluation and metrics"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate text from a prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):]  # Remove the input prompt
    
    def evaluate_perplexity(self, eval_dataset) -> float:
        """Calculate perplexity on evaluation dataset"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in eval_dataset:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                # Count non-padding tokens
                num_tokens = (inputs['attention_mask'] == 1).sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
