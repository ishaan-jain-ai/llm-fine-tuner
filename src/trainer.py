"""
Training module for LLM Fine Tuner Agent
"""
import os
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from transformers import Trainer, TrainingArguments
from datasets import DatasetDict
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from .model_manager import ModelManager
from .data_processor import DataProcessor

console = Console()
logger = logging.getLogger(__name__)


class FineTuningTrainer:
    """Handles the fine-tuning process"""
    
    def __init__(self, config):
        self.config = config
        self.model_manager = ModelManager(config)
        self.data_processor = DataProcessor(config)
        self.trainer = None
        self.training_history = []
        
    def setup_training(self, train_file: str, validation_file: Optional[str] = None):
        """Setup the training environment"""
        console.print(Panel.fit("🚀 Setting up fine-tuning environment", style="bold blue"))
        
        # Load model and tokenizer
        model, tokenizer = self.model_manager.load_model_and_tokenizer()
        
        # Setup PEFT adapters
        model = self.model_manager.setup_peft()
        
        # Set tokenizer for data processing
        self.data_processor.set_tokenizer(tokenizer)
        
        # Prepare dataset
        dataset = self.data_processor.prepare_dataset(train_file, validation_file)
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.data_processor.tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # Create data collator
        data_collator = self.data_processor.create_data_collator()
        
        # Get training arguments
        training_args = self.model_manager.get_training_arguments()
        
        # Initialize trainer
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Display setup information
        self._display_setup_info(model, tokenizer, dataset)
        
        return self.trainer
    
    def _display_setup_info(self, model, tokenizer, dataset):
        """Display training setup information"""
        table = Table(title="Training Setup Information")
        table.add_column("Component", style="cyan")
        table.add_column("Details", style="green")
        
        # Model info
        model_info = self.model_manager.get_model_info()
        table.add_row("Model", f"{model_info['model_name']} ({model_info['model_type']})")
        table.add_row("Device", model_info['device'])
        table.add_row("Parameters", f"{model_info['num_parameters']:,}")
        table.add_row("Trainable", f"{model_info['trainable_parameters']:,}")
        
        # Dataset info
        table.add_row("Training Examples", str(len(dataset["train"])))
        table.add_row("Validation Examples", str(len(dataset["validation"])))
        
        # Training config
        table.add_row("Epochs", str(self.config.training.num_epochs))
        table.add_row("Batch Size", str(self.config.training.batch_size))
        table.add_row("Learning Rate", f"{self.config.training.learning_rate:.2e}")
        
        if self.config.training.use_lora:
            table.add_row("LoRA", f"r={self.config.training.lora_r}, α={self.config.training.lora_alpha}")
        if self.config.training.use_qlora:
            table.add_row("QLoRA", "4-bit quantization enabled")
        
        console.print(table)
    
    def train(self) -> Dict[str, Any]:
        """Execute the training process"""
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call setup_training() first.")
        
        console.print(Panel.fit("🎯 Starting fine-tuning process", style="bold green"))
        
        start_time = time.time()
        
        try:
            # Start training
            train_result = self.trainer.train()
            
            # Save the final model
            self.model_manager.save_model(self.config.training.output_dir)
            
            # Evaluate final model
            eval_result = self.trainer.evaluate()
            
            training_time = time.time() - start_time
            
            # Compile results
            results = {
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'training_time': training_time,
                'train_metrics': train_result.metrics,
                'eval_metrics': eval_result
            }
            
            # Display results
            self._display_training_results(results)
            
            return results
            
        except Exception as e:
            console.print(f"[red]Training failed: {str(e)}[/red]")
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def _display_training_results(self, results: Dict[str, Any]):
        """Display training results"""
        table = Table(title="Training Results", style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Training Loss", f"{results['train_loss']:.4f}")
        table.add_row("Evaluation Loss", f"{results['eval_loss']:.4f}")
        table.add_row("Training Time", f"{results['training_time']/60:.1f} minutes")
        
        if 'eval_perplexity' in results['eval_metrics']:
            table.add_row("Perplexity", f"{results['eval_metrics']['eval_perplexity']:.2f}")
        
        console.print(table)
        
        # Save results to file
        results_file = os.path.join(self.config.training.output_dir, "training_results.json")
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"📊 Results saved to: {results_file}")
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from a checkpoint"""
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call setup_training() first.")
        
        console.print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
        
        # Resume training
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint_path)
        
        # Save the final model
        self.model_manager.save_model(self.config.training.output_dir)
        
        return train_result
    
    def evaluate_model(self, eval_dataset=None) -> Dict[str, Any]:
        """Evaluate the trained model"""
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call setup_training() first.")
        
        console.print(Panel.fit("🔍 Evaluating model", style="bold yellow"))
        
        if eval_dataset:
            eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
        else:
            eval_result = self.trainer.evaluate()
        
        # Display evaluation results
        self._display_evaluation_results(eval_result)
        
        return eval_result
    
    def _display_evaluation_results(self, eval_result: Dict[str, Any]):
        """Display evaluation results"""
        table = Table(title="Evaluation Results", style="bold yellow")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        for key, value in eval_result.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))
        
        console.print(table)
    
    def generate_sample(self, prompt: str, max_length: int = 100) -> str:
        """Generate a sample from the trained model"""
        if not self.trainer or not self.trainer.model:
            raise ValueError("Model not available. Train the model first.")
        
        from .model_manager import ModelEvaluator
        
        evaluator = ModelEvaluator(
            self.trainer.model, 
            self.trainer.tokenizer, 
            self.model_manager.device
        )
        
        generated_text = evaluator.generate_text(prompt, max_length)
        
        console.print(Panel.fit(f"Generated Text:\n{generated_text}", style="bold magenta"))
        
        return generated_text
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Save a training checkpoint"""
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call setup_training() first.")
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{self.trainer.state.global_step}")
        self.trainer.save_model(checkpoint_path)
        
        console.print(f"💾 Checkpoint saved to: {checkpoint_path}")
        return checkpoint_path
    
    def cleanup(self):
        """Clean up training resources"""
        if self.trainer:
            del self.trainer
        
        if self.model_manager:
            self.model_manager.cleanup()
        
        console.print("🧹 Training resources cleaned up")


class TrainingMonitor:
    """Monitors training progress and provides callbacks"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_history = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Callback for logging during training"""
        if logs:
            self.metrics_history.append(logs.copy())
            
            # Log to console
            if 'loss' in logs:
                console.print(f"📈 Step {state.global_step}: Loss = {logs['loss']:.4f}")
            
            if 'eval_loss' in logs:
                console.print(f"📊 Evaluation: Loss = {logs['eval_loss']:.4f}")
    
    def on_save(self, args, state, control, **kwargs):
        """Callback for model saving"""
        console.print(f"💾 Model saved at step {state.global_step}")
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Callback for evaluation"""
        console.print(f"🔍 Evaluation completed at step {state.global_step}")
    
    def get_training_curve(self) -> Dict[str, List[float]]:
        """Get training curve data for plotting"""
        if not self.metrics_history:
            return {}
        
        curves = {}
        for metric in self.metrics_history[0].keys():
            if isinstance(self.metrics_history[0][metric], (int, float)):
                curves[metric] = [log[metric] for log in self.metrics_history if metric in log]
        
        return curves
