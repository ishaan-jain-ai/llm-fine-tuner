"""
Main LLM Fine Tuner Agent
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

from .config import FineTunerConfig, DEFAULT_CONFIGS
from .trainer import FineTuningTrainer, TrainingMonitor
from .data_processor import DataProcessor, ConversationDataProcessor

console = Console()
logger = logging.getLogger(__name__)


class LLMFineTunerAgent:
    """Main agent for LLM fine-tuning"""
    
    def __init__(self, config: Optional[FineTunerConfig] = None):
        self.config = config or FineTunerConfig()
        self.trainer = None
        self.data_processor = None
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.training.output_dir, exist_ok=True)
    
    def interactive_setup(self) -> FineTunerConfig:
        """Interactive setup for fine-tuning configuration"""
        console.print(Panel.fit("🤖 LLM Fine Tuner Agent - Interactive Setup", style="bold blue"))
        
        # Model selection
        console.print("\n📚 Available model presets:")
        preset_table = Table()
        preset_table.add_column("Preset", style="cyan")
        preset_table.add_column("Model", style="green")
        preset_table.add_column("Type", style="yellow")
        
        for name, preset_config in DEFAULT_CONFIGS.items():
            preset_table.add_row(
                name,
                preset_config.model.model_name,
                preset_config.model.model_type
            )
        
        console.print(preset_table)
        
        preset_choice = Prompt.ask(
            "Choose a preset (or 'custom' for manual setup)",
            choices=list(DEFAULT_CONFIGS.keys()) + ['custom'],
            default='gpt2'
        )
        
        if preset_choice != 'custom':
            self.config = DEFAULT_CONFIGS[preset_choice]
            console.print(f"✅ Using {preset_choice} preset")
        else:
            self.config = self._custom_model_setup()
        
        # Data configuration
        self._setup_data_config()
        
        # Training configuration
        self._setup_training_config()
        
        # Save configuration
        config_file = os.path.join(self.config.training.output_dir, "config.json")
        self.config.save(config_file)
        console.print(f"💾 Configuration saved to: {config_file}")
        
        return self.config
    
    def _custom_model_setup(self) -> FineTunerConfig:
        """Custom model setup"""
        console.print("\n🔧 Custom Model Setup")
        
        model_name = Prompt.ask("Enter model name", default="gpt2")
        model_type = Prompt.ask(
            "Model type",
            choices=["causal", "seq2seq"],
            default="causal"
        )
        
        config = FineTunerConfig()
        config.model.model_name = model_name
        config.model.model_type = model_type
        
        return config
    
    def _setup_data_config(self):
        """Setup data configuration"""
        console.print("\n📊 Data Configuration")
        
        train_file = Prompt.ask("Training data file path")
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        
        self.config.data.train_file = train_file
        
        # Check if validation file exists
        validation_file = Prompt.ask(
            "Validation data file path (optional, press Enter to auto-split)",
            default=""
        )
        
        if validation_file and os.path.exists(validation_file):
            self.config.data.validation_file = validation_file
        elif validation_file:
            console.print(f"⚠️  Validation file not found: {validation_file}")
        
        # Data format detection
        if self._detect_conversation_format(train_file):
            console.print("💬 Detected conversation format data")
            use_conversation = Confirm.ask("Use conversation data processor?")
            if use_conversation:
                self.data_processor = ConversationDataProcessor(self.config)
                format_type = Prompt.ask(
                    "Conversation format",
                    choices=["dialogue", "qa", "instruction"],
                    default="dialogue"
                )
                self.data_processor.set_conversation_format(format_type)
    
    def _setup_training_config(self):
        """Setup training configuration"""
        console.print("\n⚙️  Training Configuration")
        
        # Basic training params
        self.config.training.num_epochs = int(Prompt.ask(
            "Number of epochs",
            default=str(self.config.training.num_epochs)
        ))
        
        self.config.training.batch_size = int(Prompt.ask(
            "Batch size",
            default=str(self.config.training.batch_size)
        ))
        
        self.config.training.learning_rate = float(Prompt.ask(
            "Learning rate",
            default=str(self.config.training.learning_rate)
        ))
        
        # LoRA/QLoRA setup
        use_peft = Confirm.ask("Use PEFT (LoRA/QLoRA)?", default=True)
        
        if use_peft:
            if self.config.model.model_type == "causal":
                use_qlora = Confirm.ask("Use QLoRA (4-bit quantization)?", default=False)
                if use_qlora:
                    self.config.training.use_qlora = True
                    self.config.training.bf16 = True
                    self.config.training.use_lora = False
                else:
                    self.config.training.use_lora = True
                    self.config.training.use_qlora = False
                    
                    # LoRA parameters
                    self.config.training.lora_r = int(Prompt.ask(
                        "LoRA rank (r)",
                        default=str(self.config.training.lora_r)
                    ))
                    self.config.training.lora_alpha = int(Prompt.ask(
                        "LoRA alpha",
                        default=str(self.config.training.lora_alpha)
                    ))
            else:
                console.print("⚠️  PEFT only supported for causal models")
    
    def _detect_conversation_format(self, file_path: str) -> bool:
        """Detect if data is in conversation format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
                # Check for common conversation patterns
                conversation_indicators = [
                    "Human:", "Assistant:", "Question:", "Answer:",
                    "Instruction:", "Response:", "### Instruction:"
                ]
                
                return any(indicator in first_line for indicator in conversation_indicators)
        except:
            return False
    
    def start_training(self) -> Dict[str, Any]:
        """Start the fine-tuning process"""
        if not self.config.data.train_file:
            raise ValueError("Training data file not configured")
        
        console.print(Panel.fit("🚀 Starting Fine-Tuning Process", style="bold green"))
        
        try:
            # Initialize trainer
            self.trainer = FineTuningTrainer(self.config)
            
            # Setup training environment
            self.trainer.setup_training(
                self.config.data.train_file,
                self.config.data.validation_file
            )
            
            # Start training
            results = self.trainer.train()
            
            return results
            
        except Exception as e:
            console.print(f"[red]Training failed: {str(e)}[/red]")
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            if self.trainer:
                self.trainer.cleanup()
    
    def evaluate_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate a trained model"""
        if not model_path:
            model_path = self.config.training.output_dir
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        console.print(Panel.fit("🔍 Model Evaluation", style="bold yellow"))
        
        # Initialize trainer for evaluation
        self.trainer = FineTuningTrainer(self.config)
        
        try:
            # Setup training environment
            self.trainer.setup_training(
                self.config.data.train_file,
                self.config.data.validation_file
            )
            
            # Evaluate model
            eval_results = self.trainer.evaluate_model()
            
            return eval_results
            
        finally:
            if self.trainer:
                self.trainer.cleanup()
    
    def generate_text(self, prompt: str, model_path: Optional[str] = None, max_length: int = 100) -> str:
        """Generate text using the fine-tuned model"""
        if not model_path:
            model_path = self.config.training.output_dir
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        console.print(Panel.fit("✨ Text Generation", style="bold magenta"))
        
        # Initialize trainer for generation
        self.trainer = FineTuningTrainer(self.config)
        
        try:
            # Setup training environment
            self.trainer.setup_training(
                self.config.data.train_file,
                self.config.data.validation_file
            )
            
            # Generate text
            generated_text = self.trainer.generate_sample(prompt, max_length)
            
            return generated_text
            
        finally:
            if self.trainer:
                self.trainer.cleanup()
    
    def create_sample_data(self, output_file: str, num_examples: int = 100):
        """Create sample training data for testing"""
        console.print(Panel.fit("📝 Creating Sample Training Data", style="bold cyan"))
        
        # Create sample conversation data
        sample_data = []
        
        for i in range(num_examples):
            conversation = f"Human: What is the capital of France?\nAssistant: The capital of France is Paris. It's a beautiful city known for its art, culture, and iconic landmarks like the Eiffel Tower.\nHuman: Tell me more about Paris.\nAssistant: Paris is the capital and largest city of France, located in the north-central part of the country. It's known as the 'City of Light' and is famous for its rich history, art museums, fashion, and cuisine. The city is home to world-renowned landmarks like the Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées."
            
            sample_data.append({
                "conversation": conversation,
                "category": "geography",
                "difficulty": "easy"
            })
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
        
        console.print(f"✅ Sample data created: {output_file}")
        console.print(f"📊 {num_examples} examples generated")
    
    def show_status(self):
        """Show current agent status"""
        console.print(Panel.fit("📊 Agent Status", style="bold blue"))
        
        status_table = Table()
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="yellow")
        
        # Configuration status
        status_table.add_row(
            "Configuration",
            "✅ Loaded" if self.config else "❌ Not loaded",
            f"Model: {self.config.model.model_name}" if self.config else "N/A"
        )
        
        # Data status
        if self.config and self.config.data.train_file:
            data_exists = os.path.exists(self.config.data.train_file)
            status_table.add_row(
                "Training Data",
                "✅ Found" if data_exists else "❌ Not found",
                self.config.data.train_file
            )
        else:
            status_table.add_row("Training Data", "❌ Not configured", "N/A")
        
        # Output directory status
        if self.config:
            output_exists = os.path.exists(self.config.training.output_dir)
            status_table.add_row(
                "Output Directory",
                "✅ Exists" if output_exists else "❌ Not created",
                self.config.training.output_dir
            )
        else:
            status_table.add_row("Output Directory", "❌ Not configured", "N/A")
        
        # Trainer status
        status_table.add_row(
            "Trainer",
            "✅ Initialized" if self.trainer else "❌ Not initialized",
            "Ready for training" if self.trainer else "N/A"
        )
        
        console.print(status_table)
    
    def cleanup(self):
        """Clean up agent resources"""
        if self.trainer:
            self.trainer.cleanup()
        
        console.print("🧹 Agent resources cleaned up")


def main():
    """Main entry point for the agent"""
    try:
        # Initialize agent
        agent = LLMFineTunerAgent()
        
        # Interactive setup
        config = agent.interactive_setup()
        
        # Show status
        agent.show_status()
        
        # Start training
        if Confirm.ask("Start training now?"):
            results = agent.start_training()
            console.print(f"🎉 Training completed! Results: {results}")
        
        # Generate sample text
        if Confirm.ask("Generate sample text?"):
            prompt = Prompt.ask("Enter a prompt", default="Human: What is machine learning?")
            generated = agent.generate_text(prompt)
            console.print(f"Generated: {generated}")
        
    except KeyboardInterrupt:
        console.print("\n⚠️  Operation cancelled by user")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"Error in main: {str(e)}")
    finally:
        if 'agent' in locals():
            agent.cleanup()


if __name__ == "__main__":
    main()
