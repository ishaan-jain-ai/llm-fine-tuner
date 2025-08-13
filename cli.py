#!/usr/bin/env python3
"""
Command Line Interface for LLM Fine Tuner Agent
"""
import click
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent import LLMFineTunerAgent
from config import FineTunerConfig, DEFAULT_CONFIGS


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """LLM Fine Tuner Agent - Command Line Interface"""
    pass


@cli.command()
@click.option('--preset', '-p', 
              type=click.Choice(list(DEFAULT_CONFIGS.keys()) + ['custom']),
              default='gpt2',
              help='Model preset to use')
@click.option('--train-file', '-t', 
              type=click.Path(exists=True),
              required=True,
              help='Training data file path')
@click.option('--validation-file', '-v',
              type=click.Path(exists=True),
              help='Validation data file path (optional)')
@click.option('--output-dir', '-o',
              default='./outputs',
              help='Output directory for model and logs')
@click.option('--epochs', '-e',
              default=3,
              type=int,
              help='Number of training epochs')
@click.option('--batch-size', '-b',
              default=4,
              type=int,
              help='Training batch size')
@click.option('--learning-rate', '-l',
              default=2e-5,
              type=float,
              help='Learning rate')
@click.option('--use-lora/--no-lora',
              default=True,
              help='Use LoRA for parameter-efficient training')
@click.option('--use-qlora/--no-qlora',
              default=False,
              help='Use QLoRA for 4-bit quantization')
@click.option('--lora-r',
              default=16,
              type=int,
              help='LoRA rank')
@click.option('--lora-alpha',
              default=32,
              type=int,
              help='LoRA alpha')
def train(preset, train_file, validation_file, output_dir, epochs, 
          batch_size, learning_rate, use_lora, use_qlora, lora_r, lora_alpha):
    """Train a model with the specified configuration"""
    
    click.echo("🚀 LLM Fine Tuner Agent - Training Mode")
    
    try:
        # Load preset configuration
        if preset == 'custom':
            config = FineTunerConfig()
        else:
            config = DEFAULT_CONFIGS[preset].copy()
        
        # Update configuration with CLI options
        config.data.train_file = str(train_file)
        if validation_file:
            config.data.validation_file = str(validation_file)
        
        config.training.output_dir = output_dir
        config.training.num_epochs = epochs
        config.training.batch_size = batch_size
        config.training.learning_rate = learning_rate
        
        if use_qlora:
            config.training.use_qlora = True
            config.training.use_lora = False
            config.training.bf16 = True
        elif use_lora:
            config.training.use_lora = True
            config.training.use_qlora = False
            config.training.lora_r = lora_r
            config.training.lora_alpha = lora_alpha
        
        # Initialize agent and start training
        agent = LLMFineTunerAgent(config)
        results = agent.start_training()
        
        click.echo(f"🎉 Training completed successfully!")
        click.echo(f"   Training loss: {results['train_loss']:.4f}")
        click.echo(f"   Evaluation loss: {results['eval_loss']:.4f}")
        click.echo(f"   Training time: {results['training_time']/60:.1f} minutes")
        click.echo(f"   Model saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Training failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model-path', '-m',
              type=click.Path(exists=True),
              required=True,
              help='Path to trained model')
@click.option('--prompt', '-p',
              default="Human: What is artificial intelligence?",
              help='Text prompt for generation')
@click.option('--max-length', '-l',
              default=100,
              type=int,
              help='Maximum length of generated text')
def generate(model_path, prompt, max_length):
    """Generate text using a trained model"""
    
    click.echo("✨ LLM Fine Tuner Agent - Text Generation")
    
    try:
        # Load configuration from model path
        config_file = Path(model_path) / "config.json"
        if config_file.exists():
            config = FineTunerConfig.load(str(config_file))
        else:
            click.echo("⚠️  No config.json found, using default configuration")
            config = FineTunerConfig()
        
        # Initialize agent
        agent = LLMFineTunerAgent(config)
        
        # Generate text
        generated = agent.generate_text(prompt, model_path, max_length)
        
        click.echo(f"\n📝 Generated Text:")
        click.echo(f"Prompt: {prompt}")
        click.echo(f"Generated: {generated}")
        
    except Exception as e:
        click.echo(f"❌ Generation failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model-path', '-m',
              type=click.Path(exists=True),
              required=True,
              help='Path to trained model')
@click.option('--train-file', '-t',
              type=click.Path(exists=True),
              required=True,
              help='Training data file for evaluation')
def evaluate(model_path, train_file):
    """Evaluate a trained model"""
    
    click.echo("🔍 LLM Fine Tuner Agent - Model Evaluation")
    
    try:
        # Load configuration from model path
        config_file = Path(model_path) / "config.json"
        if config_file.exists():
            config = FineTunerConfig.load(str(config_file))
        else:
            click.echo("⚠️  No config.json found, using default configuration")
            config = FineTunerConfig()
        
        config.data.train_file = str(train_file)
        
        # Initialize agent
        agent = LLMFineTunerAgent(config)
        
        # Evaluate model
        eval_results = agent.evaluate_model(model_path)
        
        click.echo(f"\n📊 Evaluation Results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                click.echo(f"   {key}: {value:.4f}")
            else:
                click.echo(f"   {key}: {value}")
        
    except Exception as e:
        click.echo(f"❌ Evaluation failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output-file', '-o',
              default='./sample_training_data.json',
              help='Output file for sample data')
@click.option('--num-examples', '-n',
              default=100,
              type=int,
              help='Number of examples to generate')
def create_sample_data(output_file, num_examples):
    """Create sample training data for testing"""
    
    click.echo("📝 LLM Fine Tuner Agent - Sample Data Creation")
    
    try:
        # Initialize agent with default config
        agent = LLMFineTunerAgent()
        
        # Create sample data
        agent.create_sample_data(output_file, num_examples)
        
        click.echo(f"✅ Sample data created successfully!")
        click.echo(f"   File: {output_file}")
        click.echo(f"   Examples: {num_examples}")
        
    except Exception as e:
        click.echo(f"❌ Sample data creation failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def interactive():
    """Start interactive mode for guided setup"""
    
    try:
        agent = LLMFineTunerAgent()
        agent.interactive_setup()
        
        # Show status
        agent.show_status()
        
        # Ask if user wants to start training
        if click.confirm("Start training now?"):
            results = agent.start_training()
            click.echo(f"🎉 Training completed! Results: {results}")
        
        # Ask if user wants to generate text
        if click.confirm("Generate sample text?"):
            prompt = click.prompt("Enter a prompt", default="Human: What is machine learning?")
            generated = agent.generate_text(prompt)
            click.echo(f"Generated: {generated}")
        
    except Exception as e:
        click.echo(f"❌ Interactive mode failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def list_presets():
    """List available model presets"""
    
    click.echo("📚 Available Model Presets")
    click.echo("=" * 50)
    
    for name, config in DEFAULT_CONFIGS.items():
        click.echo(f"\n🔹 {name.upper()}")
        click.echo(f"   Model: {config.model.model_name}")
        click.echo(f"   Type: {config.model.model_type}")
        click.echo(f"   LoRA: {'✅' if config.training.use_lora else '❌'}")
        click.echo(f"   QLoRA: {'✅' if config.training.use_qlora else '❌'}")
        click.echo(f"   Learning Rate: {config.training.learning_rate:.2e}")
        click.echo(f"   Batch Size: {config.training.batch_size}")


@cli.command()
@click.option('--config-file', '-c',
              type=click.Path(exists=True),
              required=True,
              help='Configuration file path')
def validate_config(config_file):
    """Validate a configuration file"""
    
    click.echo("✅ LLM Fine Tuner Agent - Configuration Validation")
    
    try:
        # Load configuration
        config = FineTunerConfig.load(config_file)
        
        click.echo(f"✅ Configuration loaded successfully!")
        click.echo(f"   Model: {config.model.model_name}")
        click.echo(f"   Type: {config.model.model_type}")
        click.echo(f"   Training epochs: {config.training.num_epochs}")
        click.echo(f"   Batch size: {config.training.batch_size}")
        click.echo(f"   Learning rate: {config.training.learning_rate:.2e}")
        
        if config.training.use_lora:
            click.echo(f"   LoRA: r={config.training.lora_r}, α={config.training.lora_alpha}")
        if config.training.use_qlora:
            click.echo(f"   QLoRA: 4-bit quantization enabled")
        
        # Validate configuration
        config.__post_init__()
        click.echo("✅ Configuration validation passed!")
        
    except Exception as e:
        click.echo(f"❌ Configuration validation failed: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
