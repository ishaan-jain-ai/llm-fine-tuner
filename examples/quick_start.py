#!/usr/bin/env python3
"""
Quick Start Example - LLM Fine Tuner Agent
"""
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import FineTunerConfig, DEFAULT_CONFIGS
from agent import LLMFineTunerAgent


def quick_start_example():
    """Quick start example using GPT-2"""
    
    print("🚀 LLM Fine Tuner Agent - Quick Start Example")
    print("=" * 50)
    
    # Use GPT-2 preset configuration
    config = DEFAULT_CONFIGS["gpt2"]
    
    # Modify some settings for quick testing
    config.training.num_epochs = 1  # Just 1 epoch for quick demo
    config.training.batch_size = 2
    config.training.learning_rate = 5e-5
    config.training.output_dir = "./quick_start_output"
    
    # Create sample training data
    sample_data_file = "./sample_training_data.json"
    if not os.path.exists(sample_data_file):
        print("📝 Creating sample training data...")
        agent = LLMFineTunerAgent(config)
        agent.create_sample_data(sample_data_file, num_examples=50)
    
    # Set the training data file
    config.data.train_file = sample_data_file
    
    print(f"✅ Using configuration:")
    print(f"   Model: {config.model.model_name}")
    print(f"   Training data: {config.data.train_file}")
    print(f"   Output directory: {config.training.output_dir}")
    print(f"   Epochs: {config.training.num_epochs}")
    print(f"   Batch size: {config.training.batch_size}")
    
    # Initialize agent
    agent = LLMFineTunerAgent(config)
    
    try:
        # Start training
        print("\n🎯 Starting training...")
        results = agent.start_training()
        
        print(f"\n🎉 Training completed!")
        print(f"   Training loss: {results['train_loss']:.4f}")
        print(f"   Evaluation loss: {results['eval_loss']:.4f}")
        print(f"   Training time: {results['training_time']/60:.1f} minutes")
        
        # Generate sample text
        print("\n✨ Generating sample text...")
        prompt = "Human: What is artificial intelligence?"
        generated = agent.generate_text(prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise
    finally:
        agent.cleanup()


def custom_model_example():
    """Example with custom model configuration"""
    
    print("\n🔧 Custom Model Configuration Example")
    print("=" * 50)
    
    # Create custom configuration
    config = FineTunerConfig()
    
    # Model settings
    config.model.model_name = "microsoft/DialoGPT-small"
    config.model.model_type = "causal"
    config.model.max_length = 256
    
    # Training settings
    config.training.num_epochs = 2
    config.training.batch_size = 4
    config.training.learning_rate = 3e-5
    config.training.use_lora = True
    config.training.lora_r = 8
    config.training.lora_alpha = 16
    config.training.output_dir = "./custom_model_output"
    
    # Data settings
    config.data.train_file = "./sample_training_data.json"
    config.data.max_seq_length = 256
    
    print(f"✅ Custom configuration:")
    print(f"   Model: {config.model.model_name}")
    print(f"   LoRA: r={config.training.lora_r}, α={config.training.lora_alpha}")
    print(f"   Output directory: {config.training.output_dir}")
    
    # Initialize agent
    agent = LLMFineTunerAgent(config)
    
    try:
        # Start training
        print("\n🎯 Starting training with custom config...")
        results = agent.start_training()
        
        print(f"\n🎉 Custom training completed!")
        print(f"   Training loss: {results['train_loss']:.4f}")
        print(f"   Evaluation loss: {results['eval_loss']:.4f}")
        
    except Exception as e:
        print(f"❌ Error during custom training: {str(e)}")
        raise
    finally:
        agent.cleanup()


if __name__ == "__main__":
    try:
        # Run quick start example
        quick_start_example()
        
        # Run custom model example
        custom_model_example()
        
        print("\n✅ All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed: {str(e)}")
        import traceback
        traceback.print_exc()
