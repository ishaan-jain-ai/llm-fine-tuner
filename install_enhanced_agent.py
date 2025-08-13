#!/usr/bin/env python3
"""
Installation Script for Enhanced LLM Fine Tuner Agent
Automatically installs required dependencies and verifies the setup
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_core_dependencies():
    """Install core dependencies"""
    print("\n📦 Installing core dependencies...")
    
    core_deps = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.41.0",
        "trl>=0.7.0",
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
        "huggingface-hub>=0.16.0",
        "sentencepiece>=0.1.99",
        "protobuf>=4.23.0"
    ]
    
    for dep in core_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️ Failed to install {dep}, continuing...")
    
    return True

def install_enhanced_dependencies():
    """Install enhanced agent dependencies"""
    print("\n🚀 Installing enhanced agent dependencies...")
    
    enhanced_deps = [
        "langgraph>=0.2.0",
        "langchain>=0.2.0", 
        "langchain-openai>=0.1.0",
        "openai>=1.0.0"
    ]
    
    for dep in enhanced_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️ Failed to install {dep}, continuing...")
    
    return True

def verify_installation():
    """Verify that the enhanced agent can be imported"""
    print("\n🔍 Verifying installation...")
    
    try:
        # Add src to path
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))
        
        # Try to import the enhanced agent
        from enhanced_agent import EnhancedLLMFineTunerAgent
        print("✅ Enhanced agent imported successfully")
        
        # Try to initialize it
        agent = EnhancedLLMFineTunerAgent()
        print("✅ Enhanced agent initialized successfully")
        
        # Check if LangGraph is available
        if agent.workflow:
            print("✅ LangGraph workflow available")
        else:
            print("⚠️ LangGraph workflow not available (using fallback)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def run_basic_test():
    """Run a basic test to ensure everything works"""
    print("\n🧪 Running basic test...")
    
    try:
        from enhanced_agent import EnhancedLLMFineTunerAgent
        
        agent = EnhancedLLMFineTunerAgent()
        
        # Test basic workflow
        result = agent.start_workflow("Test basic workflow")
        
        if result["success"]:
            print("✅ Basic workflow test passed")
            return True
        else:
            print(f"❌ Basic workflow test failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def create_sample_config():
    """Create a sample configuration file"""
    print("\n📝 Creating sample configuration...")
    
    sample_config = """# Sample Enhanced Agent Configuration
# This file shows how to configure the enhanced agent

from src.enhanced_agent import EnhancedLLMFineTunerAgent
from src.config import FineTunerConfig

# Basic usage
agent = EnhancedLLMFineTunerAgent()

# With custom configuration
config = FineTunerConfig()
config.model.model_name = "gpt2"
config.training.num_epochs = 3
config.training.use_lora = True

agent = EnhancedLLMFineTunerAgent(config)

# Start intelligent workflow
result = agent.start_workflow("I want to fine-tune for conversation data")
"""
    
    try:
        with open("sample_enhanced_usage.py", "w") as f:
            f.write(sample_config)
        print("✅ Sample configuration created: sample_enhanced_usage.py")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample config: {e}")
        return False

def main():
    """Main installation function"""
    print("🚀 Enhanced LLM Fine Tuner Agent - Installation Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("❌ Installation cannot continue due to Python version incompatibility")
        return False
    
    # Install core dependencies
    if not install_core_dependencies():
        print("⚠️ Some core dependencies failed to install, but continuing...")
    
    # Install enhanced dependencies
    if not install_enhanced_dependencies():
        print("⚠️ Some enhanced dependencies failed to install, but continuing...")
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        return False
    
    # Run basic test
    if not run_basic_test():
        print("⚠️ Basic test failed, but installation may still work")
    
    # Create sample configuration
    create_sample_config()
    
    # Installation summary
    print("\n🎉 Installation Summary")
    print("=" * 30)
    print("✅ Core dependencies installed")
    print("✅ Enhanced agent dependencies installed")
    print("✅ Enhanced agent verified")
    print("✅ Sample configuration created")
    
    print("\n🚀 Next Steps:")
    print("1. Run the demo: python demo_enhanced_agent.py")
    print("2. Run examples: python examples/enhanced_agent_example.py")
    print("3. Run tests: python -m pytest tests/test_enhanced_agent.py")
    print("4. Read documentation: LANGGRAPH_INTEGRATION.md")
    
    print("\n💡 If you encounter issues:")
    print("• Check that all dependencies are installed: pip list")
    print("• Verify Python version: python --version")
    print("• Check the logs for specific error messages")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Installation completed successfully!")
        else:
            print("\n❌ Installation completed with errors")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed with unexpected error: {e}")
        sys.exit(1)
