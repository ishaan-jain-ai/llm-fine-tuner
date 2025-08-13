# LLM Fine Tuner Agent

A comprehensive, user-friendly agent for fine-tuning Large Language Models (LLMs) with support for LoRA, QLoRA, and various data formats.

## 🚀 Features

- **Multiple Model Support**: GPT-2, LLaMA, T5, and custom models
- **Efficient Fine-tuning**: LoRA and QLoRA support for parameter-efficient training
- **Flexible Data Processing**: Support for JSON, CSV, TXT, and JSONL formats
- **Conversation Data**: Specialized processors for dialogue, Q&A, and instruction formats
- **Interactive Setup**: User-friendly command-line interface
- **Comprehensive Monitoring**: Rich console output with progress tracking
- **Automatic Validation**: Built-in data validation and model evaluation
- **Checkpoint Management**: Save and resume training from checkpoints

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/llm-fine-tuner.git
cd llm-fine-tuner
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "from src.agent import LLMFineTunerAgent; print('✅ Installation successful!')"
```

## 🎯 Quick Start

### Interactive Mode

Run the agent interactively for guided setup:

```bash
python main.py
```

The agent will guide you through:
- Model selection (presets or custom)
- Data file configuration
- Training parameters
- LoRA/QLoRA setup

### Programmatic Usage

```python
from src.agent import LLMFineTunerAgent
from src.config import DEFAULT_CONFIGS

# Use GPT-2 preset
config = DEFAULT_CONFIGS["gpt2"]
config.training.num_epochs = 1
config.data.train_file = "your_data.json"

# Initialize and train
agent = LLMFineTunerAgent(config)
results = agent.start_training()
```

### Example Scripts

Run the included examples:

```bash
# Quick start with GPT-2
python examples/quick_start.py

# Custom model configuration
python examples/custom_model_example.py
```

## 📊 Supported Models

### Preset Configurations

| Model | Type | Base Model | LoRA | QLoRA |
|-------|------|------------|------|-------|
| GPT-2 | Causal | `gpt2` | ✅ | ❌ |
| LLaMA | Causal | `meta-llama/Llama-2-7b-hf` | ✅ | ✅ |
| T5 | Seq2Seq | `t5-base` | ❌ | ❌ |
| DialoGPT | Causal | `microsoft/DialoGPT-medium` | ✅ | ❌ |

### Custom Models

You can use any Hugging Face model by specifying the model name and type:

```python
from src.config import FineTunerConfig

config = FineTunerConfig()
config.model.model_name = "your-model-name"
config.model.model_type = "causal"  # or "seq2seq"
```

## 📁 Data Formats

### Supported Formats

- **JSON**: List of objects or object with arrays
- **CSV**: Comma-separated values
- **TXT**: One example per line
- **JSONL**: One JSON object per line

### Data Structure Examples

#### Conversation Format
```json
[
  {
    "conversation": "Human: What is AI?\nAssistant: AI is...",
    "category": "technology"
  }
]
```

#### Q&A Format
```json
[
  {
    "question": "What is machine learning?",
    "answer": "Machine learning is..."
  }
]
```

#### Instruction Format
```json
[
  {
    "instruction": "Explain quantum computing",
    "response": "Quantum computing is..."
  }
]
```

#### Simple Text
```json
[
  {
    "text": "Your training text here..."
  }
]
```

## ⚙️ Configuration

### Training Parameters

```python
config.training.num_epochs = 3
config.training.batch_size = 4
config.training.learning_rate = 2e-5
config.training.gradient_accumulation_steps = 4
```

### LoRA Configuration

```python
config.training.use_lora = True
config.training.lora_r = 16          # Rank
config.training.lora_alpha = 32      # Alpha
config.training.lora_dropout = 0.1   # Dropout
config.training.target_modules = ["q_proj", "v_proj"]
```

### QLoRA Configuration

```python
config.training.use_qlora = True
config.training.bf16 = True          # Required for QLoRA
config.training.batch_size = 1       # Smaller batch size for quantization
```

## 🔧 Advanced Usage

### Custom Data Processor

```python
from src.data_processor import ConversationDataProcessor

processor = ConversationDataProcessor(config)
processor.set_conversation_format("dialogue")  # or "qa", "instruction"
```

### Model Evaluation

```python
# Evaluate trained model
eval_results = agent.evaluate_model()

# Generate text
generated = agent.generate_text("Human: What is deep learning?")
```

### Checkpoint Management

```python
# Save checkpoint
checkpoint_path = agent.trainer.save_checkpoint("./checkpoints")

# Resume training
agent.trainer.resume_training(checkpoint_path)
```

## 📈 Monitoring and Logging

### Console Output

The agent provides rich console output with:
- Training progress bars
- Loss curves
- Model information tables
- Status updates

### Logging

Logs are saved to:
- Console output (rich formatting)
- File logs in `output_dir/logs/`
- TensorBoard (if enabled)

### Metrics

Tracked metrics include:
- Training loss
- Validation loss
- Learning rate
- Gradient norms
- Training time

## 🧪 Testing

Run the test suite:

```bash
python -m unittest discover tests/
```

Or run specific tests:

```bash
python tests/test_config.py
```

## 📁 Project Structure

```
llm-fine-tuner/
├── src/                    # Source code
│   ├── __init__.py
│   ├── agent.py           # Main agent class
│   ├── config.py          # Configuration management
│   ├── data_processor.py  # Data loading and processing
│   ├── model_manager.py   # Model loading and PEFT setup
│   └── trainer.py         # Training orchestration
├── examples/              # Example scripts
│   ├── quick_start.py
│   └── custom_model_example.py
├── tests/                 # Unit tests
├── main.py               # Main entry point
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable QLoRA for large models

2. **Training Loss Not Decreasing**
   - Check learning rate
   - Verify data format
   - Increase training epochs

3. **Model Loading Errors**
   - Check model name spelling
   - Verify internet connection
   - Check model compatibility

### Getting Help

- Check the logs in `output_dir/logs/`
- Verify your data format matches examples
- Ensure all dependencies are installed
- Check GPU memory availability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers team
- Microsoft LoRA paper authors
- QLoRA research team
- Rich library for beautiful console output

## 📚 References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)

---

**Happy Fine-tuning! 🎉**

## 🚀 **Advanced Usage**

### **Enhanced Agent with LangGraph Integration** ⭐ **NEW!**

The system now includes an **Enhanced Agent** that integrates with **LangGraph** for intelligent workflow orchestration:

```python
from src.enhanced_agent import EnhancedLLMFineTunerAgent

# Initialize enhanced agent with workflow orchestration
agent = EnhancedLLMFineTunerAgent()

# Start intelligent workflow
result = agent.start_workflow(
    user_requirements="I want to quickly fine-tune a model for conversation data"
)

# The agent automatically:
# 1. Analyzes your requirements
# 2. Detects hardware capabilities
# 3. Optimizes configuration
# 4. Executes training
# 5. Monitors progress
# 6. Evaluates results
```

**🚀 Quick Start:**
```bash
# Install enhanced dependencies
pip install langgraph langchain langchain-openai

# Run the demo
python demo_enhanced_agent.py

# Read quick start guide
# ENHANCED_AGENT_QUICKSTART.md
```

**Key Benefits:**
- 🎯 **Structured Workflow**: Predefined execution steps with state management
- 📊 **State Tracking**: Monitor progress through each workflow step
- 🔄 **Conditional Execution**: Smart decision-making based on conditions
- 📈 **Progress Monitoring**: Real-time status updates and intervention
- 🧠 **Intelligent Optimization**: Automatic parameter tuning based on requirements
- 🛡️ **Error Handling**: Graceful fallback and recovery mechanisms

**Workflow Steps:**
1. **Requirements Analysis** → Understand user needs
2. **Hardware Detection** → Assess system capabilities
3. **Configuration Optimization** → Auto-tune parameters
4. **Training Execution** → Run fine-tuning process
5. **Progress Monitoring** → Track and intervene as needed
6. **Results Evaluation** → Assess outcomes and provide insights

**Installation for Enhanced Features:**
```bash
pip install langgraph langchain langchain-openai
```

**Example Usage:**
```python
# See examples/enhanced_agent_example.py for complete examples
python examples/enhanced_agent_example.py
```

**Documentation:**
- **Quick Start**: `ENHANCED_AGENT_QUICKSTART.md`
- **Full Guide**: `LANGGRAPH_INTEGRATION.md`
- **Demo Script**: `demo_enhanced_agent.py`
- **Installation**: `install_enhanced_agent.py`

### **LangGraph Integration (Enhanced Agent)**

The system includes an enhanced agent that integrates with **LangGraph** for advanced workflow orchestration:

```python
from src.enhanced_agent import EnhancedLLMFineTunerAgent

# Initialize enhanced agent with workflow orchestration
agent = EnhancedLLMFineTunerAgent()

# Start intelligent workflow
result = agent.start_workflow(
    user_requirements="I want to quickly fine-tune a model for conversation data"
)

# The agent automatically:
# 1. Analyzes your requirements
# 2. Detects hardware capabilities
# 3. Optimizes configuration
# 4. Executes training
# 5. Monitors progress
# 6. Evaluates results
```

**Key Benefits of LangGraph Integration:**
- 🎯 **Structured Workflow**: Predefined execution steps with state management
- 📊 **State Tracking**: Monitor progress through each workflow step
- 🔄 **Conditional Execution**: Smart decision-making based on conditions
- 📈 **Progress Monitoring**: Real-time status updates and intervention
- 🧠 **Intelligent Optimization**: Automatic parameter tuning based on requirements
- 🛡️ **Error Handling**: Graceful fallback and recovery mechanisms

**Workflow Steps:**
1. **Requirements Analysis** → Understand user needs
2. **Hardware Detection** → Assess system capabilities
3. **Configuration Optimization** → Auto-tune parameters
4. **Training Execution** → Run fine-tuning process
5. **Progress Monitoring** → Track and intervene as needed
6. **Results Evaluation** → Assess outcomes and provide insights

**Installation for Enhanced Features:**
```bash
pip install langgraph langchain langchain-openai
```

**Example Usage:**
```python
# See examples/enhanced_agent_example.py for complete examples
python examples/enhanced_agent_example.py
```

### **Custom Model Configurations**
