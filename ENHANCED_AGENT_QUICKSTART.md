# 🚀 Enhanced Agent Quick Start Guide

## Overview

This guide helps you get started with the **Enhanced LLM Fine Tuner Agent** that includes LangGraph integration for intelligent workflow orchestration.

## ⚡ Quick Start (5 minutes)

### 1. Install Dependencies

```bash
# Option A: Automatic installation
python install_enhanced_agent.py

# Option B: Manual installation
pip install langgraph langchain langchain-openai
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
python demo_enhanced_agent.py
```

This will showcase all the enhanced agent capabilities!

### 3. Basic Usage

```python
from src.enhanced_agent import EnhancedLLMFineTunerAgent

# Initialize enhanced agent
agent = EnhancedLLMFineTunerAgent()

# Start intelligent workflow
result = agent.start_workflow(
    "I want to quickly fine-tune a model for conversation data"
)

# Check results
if result["success"]:
    print(f"✅ Workflow completed with {len(result['suggestions'])} suggestions")
    print(f"⚙️ Optimized config: {result['config']}")
```

## 🎯 What You Get

### **Before (Original Agent)**
- Manual configuration
- Basic training execution
- Simple error handling

### **After (Enhanced Agent)**
- 🎯 **Intelligent Workflow**: 5-step automated process
- 🧠 **Auto-Optimization**: Hardware-aware parameter tuning
- 📊 **Progress Monitoring**: Real-time status tracking
- 🛡️ **Error Recovery**: Intelligent problem solving
- 🔄 **State Management**: Persistent workflow state

## 🔄 Workflow Steps

The enhanced agent automatically executes:

1. **Requirements Analysis** → Understand your needs
2. **Hardware Detection** → Assess system capabilities
3. **Configuration Optimization** → Auto-tune parameters
4. **Training Execution** → Run fine-tuning process
5. **Results Evaluation** → Assess outcomes

## 💡 Example Use Cases

### **Quick Prototyping**
```python
result = agent.start_workflow(
    "I want to quickly test ideas with minimal training time"
)
```

### **Quality-Focused Training**
```python
result = agent.start_workflow(
    "I need high-quality fine-tuning, speed is not important"
)
```

### **Memory-Constrained Setup**
```python
result = agent.start_workflow(
    "I have limited GPU memory, need memory-efficient fine-tuning"
)
```

### **Large Model Optimization**
```python
result = agent.start_workflow(
    "I want to fine-tune a 20B parameter model optimally"
)
```

## 📊 Monitoring & Control

### **Check Workflow Status**
```python
status = agent.get_workflow_status()
print(f"Current step: {status['current_step']}")
print(f"Steps completed: {len(status['step_history'])}")
```

### **View Suggestions**
```python
for suggestion in status['suggestions']:
    print(f"💡 {suggestion}")
```

### **Workflow Visualization**
```python
agent.show_workflow_visualization()
```

## 🔧 Configuration

### **Basic Setup**
```python
from src.enhanced_agent import EnhancedLLMFineTunerAgent
from src.config import FineTunerConfig

# Use default configuration
agent = EnhancedLLMFineTunerAgent()

# Or customize
config = FineTunerConfig()
config.model.model_name = "gpt2"
config.training.num_epochs = 3

agent = EnhancedLLMFineTunerAgent(config)
```

### **Advanced Configuration**
```python
# Hardware-aware optimization
config.training.use_qlora = True  # For large models
config.training.lora_r = 64       # Quality vs. speed trade-off
config.training.batch_size = 2    # Memory-constrained setup
```

## 🧪 Testing

### **Run Tests**
```bash
# Test the enhanced agent
python -m pytest tests/test_enhanced_agent.py -v

# Test specific functionality
python -m pytest tests/test_enhanced_agent.py::TestEnhancedAgent::test_workflow_execution -v
```

### **Run Examples**
```bash
# Enhanced agent examples
python examples/enhanced_agent_example.py

# Original agent examples (still work!)
python examples/quick_start.py
```

## 🚨 Troubleshooting

### **Common Issues**

1. **LangGraph Not Available**
   ```
   ⚠️ LangGraph not available. Install with: pip install langgraph
   ```
   - **Solution**: Install LangGraph dependencies
   - **Fallback**: Agent automatically uses manual workflow

2. **Import Errors**
   ```
   ❌ Import failed: No module named 'enhanced_agent'
   ```
   - **Solution**: Check that you're in the correct directory
   - **Verify**: Run `python -c "from src.enhanced_agent import EnhancedLLMFineTunerAgent"`

3. **Workflow Execution Fails**
   ```
   ❌ Workflow execution failed: [error details]
   ```
   - **Solution**: Check the error message for specific issues
   - **Fallback**: Agent will use manual execution mode

### **Getting Help**

- **Check Logs**: Look for detailed error messages
- **Verify Dependencies**: Run `pip list | grep langgraph`
- **Test Import**: Try importing in Python console
- **Read Documentation**: Check `LANGGRAPH_INTEGRATION.md`

## 🔄 Fallback Mode

When LangGraph is unavailable, the enhanced agent automatically falls back to manual workflow execution:

```python
# Agent automatically detects LangGraph availability
if agent.workflow:
    print("✅ LangGraph workflow available")
else:
    print("⚠️ Using fallback workflow (LangGraph not available)")
```

**Benefits of Fallback Mode:**
- ✅ **Always Works**: No dependency on external frameworks
- ✅ **Same Interface**: Maintains compatibility
- ✅ **Graceful Degradation**: Falls back to manual execution

## 🚀 Next Steps

### **Immediate Actions**
1. ✅ Install dependencies
2. ✅ Run the demo
3. ✅ Try basic workflow
4. ✅ Explore examples

### **Advanced Usage**
1. 🔧 Customize configurations
2. 📊 Monitor workflow progress
3. 🧪 Run comprehensive tests
4. 📚 Read full documentation

### **Integration**
1. 🔗 Add to existing projects
2. 🎯 Customize workflow nodes
3. 📈 Monitor production usage
4. 🤝 Contribute improvements

## 📚 Additional Resources

- **Full Documentation**: `LANGGRAPH_INTEGRATION.md`
- **API Reference**: `src/enhanced_agent.py`
- **Examples**: `examples/enhanced_agent_example.py`
- **Tests**: `tests/test_enhanced_agent.py`
- **Demo**: `demo_enhanced_agent.py`

## 🎉 Success Indicators

You've successfully set up the enhanced agent when you can:

- ✅ Import `EnhancedLLMFineTunerAgent`
- ✅ Initialize the agent without errors
- ✅ Run `agent.start_workflow("test")` successfully
- ✅ See workflow status and suggestions
- ✅ Run the demo script completely

## 💡 Pro Tips

1. **Start Simple**: Begin with basic workflows before customizing
2. **Monitor Progress**: Use `get_workflow_status()` to track execution
3. **Check Suggestions**: Review agent suggestions for optimization ideas
4. **Test Fallback**: Verify fallback mode works when LangGraph unavailable
5. **Customize Gradually**: Add custom configurations step by step

---

**🎯 Ready to revolutionize your fine-tuning workflow? Start with the demo and explore the possibilities!**
