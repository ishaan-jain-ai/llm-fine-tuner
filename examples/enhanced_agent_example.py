#!/usr/bin/env python3
"""
Enhanced Agent Example with LangGraph Integration
Demonstrates the workflow orchestration capabilities
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enhanced_agent import EnhancedLLMFineTunerAgent
from config import FineTunerConfig

def enhanced_agent_example():
    """Demonstrate the enhanced agent capabilities"""
    
    print("🚀 Enhanced LLM Fine Tuner Agent Example")
    print("=" * 60)
    
    # Initialize the enhanced agent
    print("\n1️⃣ Initializing Enhanced Agent...")
    agent = EnhancedLLMFineTunerAgent()
    
    # Show workflow visualization
    print("\n2️⃣ Workflow Structure:")
    agent.show_workflow_visualization()
    
    # Example 1: Quick fine-tuning workflow
    print("\n3️⃣ Example 1: Quick Fine-tuning Workflow")
    print("-" * 40)
    
    requirements = "I want to quickly fine-tune a model for conversation data"
    result = agent.start_workflow(requirements)
    
    if result["success"]:
        print("✅ Workflow completed successfully!")
        print(f"📊 Steps completed: {len(result['state']['step_history'])}")
        print(f"💡 Suggestions: {len(result['state']['suggestions'])}")
        
        if result["config"]:
            print(f"⚙️ Optimized batch size: {result['config'].training.batch_size}")
            print(f"⚙️ Optimized LoRA rank: {result['config'].training.lora_r}")
            print(f"⚙️ Using QLoRA: {result['config'].training.use_qlora}")
    else:
        print(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
    
    # Show current status
    print("\n4️⃣ Current Agent Status:")
    agent.show_status()
    
    # Example 2: Quality-focused workflow
    print("\n5️⃣ Example 2: Quality-Focused Workflow")
    print("-" * 40)
    
    requirements = "I need high-quality fine-tuning for a large model, speed is not important"
    result = agent.start_workflow(requirements)
    
    if result["success"]:
        print("✅ Quality workflow completed!")
        if result["config"]:
            print(f"⚙️ Quality-focused epochs: {result['config'].training.num_epochs}")
            print(f"⚙️ Quality-focused LoRA rank: {result['config'].training.lora_r}")
    
    # Example 3: Hardware-constrained workflow
    print("\n6️⃣ Example 3: Hardware-Constrained Workflow")
    print("-" * 40)
    
    requirements = "I have limited GPU memory, need memory-efficient fine-tuning"
    result = agent.start_workflow(requirements)
    
    if result["success"]:
        print("✅ Memory-efficient workflow completed!")
        if result["config"]:
            print(f"⚙️ Memory-efficient batch size: {result['config'].training.batch_size}")
            print(f"⚙️ Using QLoRA: {result['config'].training.use_qlora}")
            print(f"⚙️ Gradient accumulation: {result['config'].training.gradient_accumulation_steps}")
    
    # Final status
    print("\n7️⃣ Final Agent Status:")
    agent.show_status()
    
    # Cleanup
    print("\n8️⃣ Cleaning up...")
    agent.cleanup()
    
    print("\n🎉 Enhanced Agent Example Completed!")
    print("\nKey Benefits of LangGraph Integration:")
    print("• 🎯 Structured workflow execution")
    print("• 📊 State management and tracking")
    print("• 🔄 Conditional execution paths")
    print("• 📈 Progress monitoring")
    print("• 🧠 Intelligent decision making")
    print("• 🛡️ Error handling and recovery")

def compare_agents():
    """Compare the original agent with the enhanced agent"""
    
    print("\n🔄 Agent Comparison")
    print("=" * 40)
    
    # Import original agent
    from agent import LLMFineTunerAgent
    
    print("\n📊 Original Agent:")
    original_agent = LLMFineTunerAgent()
    print("• Basic agentic behavior")
    print("• Manual configuration")
    print("• Simple execution flow")
    
    print("\n📊 Enhanced Agent:")
    enhanced_agent = EnhancedLLMFineTunerAgent()
    print("• LangGraph workflow orchestration")
    print("• State management")
    print("• Conditional execution")
    print("• Progress tracking")
    print("• Intelligent optimization")
    
    print("\n💡 The enhanced agent adds workflow orchestration")
    print("   while maintaining compatibility with the original interface!")

if __name__ == "__main__":
    try:
        enhanced_agent_example()
        compare_agents()
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install langgraph langchain langchain-openai")
