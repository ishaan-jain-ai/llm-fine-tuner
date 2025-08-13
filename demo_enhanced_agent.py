#!/usr/bin/env python3
"""
Comprehensive Demo of the Enhanced LLM Fine Tuner Agent
Showcases LangGraph integration, workflow orchestration, and intelligent optimization
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main demo function"""
    
    print("🚀 Enhanced LLM Fine Tuner Agent - Complete Demo")
    print("=" * 70)
    
    try:
        # Import the enhanced agent
        from enhanced_agent import EnhancedLLMFineTunerAgent
        from config import FineTunerConfig
        
        print("✅ Enhanced agent imported successfully")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Make sure you have installed the required dependencies:")
        print("   pip install langgraph langchain langchain-openai")
        return
    
    # Initialize the enhanced agent
    print("\n1️⃣ Initializing Enhanced Agent...")
    agent = EnhancedLLMFineTunerAgent()
    
    # Show workflow visualization
    print("\n2️⃣ Workflow Structure:")
    agent.show_workflow_visualization()
    
    # Demo 1: Quick fine-tuning workflow
    print("\n3️⃣ Demo 1: Quick Fine-tuning Workflow")
    print("-" * 50)
    
    requirements = "I want to quickly fine-tune a model for conversation data"
    print(f"📝 Requirements: {requirements}")
    
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
    
    # Demo 2: Quality-focused workflow
    print("\n5️⃣ Demo 2: Quality-Focused Workflow")
    print("-" * 50)
    
    requirements = "I need high-quality fine-tuning for a large model, speed is not important"
    print(f"📝 Requirements: {requirements}")
    
    result = agent.start_workflow(requirements)
    
    if result["success"]:
        print("✅ Quality workflow completed!")
        if result["config"]:
            print(f"⚙️ Quality-focused epochs: {result['config'].training.num_epochs}")
            print(f"⚙️ Quality-focused LoRA rank: {result['config'].training.lora_r}")
    else:
        print(f"❌ Quality workflow failed: {result.get('error', 'Unknown error')}")
    
    # Demo 3: Hardware-constrained workflow
    print("\n6️⃣ Demo 3: Hardware-Constrained Workflow")
    print("-" * 50)
    
    requirements = "I have limited GPU memory, need memory-efficient fine-tuning"
    print(f"📝 Requirements: {requirements}")
    
    result = agent.start_workflow(requirements)
    
    if result["success"]:
        print("✅ Memory-efficient workflow completed!")
        if result["config"]:
            print(f"⚙️ Memory-efficient batch size: {result['config'].training.batch_size}")
            print(f"⚙️ Using QLoRA: {result['config'].training.use_qlora}")
            print(f"⚙️ Gradient accumulation: {result['config'].training.gradient_accumulation_steps}")
    else:
        print(f"❌ Memory workflow failed: {result.get('error', 'Unknown error')}")
    
    # Demo 4: Large model workflow (GPT-OSS-20B equivalent)
    print("\n7️⃣ Demo 4: Large Model Workflow")
    print("-" * 50)
    
    requirements = "I want to fine-tune a 20B parameter model with optimal memory usage"
    print(f"📝 Requirements: {requirements}")
    
    result = agent.start_workflow(requirements)
    
    if result["success"]:
        print("✅ Large model workflow completed!")
        if result["config"]:
            print(f"⚙️ Large model batch size: {result['config'].training.batch_size}")
            print(f"⚙️ Using QLoRA: {result['config'].training.use_qlora}")
            print(f"⚙️ LoRA rank: {result['config'].training.lora_r}")
            print(f"⚙️ Gradient accumulation: {result['config'].training.gradient_accumulation_steps}")
    else:
        print(f"❌ Large model workflow failed: {result.get('error', 'Unknown error')}")
    
    # Final status
    print("\n8️⃣ Final Agent Status:")
    agent.show_status()
    
    # Demo 5: Workflow monitoring and control
    print("\n9️⃣ Demo 5: Workflow Monitoring & Control")
    print("-" * 50)
    
    # Get detailed workflow status
    status = agent.get_workflow_status()
    
    print(f"📊 Current Step: {status.get('current_step', 'unknown')}")
    print(f"📈 Steps Completed: {len(status.get('step_history', []))}")
    print(f"💡 Total Suggestions: {len(status.get('suggestions', []))}")
    print(f"⚠️ Total Issues: {len(status.get('errors', []))}")
    
    if status.get('step_history'):
        print("\n📋 Step-by-step History:")
        for i, step in enumerate(status['step_history'], 1):
            print(f"  {i}. {step}")
    
    if status.get('suggestions'):
        print("\n💡 Key Suggestions:")
        for suggestion in status['suggestions'][:5]:  # Show first 5
            print(f"  • {suggestion}")
    
    # Demo 6: Fallback workflow demonstration
    print("\n🔟 Demo 6: Fallback Workflow (LangGraph Unavailable)")
    print("-" * 50)
    
    # Temporarily disable LangGraph to show fallback
    original_workflow = agent.workflow
    agent.workflow = None
    
    print("🔄 Testing fallback workflow...")
    result = agent.start_workflow("Test fallback workflow")
    
    if result["success"]:
        print("✅ Fallback workflow works!")
        print(f"📊 Fallback steps completed: {len(result['state']['step_history'])}")
    else:
        print(f"❌ Fallback workflow failed: {result.get('error', 'Unknown error')}")
    
    # Restore original workflow
    agent.workflow = original_workflow
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    agent.cleanup()
    
    # Summary
    print("\n🎉 Enhanced Agent Demo Completed!")
    print("\n" + "=" * 70)
    print("📋 SUMMARY OF CAPABILITIES")
    print("=" * 70)
    
    print("\n🎯 **Workflow Orchestration**")
    print("• 5-step automated workflow execution")
    print("• State management across all steps")
    print("• Conditional execution paths")
    print("• Error handling and recovery")
    
    print("\n🧠 **Intelligent Optimization**")
    print("• Hardware-aware parameter tuning")
    print("• Requirement-based optimization")
    print("• Automatic QLoRA/LoRA selection")
    print("• Memory-efficient configurations")
    
    print("\n📊 **Monitoring & Control**")
    print("• Real-time workflow status")
    print("• Step-by-step progress tracking")
    print("• Intelligent suggestions")
    print("• Error diagnosis and recovery")
    
    print("\n🔄 **Compatibility & Fallback**")
    print("• Maintains original agent interface")
    print("• Graceful fallback when LangGraph unavailable")
    print("• Same configuration and training capabilities")
    print("• Enhanced user experience")
    
    print("\n🚀 **Next Steps**")
    print("• Install dependencies: pip install langgraph langchain langchain-openai")
    print("• Run examples: python examples/enhanced_agent_example.py")
    print("• Run tests: python -m pytest tests/test_enhanced_agent.py")
    print("• Read docs: LANGGRAPH_INTEGRATION.md")
    
    print("\n" + "=" * 70)
    print("🎉 Happy Fine-tuning with Enhanced Workflows! 🎉")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("💡 Check that all dependencies are installed and the system is properly configured")
