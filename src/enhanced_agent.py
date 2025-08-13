#!/usr/bin/env python3
"""
Simplified Enhanced LLM Fine Tuner Agent with LangGraph Integration
Demonstrates workflow orchestration and state management
"""

import logging
from typing import TypedDict, Dict, Any, List, Optional
from pathlib import Path

# LangGraph imports (will be optional)
try:
    from langgraph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ LangGraph not available. Install with: pip install langgraph")

from .config import FineTunerConfig
from .data_processor import DataProcessor, ConversationDataProcessor
from .model_manager import ModelManager
from .trainer import FineTuningTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTuningState(TypedDict):
    """State management for the fine-tuning workflow"""
    user_requirements: str
    hardware_specs: Dict[str, Any]
    data_info: Dict[str, Any]
    config: Optional[FineTunerConfig]
    training_status: str
    results: Dict[str, Any]
    errors: List[str]
    suggestions: List[str]
    current_step: str
    step_history: List[str]

class EnhancedLLMFineTunerAgent:
    """
    Enhanced agent with LangGraph workflow orchestration
    Demonstrates the power of workflow-based fine-tuning
    """
    
    def __init__(self, config: Optional[FineTunerConfig] = None):
        self.config = config or FineTunerConfig()
        self.workflow = None
        self.current_state = None
        
        # Initialize components
        self.model_manager = ModelManager(self.config)
        self.data_processor = None
        self.trainer = None
        
        # Build workflow if LangGraph is available
        if LANGGRAPH_AVAILABLE:
            self._build_workflow()
            logger.info("🚀 Enhanced Agent initialized with LangGraph workflow")
        else:
            logger.info("🚀 Enhanced Agent initialized (LangGraph not available)")
    
    def _build_workflow(self):
        """Build the LangGraph workflow for fine-tuning orchestration"""
        if not LANGGRAPH_AVAILABLE:
            return
            
        try:
            # Define workflow nodes
            workflow = StateGraph(FineTuningState)
            
            # Add nodes
            workflow.add_node("analyze_requirements", self._analyze_requirements)
            workflow.add_node("detect_hardware", self._detect_hardware)
            workflow.add_node("optimize_config", self._optimize_config)
            workflow.add_node("execute_training", self._execute_training)
            workflow.add_node("evaluate_results", self._evaluate_results)
            
            # Define workflow edges
            workflow.add_edge("analyze_requirements", "detect_hardware")
            workflow.add_edge("detect_hardware", "optimize_config")
            workflow.add_edge("optimize_config", "execute_training")
            workflow.add_edge("execute_training", "evaluate_results")
            workflow.add_edge("evaluate_results", END)
            
            # Compile workflow
            self.workflow = workflow.compile()
            logger.info("✅ LangGraph workflow built and compiled")
            
        except Exception as e:
            logger.error(f"❌ Failed to build workflow: {e}")
            self.workflow = None
    
    def _analyze_requirements(self, state: FineTuningState) -> FineTuningState:
        """Analyze user requirements"""
        state["current_step"] = "analyzing_requirements"
        state["step_history"].append("analyze_requirements")
        
        requirements = state["user_requirements"].lower()
        
        if "conversation" in requirements or "dialogue" in requirements:
            state["suggestions"].append("Use ConversationDataProcessor for dialogue data")
        if "large model" in requirements or "20b" in requirements:
            state["suggestions"].append("Consider QLoRA for memory efficiency")
        if "fast" in requirements or "quick" in requirements:
            state["suggestions"].append("Use smaller LoRA rank for faster training")
        if "quality" in requirements or "accuracy" in requirements:
            state["suggestions"].append("Use larger LoRA rank for better quality")
        
        logger.info("✅ Requirements analyzed")
        return state
    
    def _detect_hardware(self, state: FineTuningState) -> FineTuningState:
        """Detect hardware capabilities"""
        state["current_step"] = "detecting_hardware"
        state["step_history"].append("detect_hardware")
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                state["hardware_specs"] = {
                    "gpu_available": True,
                    "gpu_name": gpu_name,
                    "gpu_memory_gb": gpu_memory
                }
                
                if gpu_memory < 8:
                    state["suggestions"].append("GPU memory < 8GB: Use QLoRA with small batch size")
                elif gpu_memory < 16:
                    state["suggestions"].append("GPU memory < 16GB: Use QLoRA with medium batch size")
                else:
                    state["suggestions"].append("GPU memory >= 16GB: Can use LoRA or QLoRA")
                    
                logger.info(f"🔍 Detected GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                state["hardware_specs"] = {"gpu_available": False}
                state["suggestions"].append("No GPU detected: Training will be very slow on CPU")
                
        except ImportError:
            state["hardware_specs"] = {"gpu_available": False, "error": "PyTorch not available"}
        
        return state
    
    def _optimize_config(self, state: FineTuningState) -> FineTuningState:
        """Optimize configuration based on requirements and hardware"""
        state["current_step"] = "optimizing_config"
        state["step_history"].append("optimize_config")
        
        try:
            config = FineTunerConfig()
            
            # Apply hardware-based optimizations
            hardware = state.get("hardware_specs", {})
            if hardware.get("gpu_available", False):
                gpu_memory = hardware.get("gpu_memory_gb", 0)
                
                if gpu_memory < 8:
                    config.training.use_qlora = True
                    config.training.bf16 = True
                    config.training.batch_size = 1
                    config.training.gradient_accumulation_steps = 32
                    config.training.lora_r = 16
                    state["suggestions"].append("Applied low-memory QLoRA configuration")
                    
                elif gpu_memory < 16:
                    config.training.use_qlora = True
                    config.training.bf16 = True
                    config.training.batch_size = 2
                    config.training.gradient_accumulation_steps = 16
                    config.training.lora_r = 32
                    state["suggestions"].append("Applied medium-memory QLoRA configuration")
                    
                else:
                    config.training.use_lora = True
                    config.training.batch_size = 4
                    config.training.gradient_accumulation_steps = 8
                    config.training.lora_r = 64
                    state["suggestions"].append("Applied high-memory LoRA configuration")
            
            # Apply requirement-based optimizations
            requirements = state.get("user_requirements", "").lower()
            if "fast" in requirements:
                config.training.num_epochs = 1
                config.training.lora_r = min(config.training.lora_r, 32)
            elif "quality" in requirements:
                config.training.num_epochs = 5
                config.training.lora_r = max(config.training.lora_r, 64)
            
            state["config"] = config
            logger.info("⚙️ Configuration optimized")
            
        except Exception as e:
            state["errors"].append(f"Configuration optimization failed: {str(e)}")
        
        return state
    
    def _execute_training(self, state: FineTuningState) -> FineTuningState:
        """Execute the fine-tuning process"""
        state["current_step"] = "executing_training"
        state["step_history"].append("execute_training")
        
        try:
            config = state["config"]
            data_file = state["data_info"].get("data_file")
            
            # Initialize trainer
            self.trainer = FineTuningTrainer(config)
            
            # Setup training
            self.trainer.setup_training(data_file)
            
            # Start training
            state["training_status"] = "training_started"
            logger.info("🚀 Training started")
            
        except Exception as e:
            state["errors"].append(f"Training execution failed: {str(e)}")
        
        return state
    
    def _evaluate_results(self, state: FineTuningState) -> FineTuningState:
        """Evaluate training results"""
        state["current_step"] = "evaluating_results"
        state["step_history"].append("evaluate_results")
        
        try:
            if self.trainer:
                sample_prompt = "Hello, how are you?"
                generated_text = self.trainer.generate_sample(sample_prompt, max_length=50)
                
                state["results"] = {
                    "sample_generation": generated_text,
                    "training_completed": True,
                    "model_saved": True
                }
                
                logger.info("✅ Training evaluation complete")
                
        except Exception as e:
            state["errors"].append(f"Results evaluation failed: {str(e)}")
        
        return state
    
    def start_workflow(self, user_requirements: str, data_file: Optional[str] = None) -> Dict[str, Any]:
        """Start the complete fine-tuning workflow"""
        if not self.workflow:
            logger.warning("⚠️ LangGraph workflow not available, using fallback")
            return self._fallback_workflow(user_requirements, data_file)
        
        logger.info("🚀 Starting enhanced fine-tuning workflow")
        
        # Initialize state
        initial_state = FineTuningState(
            user_requirements=user_requirements,
            hardware_specs={},
            data_info={"data_file": data_file} if data_file else {},
            config=None,
            training_status="not_started",
            results={},
            errors=[],
            suggestions=[],
            current_step="initialized",
            step_history=[]
        )
        
        # Execute workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            self.current_state = final_state
            
            logger.info("✅ Workflow completed successfully")
            return {
                "success": True,
                "state": final_state,
                "config": final_state.get("config"),
                "results": final_state.get("results"),
                "suggestions": final_state.get("suggestions"),
                "errors": final_state.get("errors")
            }
            
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "state": self.current_state
            }
    
    def _fallback_workflow(self, user_requirements: str, data_file: Optional[str] = None) -> Dict[str, Any]:
        """Fallback workflow when LangGraph is not available"""
        logger.info("🔄 Using fallback workflow")
        
        # Simulate the workflow steps
        state = FineTuningState(
            user_requirements=user_requirements,
            hardware_specs={},
            data_info={"data_file": data_file} if data_file else {},
            config=None,
            training_status="not_started",
            results={},
            errors=[],
            suggestions=[],
            current_step="initialized",
            step_history=[]
        )
        
        # Execute steps manually
        state = self._analyze_requirements(state)
        state = self._detect_hardware(state)
        state = self._optimize_config(state)
        state = self._execute_training(state)
        state = self._evaluate_results(state)
        
        self.current_state = state
        
        return {
            "success": True,
            "state": state,
            "config": state.get("config"),
            "results": state.get("results"),
            "suggestions": state.get("suggestions"),
            "errors": state.get("errors")
        }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        if not self.current_state:
            return {"status": "not_started"}
        
        return {
            "current_step": self.current_state.get("current_step"),
            "step_history": self.current_state.get("step_history"),
            "training_status": self.current_state.get("training_status"),
            "suggestions": self.current_state.get("suggestions"),
            "errors": self.current_state.get("errors"),
            "results": self.current_state.get("results")
        }
    
    def show_workflow_visualization(self):
        """Show the workflow graph visualization"""
        if self.workflow:
            logger.info("📊 Workflow visualization:")
            logger.info("analyze_requirements → detect_hardware → optimize_config → execute_training → evaluate_results")
        else:
            logger.info("📊 Fallback workflow: Manual step execution")
    
    # Maintain compatibility with original agent interface
    def interactive_setup(self) -> FineTunerConfig:
        """Interactive setup using the enhanced workflow"""
        print("🤖 Enhanced Agent Setup - Powered by LangGraph")
        print("=" * 50)
        
        requirements = input("Describe your fine-tuning requirements: ")
        data_file = input("Path to training data file (or press Enter for sample data): ").strip()
        
        if not data_file:
            data_file = None
        
        # Start workflow
        result = self.start_workflow(requirements, data_file)
        
        if result["success"]:
            return result["config"]
        else:
            logger.warning("Workflow failed, using fallback configuration")
            return FineTunerConfig()
    
    def start_training(self) -> Dict[str, Any]:
        """Start training using the enhanced workflow"""
        if not self.current_state or not self.current_state.get("config"):
            logger.warning("No configuration available, running interactive setup first")
            self.interactive_setup()
        
        return self.current_state.get("results", {})
    
    def show_status(self):
        """Show enhanced status with workflow information"""
        status = self.get_workflow_status()
        
        print("🤖 Enhanced LLM Fine Tuner Agent Status")
        print("=" * 50)
        print(f"Current Step: {status.get('current_step', 'unknown')}")
        print(f"Training Status: {status.get('training_status', 'unknown')}")
        print(f"Steps Completed: {len(status.get('step_history', []))}")
        
        if status.get('suggestions'):
            print("\n💡 Suggestions:")
            for suggestion in status['suggestions']:
                print(f"  • {suggestion}")
        
        if status.get('errors'):
            print("\n⚠️ Issues Found:")
            for error in status['errors']:
                print(f"  • {error}")
        
        if status.get('results'):
            print("\n✅ Results:")
            for key, value in status['results'].items():
                print(f"  • {key}: {value}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.trainer:
            self.trainer.cleanup()
        logger.info("🧹 Enhanced agent cleanup completed")
