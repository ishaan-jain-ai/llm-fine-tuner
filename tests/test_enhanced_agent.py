#!/usr/bin/env python3
"""
Tests for the Enhanced LLM Fine Tuner Agent
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enhanced_agent import EnhancedLLMFineTunerAgent, FineTuningState
from config import FineTunerConfig

class TestEnhancedAgent(unittest.TestCase):
    """Test cases for the enhanced agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = EnhancedLLMFineTunerAgent()
    
    def test_agent_initialization(self):
        """Test that the agent initializes correctly"""
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.config)
        self.assertIsNotNone(self.agent.model_manager)
    
    def test_state_creation(self):
        """Test FineTuningState creation"""
        state = FineTuningState(
            user_requirements="test requirements",
            hardware_specs={},
            data_info={},
            config=None,
            training_status="not_started",
            results={},
            errors=[],
            suggestions=[],
            current_step="initialized",
            step_history=[]
        )
        
        self.assertEqual(state["user_requirements"], "test requirements")
        self.assertEqual(state["current_step"], "initialized")
        self.assertEqual(len(state["step_history"]), 0)
    
    def test_requirements_analysis(self):
        """Test requirements analysis functionality"""
        state = FineTuningState(
            user_requirements="I want to fine-tune for conversation data",
            hardware_specs={},
            data_info={},
            config=None,
            training_status="not_started",
            results={},
            errors=[],
            suggestions=[],
            current_step="initialized",
            step_history=[]
        )
        
        # Test the requirements analysis method
        updated_state = self.agent._analyze_requirements(state)
        
        self.assertEqual(updated_state["current_step"], "analyzing_requirements")
        self.assertIn("analyze_requirements", updated_state["step_history"])
        self.assertGreater(len(updated_state["suggestions"]), 0)
        
        # Check that conversation-related suggestions were added
        conversation_suggestions = [s for s in updated_state["suggestions"] 
                                 if "ConversationDataProcessor" in s]
        self.assertGreater(len(conversation_suggestions), 0)
    
    def test_hardware_detection(self):
        """Test hardware detection functionality"""
        state = FineTuningState(
            user_requirements="test",
            hardware_specs={},
            data_info={},
            config=None,
            training_status="not_started",
            results={},
            errors=[],
            suggestions=[],
            current_step="initialized",
            step_history=[]
        )
        
        # Test the hardware detection method
        updated_state = self.agent._detect_hardware(state)
        
        self.assertEqual(updated_state["current_step"], "detecting_hardware")
        self.assertIn("detect_hardware", updated_state["step_history"])
        self.assertIn("hardware_specs", updated_state)
        
        # Hardware specs should be populated
        hardware = updated_state["hardware_specs"]
        self.assertIsInstance(hardware, dict)
    
    def test_config_optimization(self):
        """Test configuration optimization functionality"""
        state = FineTuningState(
            user_requirements="I need fast fine-tuning",
            hardware_specs={"gpu_available": True, "gpu_memory_gb": 24},
            data_info={},
            config=None,
            training_status="not_started",
            results={},
            errors=[],
            suggestions=[],
            current_step="initialized",
            step_history=[]
        )
        
        # Test the configuration optimization method
        updated_state = self.agent._optimize_config(state)
        
        self.assertEqual(updated_state["current_step"], "optimizing_config")
        self.assertIn("optimize_config", updated_state["step_history"])
        self.assertIsNotNone(updated_state["config"])
        
        # Check that config was optimized for speed
        config = updated_state["config"]
        self.assertEqual(config.training.num_epochs, 1)
        self.assertLessEqual(config.training.lora_r, 32)
    
    def test_workflow_execution(self):
        """Test workflow execution"""
        requirements = "I want to fine-tune for conversation data quickly"
        
        # Test workflow execution
        result = self.agent.start_workflow(requirements)
        
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        
        if result["success"]:
            self.assertIn("state", result)
            self.assertIn("config", result)
            self.assertIn("suggestions", result)
            
            # Check that workflow steps were executed
            state = result["state"]
            self.assertGreater(len(state["step_history"]), 0)
            self.assertIn("analyze_requirements", state["step_history"])
    
    def test_status_tracking(self):
        """Test status tracking functionality"""
        # Run a simple workflow first
        result = self.agent.start_workflow("test requirements")
        
        if result["success"]:
            # Test status retrieval
            status = self.agent.get_workflow_status()
            
            self.assertIsInstance(status, dict)
            self.assertIn("current_step", status)
            self.assertIn("step_history", status)
            self.assertIn("suggestions", status)
    
    def test_workflow_visualization(self):
        """Test workflow visualization"""
        # This should not raise an error
        try:
            self.agent.show_workflow_visualization()
        except Exception as e:
            self.fail(f"Workflow visualization failed: {e}")
    
    def test_cleanup(self):
        """Test cleanup functionality"""
        # This should not raise an error
        try:
            self.agent.cleanup()
        except Exception as e:
            self.fail(f"Cleanup failed: {e}")
    
    def test_fallback_workflow(self):
        """Test fallback workflow when LangGraph is not available"""
        # Force fallback by setting workflow to None
        self.agent.workflow = None
        
        result = self.agent.start_workflow("test requirements")
        
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        
        if result["success"]:
            self.assertIn("state", result)
            self.assertIn("config", result)

if __name__ == "__main__":
    unittest.main()
