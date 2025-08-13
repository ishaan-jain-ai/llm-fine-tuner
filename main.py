#!/usr/bin/env python3
"""
LLM Fine Tuner Agent - Main Entry Point
"""
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent import LLMFineTunerAgent, main as agent_main

if __name__ == "__main__":
    agent_main()
