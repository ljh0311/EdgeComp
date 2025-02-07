#!/usr/bin/env python3
"""
Baby Monitor System - Launcher
===========================
Main entry point for launching the Baby Monitor System.
"""

import os
import sys

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# Import and run the application
from main import main

if __name__ == "__main__":
    main() 