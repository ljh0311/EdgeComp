#!/usr/bin/env python3
"""
Baby Monitor System - Launcher
===========================
Main entry point for launching the Baby Monitor System.
"""

import os
import sys
import signal
import logging

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

# Import the BabyMonitorSystem
from babymonitor.core.main import BabyMonitorSystem

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\nShutting down Baby Monitor System...")
    if 'monitor' in globals():
        monitor.stop()
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("Starting Baby Monitor System...")
        # Initialize the system in web-only mode
        monitor = BabyMonitorSystem(dev_mode=False, only_local=False, only_web=True)
        
        print("Access the web interface at http://localhost:5000")
        print("Press Ctrl+C to stop")
        
        # Start the monitor system (this will start the web interface)
        monitor.start()
        
    except KeyboardInterrupt:
        print("\nShutting down Baby Monitor System...")
        monitor.stop()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'monitor' in locals():
            monitor.stop()
        sys.exit(1) 