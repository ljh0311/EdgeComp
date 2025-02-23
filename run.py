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
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# Import and run the web application
from web_app import app, socketio, monitor

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\nShutting down Baby Monitor System...")
    monitor.stop()
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("Starting Baby Monitor System...")
        print("Access the web interface at http://localhost:5000")
        print("Press Ctrl+C to stop")
        socketio.run(app, debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nShutting down Baby Monitor System...")
        monitor.stop()
        sys.exit(0) 