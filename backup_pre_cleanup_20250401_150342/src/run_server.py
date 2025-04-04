#!/usr/bin/env python
"""
Run Server Script
===================
This script runs the baby monitor web server with proper initialization.
"""

import os
import sys
import logging
import time
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the baby monitor web server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--mode', default='normal', choices=['normal', 'dev'], 
                      help='Operation mode: normal for families, dev for developers (default: normal)')
    return parser.parse_args()

def main():
    """Run the server."""
    args = parse_args()
    
    # Make sure we're in the correct directory
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Add the current directory to Python path
    sys.path.insert(0, str(script_dir.parent))
    
    # Import the server module
    try:
        from babymonitor.web.app import BabyMonitorWeb
    except ImportError as e:
        logger.error(f"Error importing BabyMonitorWeb: {str(e)}")
        logger.error("Make sure you have the correct Python path set up.")
        return 1
    
    # Initialize and run the server
    try:
        logger.info(f"Starting baby monitor web server on {args.host}:{args.port} in {args.mode} mode")
        server = BabyMonitorWeb(host=args.host, port=args.port, mode=args.mode)
        
        # Log initialization start
        logger.info("Initializing server components...")
        
        # Run the server
        server.start()
        
        return 0
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 