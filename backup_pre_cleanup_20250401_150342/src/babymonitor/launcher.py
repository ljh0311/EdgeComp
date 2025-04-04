#!/usr/bin/env python
"""
Baby Monitor Launcher
====================
Main entry point for the Baby Monitor application.
Parses command-line arguments and launches the appropriate interface.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('babymonitor.log')
    ]
)

logger = logging.getLogger("launcher")

def setup_parser():
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Baby Monitor System - A comprehensive monitoring solution."
    )
    
    # Interface type
    parser.add_argument(
        "--interface", "-i",
        choices=["gui", "web"],
        default="gui",
        help="Interface type: GUI (PyQt) or Web (Flask)"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", "-m",
        choices=["normal", "dev"],
        default="normal",
        help="Operation mode: normal or developer"
    )
    
    # Camera options
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device index to use"
    )
    
    # Audio options
    parser.add_argument(
        "--audio-device", "-a",
        type=int,
        default=None,
        help="Audio device index to use"
    )
    
    # Debug options
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging"
    )
    
    # Model options
    parser.add_argument(
        "--person-model",
        choices=["yolov8n", "yolov8s", "yolov8m"],
        default="yolov8n",
        help="Person detection model to use"
    )
    
    parser.add_argument(
        "--emotion-model",
        default="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        help="Emotion detection model to use"
    )
    
    return parser

def launch_gui(args):
    """Launch the PyQt GUI interface."""
    from babymonitor.gui.main_gui import launch_main_gui
    
    logger.info(f"Launching GUI in {args.mode} mode...")
    
    # Pass command-line args to sys.argv for the GUI to parse
    sys.argv = [sys.argv[0]]
    if args.mode:
        sys.argv.extend(["--mode", args.mode])
    if args.debug:
        sys.argv.append("--debug")
    
    # Launch the GUI
    return launch_main_gui()

def launch_web(args):
    """Launch the web interface."""
    try:
        from babymonitor.web.app import run_web_app
        
        logger.info(f"Launching web interface in {args.mode} mode...")
        
        # Configure the web interface 
        flask_args = {
            "debug": args.debug,
            "mode": args.mode,
            "camera": args.camera
        }
        
        # Launch the web app
        return run_web_app(**flask_args)
    except ImportError:
        logger.error("Web interface not available. Please install required dependencies.")
        return 1

def main():
    """Main entry point for the application."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set up logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Print startup message
    logger.info("Starting Baby Monitor System")
    logger.info(f"Interface: {args.interface}")
    logger.info(f"Mode: {args.mode}")
    
    try:
        # Launch appropriate interface
        if args.interface == "gui":
            return launch_gui(args)
        elif args.interface == "web":
            return launch_web(args)
        else:
            logger.error(f"Unknown interface type: {args.interface}")
            return 1
    except Exception as e:
        logger.exception(f"Error launching application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 