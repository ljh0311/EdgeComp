#!/usr/bin/env python3
"""
Baby Monitor System - Main Entry Point

This script serves as the main entry point for the Baby Monitor System, supporting three launch modes:
1. Normal Mode: Default for standard users, showing the main page with camera feed and metrics, but no access to dev tools.
2. Dev Mode: Displays metrics page with access to all development tools.
3. Local Mode: Shows the local GUI version of the baby monitor.

Usage:
    python main.py --mode [normal|dev|local] [options]

Options:
    --mode MODE             Launch mode (normal, dev, local) [default: normal]
    --threshold THRESHOLD   Detection threshold [default: 0.5]
    --camera_id CAMERA_ID   Camera ID [default: 0]
    --input_device INPUT    Audio input device ID [default: None]
    --host HOST             Host for web interface [default: 0.0.0.0]
    --port PORT             Port for web interface [default: 5000]
    --debug                 Enable debug mode
"""

import os
import sys
import time
import signal
import logging
import argparse
from threading import Event

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import Baby Monitor components
from babymonitor.camera import Camera
from babymonitor.audio import AudioProcessor
from babymonitor.detectors.person_detector import PersonDetector
from babymonitor.detectors.emotion_detector import EmotionDetector
from babymonitor.web.server import BabyMonitorWeb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'baby_monitor.log'))
    ]
)
logger = logging.getLogger('baby_monitor')

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Signal handler for graceful shutdown
stop_event = Event()

def signal_handler(sig, frame):
    """Handle signals for graceful shutdown."""
    logger.info("Shutdown signal received. Stopping Baby Monitor System...")
    stop_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_normal_mode(args):
    """
    Start the Baby Monitor in normal mode.
    
    This mode is intended for standard users and provides access to the main
    dashboard and metrics, but not to development tools.
    """
    logger.info("Starting Baby Monitor System in NORMAL mode")
    
    try:
        # Initialize camera
        logger.info("Initializing camera...")
        camera = Camera(camera_id=args.camera_id)
        
        # Initialize audio processor
        logger.info("Initializing audio processor...")
        audio_processor = AudioProcessor(device=args.input_device)
        
        # Initialize person detector
        logger.info("Initializing person detector...")
        person_detector = PersonDetector(
            detection_threshold=args.threshold,
            camera=camera
        )
        
        # Initialize emotion detector
        logger.info("Initializing emotion detector...")
        emotion_detector = EmotionDetector(
            audio_processor=audio_processor
        )
        
        # Start web interface
        logger.info("Starting web interface...")
        web_interface = BabyMonitorWeb(
            camera=camera,
            person_detector=person_detector,
            emotion_detector=emotion_detector,
            host=args.host,
            port=args.port,
            mode="normal",
            debug=args.debug
        )
        web_interface.start()
        
        # Keep main thread alive until stop event is set
        while not stop_event.is_set():
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Error in normal mode: {e}")
        raise
    finally:
        logger.info("Shutting down Baby Monitor System...")
        if 'web_interface' in locals():
            web_interface.stop()
        if 'person_detector' in locals():
            person_detector.stop()
        if 'emotion_detector' in locals():
            emotion_detector.stop()
        if 'camera' in locals():
            camera.release()
        if 'audio_processor' in locals():
            audio_processor.stop()

def run_dev_mode(args):
    """
    Start the Baby Monitor in developer mode.
    
    This mode provides access to all development tools and metrics,
    and is intended for developers and testers.
    """
    logger.info("Starting Baby Monitor System in DEVELOPER mode")
    
    # Set logging level to DEBUG in dev mode
    logger.setLevel(logging.DEBUG)
    
    try:
        # Initialize camera
        logger.info("Initializing camera...")
        camera = Camera(camera_id=args.camera_id)
        
        # Initialize audio processor
        logger.info("Initializing audio processor...")
        audio_processor = AudioProcessor(device=args.input_device)
        
        # Initialize person detector
        logger.info("Initializing person detector...")
        person_detector = PersonDetector(
            detection_threshold=args.threshold,
            camera=camera
        )
        
        # Initialize emotion detector
        logger.info("Initializing emotion detector...")
        emotion_detector = EmotionDetector(
            audio_processor=audio_processor
        )
        
        # Start web interface in dev mode
        logger.info("Starting web interface in developer mode...")
        web_interface = BabyMonitorWeb(
            camera=camera,
            person_detector=person_detector,
            emotion_detector=emotion_detector,
            host=args.host,
            port=args.port,
            mode="dev",
            debug=True  # Always enable debug in dev mode
        )
        web_interface.start()
        
        # Keep main thread alive until stop event is set
        while not stop_event.is_set():
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Error in developer mode: {e}")
        raise
    finally:
        logger.info("Shutting down Baby Monitor System...")
        if 'web_interface' in locals():
            web_interface.stop()
        if 'person_detector' in locals():
            person_detector.stop()
        if 'emotion_detector' in locals():
            emotion_detector.stop()
        if 'camera' in locals():
            camera.release()
        if 'audio_processor' in locals():
            audio_processor.stop()

def run_local_mode(args):
    """
    Start the Baby Monitor in local GUI mode.
    
    This mode runs the local GUI version of the baby monitor using PyQt5.
    """
    logger.info("Starting Baby Monitor System in LOCAL mode")
    
    try:
        # Import GUI components
        try:
            from PyQt5.QtWidgets import QApplication
            from sound_emotion_gui import EmotionDetectorGUI
        except ImportError as e:
            logger.error(f"Failed to import GUI components: {e}")
            logger.error("Make sure PyQt5 is installed: pip install PyQt5")
            return
        
        # Create QApplication
        app = QApplication(sys.argv)
        
        # Create and show the GUI
        logger.info("Starting local GUI...")
        gui = EmotionDetectorGUI()
        gui.show()
        
        # Start the application event loop
        logger.info("Local GUI started. Close the window to exit.")
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Error in local mode: {e}")
        raise

def main():
    """Parse command line arguments and start the appropriate mode."""
    parser = argparse.ArgumentParser(description='Baby Monitor System')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'dev', 'local'],
                        help='Launch mode (normal, dev, local)')
    
    # Common options
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera ID')
    parser.add_argument('--input_device', type=int, default=None,
                        help='Audio input device ID')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host for web interface')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for web interface')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Start the appropriate mode
    if args.mode == 'normal':
        run_normal_mode(args)
    elif args.mode == 'dev':
        run_dev_mode(args)
    elif args.mode == 'local':
        run_local_mode(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == '__main__':
    main()
