#!/usr/bin/env python3
"""
Baby Monitor Example
------------------
Example script demonstrating how to use the baby monitor system with the lightweight detector.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the src directory to the Python path
src_dir = str(Path(__file__).resolve().parent.parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import baby monitor components
from babymonitor.config import Config
from babymonitor.web.app import BabyMonitorWeb
from babymonitor.detectors.detector_factory import DetectorFactory, DetectorType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Baby Monitor Example')
    parser.add_argument('--detector', type=str, default='lightweight',
                        choices=['lightweight', 'yolov8'],
                        help='Detector type to use (default: lightweight)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to listen on (default: 5000)')
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of threads for lightweight detector (default: 4)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--resolution', type=str, default='640x480',
                        help='Camera resolution in WxH format (default: 640x480)')
    return parser.parse_args()

def main():
    """Main entry point for the example."""
    args = parse_args()
    
    try:
        # Parse resolution
        width, height = map(int, args.resolution.split('x'))
        
        # Update configuration based on command line arguments
        Config.DETECTOR_TYPE = args.detector
        Config.WEB_HOST = args.host
        Config.WEB_PORT = args.port
        Config.LIGHTWEIGHT_DETECTION['num_threads'] = args.threads
        Config.LIGHTWEIGHT_DETECTION['camera_index'] = args.camera
        Config.LIGHTWEIGHT_DETECTION['resolution'] = (width, height)
        
        # Initialize components
        logger.info("Initializing Baby Monitor System...")
        
        # Initialize web interface
        web_interface = BabyMonitorWeb(
            host=Config.WEB_HOST,
            port=Config.WEB_PORT
        )
        
        # Start web interface
        logger.info(f"Starting web interface on http://{Config.WEB_HOST}:{Config.WEB_PORT}")
        logger.info("Press Ctrl+C to stop")
        web_interface.start()
        
    except KeyboardInterrupt:
        logger.info("Stopping Baby Monitor System...")
        if 'web_interface' in locals():
            web_interface.stop()
        
    except Exception as e:
        logger.error(f"Error in main program: {e}")
        raise

if __name__ == "__main__":
    main() 