import sys
import os
import time
import json
import argparse
import requests
import threading
import logging
from datetime import datetime
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('BabyClient')

# Prevent duplicate logs
logging.getLogger('socketio').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

def handle_video_error(self, error_message):
    """Show video error in the video label"""
    logger.error(f"Video error: {error_message}")
    self.video_label.setText(error_message)

def process_video_frame(self, frame_data):
    try:
        self.video_thread.process_frame(frame_data)
        if not self.video_thread.connected:
            logger.info("Camera connection established")
    except Exception as e:
        logger.error(f"Error processing video frame: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Baby Monitor Client")
    parser.add_argument("--host", default="192.168.1.100", help="Host IP address")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--mqtt-host", help="MQTT broker host (defaults to --host)")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--resolution", default="640x480", help="Video resolution (e.g., 640x480, 320x240)")
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set video resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        os.environ['VIDEO_WIDTH'] = str(width)
        os.environ['VIDEO_HEIGHT'] = str(height)
    except ValueError:
        logger.warning(f"Invalid resolution format: {args.resolution}. Using default 640x480")
        os.environ['VIDEO_WIDTH'] = '640'
        os.environ['VIDEO_HEIGHT'] = '480'

    # ... existing code ... 