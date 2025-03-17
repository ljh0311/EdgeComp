#!/usr/bin/env python3
"""
Run script for the lightweight detection system.
This script integrates the lightweight detector with the main baby monitor system.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
import threading
import cv2
import numpy as np

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

# Import the lightweight detector
from babymonitor.detectors.lightweight_detector import VideoStream, LightweightDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(src_path, 'baby_monitor.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LightweightDetectionSystem:
    """Lightweight detection system for the baby monitor."""
    
    def __init__(self, model_path, label_path, threshold=0.5, resolution=(640, 480)):
        """Initialize the lightweight detection system.
        
        Args:
            model_path: Path to the TFLite model file
            label_path: Path to the labels file
            threshold: Confidence threshold for detections
            resolution: Camera resolution
        """
        self.model_path = model_path
        self.label_path = label_path
        self.threshold = threshold
        self.resolution = resolution
        
        self.video_stream = None
        self.detector = None
        self.processing_thread = None
        self.running = False
        
        # Detection results
        self.current_frame = None
        self.current_detections = []
        self.fps = 0
        self.lock = threading.Lock()
        
    def start(self):
        """Start the detection system."""
        if self.running:
            logger.warning("Detection system is already running")
            return
            
        logger.info("Starting lightweight detection system")
        
        # Initialize video stream
        logger.info(f"Starting video stream with resolution {self.resolution}")
        self.video_stream = VideoStream(resolution=self.resolution).start()
        time.sleep(1.0)  # Allow camera to warm up
        
        # Initialize detector
        logger.info(f"Initializing detector with model {self.model_path}")
        self.detector = LightweightDetector(
            model_path=self.model_path,
            label_path=self.label_path,
            threshold=self.threshold,
            resolution=self.resolution
        )
        
        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Lightweight detection system started")
        
    def _process_frames(self):
        """Process frames from the video stream."""
        while self.running:
            # Get frame from video stream
            frame = self.video_stream.read()
            if frame is None:
                logger.error("Failed to get frame from camera")
                time.sleep(0.1)
                continue
                
            # Process the frame with the detector
            results = self.detector.process_frame(frame)
            
            # Update detection results
            with self.lock:
                self.current_frame = results['frame']
                self.current_detections = results['detections']
                self.fps = results['fps']
                
    def get_current_frame(self):
        """Get the current processed frame."""
        with self.lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()
            
    def get_current_detections(self):
        """Get the current detections."""
        with self.lock:
            return self.current_detections.copy()
            
    def get_fps(self):
        """Get the current FPS."""
        with self.lock:
            return self.fps
            
    def stop(self):
        """Stop the detection system."""
        if not self.running:
            logger.warning("Detection system is not running")
            return
            
        logger.info("Stopping lightweight detection system")
        
        # Stop the processing thread
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            
        # Stop the video stream
        if self.video_stream:
            self.video_stream.stop()
            
        # Clean up the detector
        if self.detector:
            self.detector.cleanup()
            
        logger.info("Lightweight detection system stopped")

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lightweight Detection System')
    parser.add_argument('--model', type=str, default='models/person_detection_model.tflite',
                        help='Path to TFLite model file')
    parser.add_argument('--labels', type=str, default='models/person_labels.txt',
                        help='Path to label file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--resolution', type=str, default='640x480',
                        help='Camera resolution (WxH)')
    parser.add_argument('--display', action='store_true',
                        help='Display the detection results')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    resolution = (width, height)
    
    # Check if model file exists
    if not os.path.exists(args.model):
        # Try to find the model in the models directory
        model_dir = os.path.join(os.path.dirname(src_path), 'models')
        model_path = os.path.join(model_dir, os.path.basename(args.model))
        if os.path.exists(model_path):
            args.model = model_path
            logger.info(f"Using model from models directory: {args.model}")
        else:
            logger.warning(f"Model file not found: {args.model}")
            logger.warning("Using the pet detection model from BirdRepeller as fallback")
            args.model = os.path.join(os.path.dirname(src_path), 'BirdRepeller', 'models', 'pet_detection_model.tflite')
            args.labels = os.path.join(os.path.dirname(src_path), 'BirdRepeller', 'models', 'pet_labels.txt')
    
    # Initialize and start the detection system
    detection_system = LightweightDetectionSystem(
        model_path=args.model,
        label_path=args.labels,
        threshold=args.threshold,
        resolution=resolution
    )
    detection_system.start()
    
    try:
        if args.display:
            # Display the detection results
            while True:
                frame = detection_system.get_current_frame()
                if frame is not None:
                    cv2.imshow('Lightweight Detection', frame)
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        else:
            # Just keep the system running
            logger.info("Detection system running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1.0)
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Clean up
        detection_system.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 