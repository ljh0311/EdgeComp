#!/usr/bin/env python3
"""
Example script demonstrating how to use the detector factory.
This script shows how to easily switch between different detector implementations.
"""

import os
import sys
import time
import cv2
import argparse
import logging
from pathlib import Path

# Add the src directory to Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

# Import the detector factory
from babymonitor.detectors.detector_factory import DetectorFactory, DetectorType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detector Example')
    parser.add_argument('--detector', type=str, default='lightweight',
                        choices=['lightweight', 'yolov8', 'tracker'],
                        help='Type of detector to use')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (optional)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--resolution', type=str, default='640x480',
                        help='Camera resolution (WxH)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run inference on (cpu or cuda)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index')
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of threads for lightweight detector')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    resolution = (width, height)
    
    # Create configuration for detector
    detector_config = {
        "threshold": args.threshold,
        "resolution": resolution,
        "device": args.device,
        "num_threads": args.threads
    }
    
    # Add model path if specified
    if args.model:
        detector_config["model_path"] = args.model
    
    # Create configuration for video stream
    video_config = {
        "resolution": resolution,
        "framerate": 30,
        "camera_index": args.camera
    }
    
    # Create detector and video stream
    logger.info(f"Creating {args.detector} detector")
    detector = DetectorFactory.create_detector(args.detector, detector_config)
    
    logger.info(f"Creating video stream for {args.detector} detector")
    video_stream = DetectorFactory.create_video_stream(args.detector, video_config)
    
    # For lightweight detector, start the video stream
    if args.detector == DetectorType.LIGHTWEIGHT.value:
        video_stream.start()
        time.sleep(1.0)  # Allow camera to warm up
    
    # Main processing loop
    logger.info("Starting detection loop")
    try:
        while True:
            # Get frame from video stream
            if args.detector == DetectorType.LIGHTWEIGHT.value:
                frame = video_stream.read()
            else:
                ret, frame = video_stream.read()
                if not ret:
                    logger.error("Failed to get frame from camera")
                    break
            
            if frame is None:
                logger.error("Failed to get frame from camera")
                time.sleep(0.1)
                continue
            
            # Process the frame with the detector
            start_time = time.time()
            
            if args.detector == DetectorType.LIGHTWEIGHT.value:
                results = detector.process_frame(frame)
                processed_frame = results['frame']  # Frame with detections drawn
                detections = results['detections']
                fps = results['fps']
            else:
                # For YOLOv8 and tracker
                processed_frame, detections = detector.process_frame(frame)
                end_time = time.time()
                fps = 1.0 / (end_time - start_time)
                
                # Draw FPS on frame
                cv2.putText(processed_frame, f"FPS: {fps:.2f}",
                           (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Display information
            logger.debug(f"FPS: {fps:.2f}, Detections: {len(detections)}")
            
            # Display the frame
            cv2.imshow('Detector Example', processed_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Clean up
        logger.info("Cleaning up")
        if args.detector == DetectorType.LIGHTWEIGHT.value:
            video_stream.stop()
        else:
            video_stream.release()
            
        detector.cleanup()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 