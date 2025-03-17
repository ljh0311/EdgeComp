"""
Test Detectors
------------
This script tests all detectors with a camera feed.
It allows testing different detector types and configurations.
"""

import os
import sys
import time
import logging
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run the detector test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Detectors')
    parser.add_argument('--detector', type=str, default='tracker', 
                        choices=['yolov8', 'lightweight', 'tracker', 'motion', 'emotion'],
                        help='Detector type to test')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    parser.add_argument('--fps', type=int, default=30, help='Camera FPS')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model to use (for applicable detectors)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--frame-skip', type=int, default=2, help='Frame skip value')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Import modules
    from src.babymonitor.detectors.detector_factory import DetectorFactory
    from src.babymonitor.utils.model_manager import ModelManager
    
    # Configure detector
    detector_config = {
        "model_path": args.model,
        "threshold": args.threshold,
        "force_cpu": args.force_cpu,
        "frame_skip": args.frame_skip,
        "process_resolution": (320, 320)
    }
    
    # Create detector
    logger.info(f"Creating {args.detector} detector...")
    detector = DetectorFactory.create_detector(
        detector_type=args.detector,
        config=detector_config
    )
    
    # Create video stream
    logger.info(f"Creating video stream with camera index {args.camera}...")
    video_stream = DetectorFactory.create_video_stream(
        detector_type=args.detector,
        config={
            "resolution": (args.width, args.height),
            "framerate": args.fps,
            "camera_index": args.camera,
            "buffer_size": 1
        }
    )
    
    # Start video stream if it has a start method
    if hasattr(video_stream, 'start'):
        video_stream.start()
        logger.info("Video stream started")
    
    # Process frames
    logger.info("Processing frames...")
    
    try:
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        while True:
            # Read frame
            if hasattr(video_stream, 'read'):
                ret_val = video_stream.read()
                
                # Handle different return types from read()
                if isinstance(ret_val, tuple) and len(ret_val) == 2:
                    ret, frame = ret_val
                    if not ret:
                        logger.warning("Failed to capture frame")
                        time.sleep(0.01)
                        continue
                else:
                    frame = ret_val
                    if frame is None:
                        logger.warning("Failed to capture frame (None)")
                        time.sleep(0.01)
                        continue
            else:
                logger.error("Video stream does not have a read method")
                break
            
            # Process frame
            frame_count += 1
            results = detector.process_frame(frame)
            
            # Get processed frame
            processed_frame = results.get("frame", frame)
            
            # Get detections
            detections = results.get("detections", [])
            
            # Draw detections on frame
            for detection in detections:
                # Get detection information
                box = detection.get("box", [0, 0, 0, 0])
                confidence = detection.get("confidence", 0)
                class_name = detection.get("class", "unknown")
                
                # Convert box to integers
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Draw bounding box
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(
                    processed_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
                # Draw track ID if available
                if "track_id" in detection:
                    track_id = detection["track_id"]
                    cv2.putText(
                        processed_frame,
                        f"ID: {track_id}",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )
                    
                # Draw status if available
                if "status" in detection:
                    status = detection["status"]
                    color = (0, 255, 0)  # Green for normal
                    
                    if status == "falling":
                        color = (0, 0, 255)  # Red for falling
                    elif status == "lying":
                        color = (0, 165, 255)  # Orange for lying
                        
                    cv2.putText(
                        processed_frame,
                        f"Status: {status}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
            
            # Calculate FPS
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = current_time
                
                # Log FPS and detection count
                logger.info(f"FPS: {fps:.2f}, Detections: {len(detections)}")
            
            # Add FPS to frame
            cv2.putText(
                processed_frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Show frame
            cv2.imshow("Detector Test", processed_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on 'q' key
            if key == ord('q'):
                break
                
            # Toggle frame skip on 's' key
            elif key == ord('s'):
                new_frame_skip = 0 if detector.frame_skip > 0 else 2
                detector.set_frame_skip(new_frame_skip)
                logger.info(f"Frame skip set to {new_frame_skip}")
                
            # Adjust frame skip on '+' and '-' keys
            elif key == ord('+'):
                detector.set_frame_skip(detector.frame_skip + 1)
                logger.info(f"Frame skip set to {detector.frame_skip}")
            elif key == ord('-'):
                detector.set_frame_skip(max(0, detector.frame_skip - 1))
                logger.info(f"Frame skip set to {detector.frame_skip}")
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Clean up
        logger.info("Cleaning up...")
        
        # Release video stream
        if hasattr(video_stream, 'stop'):
            video_stream.stop()
        elif hasattr(video_stream, 'release'):
            video_stream.release()
            
        # Clean up detector
        detector.cleanup()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main() 