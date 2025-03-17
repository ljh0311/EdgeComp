"""
Simple Person Detector Module
---------------------------
Uses MobileNet SSD for efficient person detection and basic activity status tracking.
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any

# Configure logging
logger = logging.getLogger(__name__)

class SimpleDetector:
    """Simple person detector using MobileNet SSD."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize the detector.
        
        Args:
            confidence_threshold: Minimum confidence threshold for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_model_loaded = False
        self.process_resolution = (300, 300)  # MobileNet default input size
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Activity tracking
        self.last_positions = []
        self.position_history_size = 5
        self.movement_threshold = 20  # pixels
        self.lying_aspect_ratio_threshold = 1.8
        
        # Initialize model
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the MobileNet SSD model."""
        try:
            # Load model files from the standard OpenCV path
            model_path = "models/mobilenet_ssd/"
            prototxt = f"{model_path}MobileNetSSD_deploy.prototxt"
            model = f"{model_path}MobileNetSSD_deploy.caffemodel"
            
            self.model = cv2.dnn.readNetFromCaffe(prototxt, model)
            
            # Use CPU backend only
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("Using CPU backend")
            
            self.is_model_loaded = True
            logger.info("MobileNet SSD model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_model_loaded = False
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a frame to detect people and their activities.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Dictionary containing detections and metadata
        """
        if not self.is_model_loaded:
            return {"frame": frame, "detections": [], "fps": 0}
        
        # Update FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_fps_time
        
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # Prepare frame for detection
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            0.007843,
            self.process_resolution,
            127.5
        )
        
        # Detect objects
        self.model.setInput(blob)
        detections = self.model.forward()
        
        # Process detections
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            
            # Class 15 is person in MobileNet SSD
            if class_id == 15 and confidence > self.confidence_threshold:
                # Get coordinates
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                xmin, ymin, xmax, ymax = box.astype(int)
                
                # Ensure coordinates are within frame
                xmin = max(0, min(xmin, width - 1))
                ymin = max(0, min(ymin, height - 1))
                xmax = max(0, min(xmax, width - 1))
                ymax = max(0, min(ymax, height - 1))
                
                # Calculate box center for movement tracking
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2
                
                # Calculate aspect ratio for posture detection
                box_width = xmax - xmin
                box_height = ymax - ymin
                aspect_ratio = box_width / box_height if box_height > 0 else 0
                
                # Determine activity status
                status = self._determine_activity_status(
                    (center_x, center_y),
                    aspect_ratio
                )
                
                # Create detection result
                detection = {
                    "box": [xmin, ymin, xmax, ymax],
                    "confidence": float(confidence),
                    "class": "person",
                    "status": status
                }
                
                results.append(detection)
        
        return {
            "frame": frame,
            "detections": results,
            "fps": self.fps
        }
    
    def _determine_activity_status(self, current_position: Tuple[int, int], aspect_ratio: float) -> str:
        """Determine the activity status based on position history and aspect ratio.
        
        Args:
            current_position: Current center position (x, y)
            aspect_ratio: Width/height ratio of bounding box
            
        Returns:
            Activity status string
        """
        # Add current position to history
        self.last_positions.append(current_position)
        if len(self.last_positions) > self.position_history_size:
            self.last_positions.pop(0)
        
        # Check if lying down based on aspect ratio
        if aspect_ratio > self.lying_aspect_ratio_threshold:
            return "lying"
        
        # Check if moving based on position history
        if len(self.last_positions) >= 2:
            prev_x, prev_y = self.last_positions[-2]
            curr_x, curr_y = current_position
            
            movement = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            if movement > self.movement_threshold:
                return "moving"
        
        return "seated"
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.model = None
        self.is_model_loaded = False
        logger.info("SimpleDetector cleaned up") 