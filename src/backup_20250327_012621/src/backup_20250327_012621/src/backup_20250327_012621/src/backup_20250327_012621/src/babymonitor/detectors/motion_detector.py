"""
Motion Detector Module
===================
Detects motion in video frames using OpenCV.
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple
from .base_detector import BaseDetector

class MotionDetector(BaseDetector):
    """Detects motion in video frames using background subtraction."""
    
    def __init__(self, 
                 threshold: float = 0.5,
                 min_area: int = 500,
                 history: int = 50):
        """Initialize the motion detector.
        
        Args:
            threshold: Detection confidence threshold
            min_area: Minimum contour area to consider as motion
            history: History length for background subtractor
        """
        super().__init__(threshold=threshold)
        
        self.logger = logging.getLogger(__name__)
        self.min_area = min_area
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=16,
            detectShadows=False
        )
        
        # Initialize state
        self.last_frame = None
        self.last_motion_time = time.time()
        self.motion_detected = False
        
        self.logger.info("Motion detector initialized")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a frame and detect motion.
        
        Args:
            frame: Input frame
            
        Returns:
            Dict containing processed frame and motion detection results
        """
        start_time = time.time()
        
        if frame is None or not isinstance(frame, np.ndarray):
            return {
                'frame': frame, 
                'motion_detected': False,
                'motion_areas': [],
                'fps': 0
            }

        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(gray)
            
            # Threshold the mask
            thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            motion_areas = []
            motion_detected = False
            
            for contour in contours:
                if cv2.contourArea(contour) < self.min_area:
                    continue
                    
                # Motion detected
                motion_detected = True
                
                # Get bounding box
                (x, y, w, h) = cv2.boundingRect(contour)
                motion_areas.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': float(cv2.contourArea(contour))
                })
                
                # Draw rectangle around motion area
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Update state
            if motion_detected:
                self.last_motion_time = time.time()
                self.motion_detected = True
            elif time.time() - self.last_motion_time > 1.0:  # Reset after 1 second of no motion
                self.motion_detected = False
            
            # Calculate FPS
            self.update_fps(time.time() - start_time)
            
            # Add text to frame
            cv2.putText(
                frame,
                f"Motion: {'Yes' if self.motion_detected else 'No'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if self.motion_detected else (0, 255, 0),
                2
            )
            
            return {
                'frame': frame,
                'motion_detected': self.motion_detected,
                'motion_areas': motion_areas,
                'fps': self.get_fps()
            }
            
        except Exception as e:
            self.logger.error(f"Error in motion detection: {str(e)}")
            return {
                'frame': frame,
                'motion_detected': False,
                'motion_areas': [],
                'fps': 0
            }

    def cleanup(self):
        """Clean up resources."""
        self.bg_subtractor = None
        self.last_frame = None 