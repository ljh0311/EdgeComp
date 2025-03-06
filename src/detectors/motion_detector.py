"""
Motion Detection Module
=====================
Handles motion detection and fall detection using frame analysis and aspect ratios.
"""

import cv2
import numpy as np
import collections
import logging
from typing import Dict, List, Tuple, Optional
import torch

class MotionDetector:
    """Motion detector for detecting rapid motion and potential falls."""

    def __init__(self, config, device=None):
        """Initialize the motion detector."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.get('HISTORY', 500),
            varThreshold=self.config.get('VAR_THRESHOLD', 16),
            detectShadows=False
        )
        
        # Initialize parameters
        self.motion_history = []
        self.fall_threshold = self.config.get('FALL_THRESHOLD', 0.6)
        self.rapid_motion_threshold = self.config.get('RAPID_MOTION_THRESHOLD', 0.3)
        self.history_size = self.config.get('MOTION_HISTORY_SIZE', 10)
        
        self.logger.info(f"Motion detector initialized on {self.device}")

    def detect(self, frame, person_detections):
        """Detect motion and potential falls in the frame."""
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # If using GPU, move frame to GPU
            if self.device.type == 'cuda':
                gray_tensor = torch.from_numpy(gray).to(self.device)
                
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(gray)
            
            # Calculate motion metrics
            motion_pixels = np.count_nonzero(fg_mask)
            total_pixels = fg_mask.size
            motion_ratio = motion_pixels / total_pixels
            
            # Update motion history
            self.motion_history.append(motion_ratio)
            if len(self.motion_history) > self.history_size:
                self.motion_history.pop(0)
            
            # Detect rapid motion
            rapid_motion = motion_ratio > self.rapid_motion_threshold
            
            # Process person detections for fall detection
            fall_detected = False
            if person_detections:
                for detection in person_detections:
                    x1, y1, x2, y2 = map(int, detection[:4])
                    person_height = y2 - y1
                    person_width = x2 - x1
                    aspect_ratio = person_width / person_height if person_height > 0 else 0
                    
                    # Check for fall based on aspect ratio
                    if aspect_ratio > self.fall_threshold:
                        fall_detected = True
                        break
            
            # Draw motion visualization
            motion_frame = frame.copy()
            if rapid_motion:
                cv2.putText(motion_frame, "RAPID MOTION", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if fall_detected:
                cv2.putText(motion_frame, "FALL DETECTED", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Draw person detections
            for detection in person_detections:
                x1, y1, x2, y2 = map(int, detection[:4])
                cv2.rectangle(motion_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            return motion_frame, rapid_motion, fall_detected
            
        except Exception as e:
            self.logger.error(f"Error in motion detection: {str(e)}")
            return frame, False, False

    def reset(self):
        """Reset the motion detector state."""
        self.motion_history.clear()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.get('HISTORY', 500),
            varThreshold=self.config.get('VAR_THRESHOLD', 16),
            detectShadows=False
        ) 