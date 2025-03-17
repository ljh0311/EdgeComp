"""
Motion Detector Module
---------------------
Implements motion detection and fall detection using OpenCV.
This module provides efficient motion detection with minimal resource usage.
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Tuple, List, Optional
import psutil
from .base_detector import BaseDetector

# Configure logging
logger = logging.getLogger(__name__)

class MotionDetector(BaseDetector):
    """Motion and fall detector using OpenCV."""
    
    def __init__(self, 
                 motion_threshold: float = 0.02,
                 fall_threshold: float = 0.15,
                 history: int = 500,
                 var_threshold: float = 16,
                 detect_shadows: bool = False):
        """Initialize the motion detector.
        
        Args:
            motion_threshold: Threshold for motion detection (percentage of frame)
            fall_threshold: Threshold for fall detection (percentage of frame)
            history: History length for background subtractor
            var_threshold: Variance threshold for background subtractor
            detect_shadows: Whether to detect shadows
        """
        super().__init__(threshold=motion_threshold)
        
        # Motion detection parameters
        self.motion_threshold = motion_threshold
        self.fall_threshold = fall_threshold
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        
        # Performance optimization
        self.frame_skip = 1
        self.process_resolution = (320, 240)  # Lower resolution for processing
        self.original_resolution = None
        self.processing_scale = 1.0
        
        # Motion history
        self.motion_history = []
        self.max_motion_history = 10
        self.last_motion_level = 0
        self.last_fall_detected = False
        
        logger.info(f"Motion detector initialized with threshold {motion_threshold}")
    
    def process_frame(self, frame: np.ndarray, detections: List[Dict] = None) -> Dict:
        """Process a single frame for motion and fall detection.
        
        Args:
            frame: Input frame (numpy array)
            detections: Optional list of detections from other detectors
            
        Returns:
            Dict containing processed frame and detection results
        """
        start_time = time.time()
        
        # Skip frames for performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0 and self.last_frame is not None:
            return {
                'frame': self.last_frame,
                'motion_level': self.last_motion_level,
                'fall_detected': self.last_fall_detected,
                'fps': self.fps
            }
        
        # Store original frame dimensions
        if self.original_resolution is None:
            self.original_resolution = (frame.shape[1], frame.shape[0])
        
        # Create a working copy of the frame
        display_frame = frame.copy()
        
        try:
            # Resize frame for processing (smaller = faster)
            process_frame = cv2.resize(frame, self.process_resolution)
            
            # Convert to grayscale for faster processing
            gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(gray)
            
            # Remove shadows (if any)
            _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Calculate motion level (percentage of pixels with motion)
            motion_level = np.count_nonzero(fg_mask) / (fg_mask.shape[0] * fg_mask.shape[1])
            
            # Store motion level in history
            self.motion_history.append(motion_level)
            if len(self.motion_history) > self.max_motion_history:
                self.motion_history.pop(0)
            
            # Detect rapid motion (potential fall)
            fall_detected = False
            if len(self.motion_history) > 1:
                motion_diff = abs(motion_level - self.motion_history[-2])
                if motion_diff > self.fall_threshold:
                    fall_detected = True
                    logger.info(f"Fall detected! Motion difference: {motion_diff:.4f}")
            
            # Detect if motion is above threshold
            motion_detected = motion_level > self.motion_threshold
            
            # Draw motion information on frame
            cv2.putText(display_frame, f"Motion: {motion_level:.4f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if motion_detected else (0, 255, 0), 2)
            
            if fall_detected:
                cv2.putText(display_frame, "FALL DETECTED!", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # If we have person detections, check if they're in a lying position
            if detections:
                for detection in detections:
                    if 'bbox' in detection and detection.get('class', '').lower() == 'person':
                        xmin, ymin, xmax, ymax = detection['bbox']
                        width = xmax - xmin
                        height = ymax - ymin
                        
                        # Check if person is in a lying position (width > height)
                        if width > height * 1.2:  # Person is wider than tall (with some margin)
                            cv2.putText(display_frame, "PERSON LYING DOWN", (xmin, ymin - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            detection['status'] = 'lying_down'
                            
                            # If motion was also detected, this is likely a fall
                            if motion_detected:
                                fall_detected = True
                                cv2.putText(display_frame, "FALL DETECTED!", (10, 70),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Resize the mask to match the original frame for visualization
            if motion_detected or fall_detected:
                # Resize mask for display
                display_mask = cv2.resize(fg_mask, (display_frame.shape[1], display_frame.shape[0]))
                
                # Create a colored mask for visualization
                colored_mask = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2BGR)
                colored_mask[display_mask > 0] = [0, 0, 255]  # Red for motion areas
                
                # Blend the mask with the original frame
                alpha = 0.3
                display_frame = cv2.addWeighted(display_frame, 1 - alpha, colored_mask, alpha, 0)
            
            # Store results for frame skipping
            self.last_frame = display_frame
            self.last_motion_level = motion_level
            self.last_fall_detected = fall_detected
            
            # Calculate FPS
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_history:
                self.frame_times.pop(0)
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            
            # Adjust frame skip based on performance
            self._adjust_frame_skip()
            
            return {
                'frame': display_frame,
                'motion_level': motion_level,
                'motion_detected': motion_detected,
                'fall_detected': fall_detected,
                'fps': self.fps
            }
            
        except Exception as e:
            logger.error(f"Error processing frame for motion detection: {e}")
            return {
                'frame': frame,
                'motion_level': 0,
                'motion_detected': False,
                'fall_detected': False,
                'fps': 0
            }
    
    def _adjust_frame_skip(self):
        """Adjust frame skip based on performance."""
        avg_time = self.get_processing_time()
        if avg_time > 0:
            target_fps = 20  # Target FPS (higher than person detector since motion detection is faster)
            ideal_skip = max(1, int(avg_time * target_fps))
            
            # Gradually adjust frame skip
            if ideal_skip > self.frame_skip:
                self.frame_skip = min(ideal_skip, self.frame_skip + 1)
            elif ideal_skip < self.frame_skip and self.frame_skip > 1:
                self.frame_skip = max(1, self.frame_skip - 1)
    
    def detect(self, frame: np.ndarray, detections: List[Dict] = None) -> Dict:
        """Alias for process_frame to maintain backward compatibility."""
        return self.process_frame(frame, detections)
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Release background subtractor
            self.bg_subtractor = None
        except Exception as e:
            logger.error(f"Error cleaning up motion detector: {e}") 