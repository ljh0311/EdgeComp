"""
Person Tracker Module
===================
Tracks person movements and status using optimized person detection.
"""

import cv2
import numpy as np
import time
import logging
from datetime import datetime
from threading import Lock
from typing import Dict, List, Any, Optional
from .base_detector import BaseDetector
from .person_detector import PersonDetector

class PersonTracker(BaseDetector):
    """Tracks person status and history using optimized PersonDetector."""
    
    def __init__(self, 
                 detector: Optional[PersonDetector] = None,
                 threshold: float = 0.5,
                 max_history: int = 30):
        """Initialize the person tracker.
        
        Args:
            detector: PersonDetector instance (will create new one if None)
            threshold: Detection confidence threshold
            max_history: Maximum number of frames to keep in history
        """
        super().__init__(threshold=threshold)
        
        self.logger = logging.getLogger(__name__)
        self.detector = detector or PersonDetector()
        
        # Performance optimization
        self.frame_skip = 1
        self.process_resolution = (640, 480)
        self.scale_factor = 1.0
        
        # Thread safety
        self.lock = Lock()
        
        # Detection history
        self.detection_history = []
        self.max_history = max_history
        self.prev_detections = []
        self.last_motion_status = 'unknown'
        
        # Performance metrics
        self.frame_count = 0
        self.fps = 0.0
        self.frame_times = []
        self.max_frame_history = 30
        
        self.logger.info("Person tracker initialized")

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame for processing while maintaining aspect ratio."""
        if frame is None:
            return None
            
        height, width = frame.shape[:2]
        target_width, target_height = self.process_resolution
        
        # Calculate scale factor
        scale = min(target_width/width, target_height/height)
        if scale >= 1:
            return frame
            
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return cv2.resize(frame, (new_width, new_height))

    def _analyze_motion(self, detections: List[Dict]) -> str:
        """Analyze motion status based on current detections."""
        if not detections:
            return 'no_detection'
            
        # Count number of people
        num_people = len(detections)
        
        if num_people == 0:
            return 'no_detection'
        elif num_people == 1:
            return 'single_person'
        else:
            return f'multiple_people_{num_people}'

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a frame and return detection results.
        
        Args:
            frame: Input frame
            
        Returns:
            Dict containing processed frame and detection results
        """
        start_time = time.time()
        
        if frame is None or not isinstance(frame, np.ndarray):
            return {
                'frame': frame, 
                'detections': [], 
                'motion_status': 'invalid_frame',
                'fps': self.fps
            }

        try:
            self.frame_count += 1
            
            # Skip frames for performance
            if self.frame_count % self.frame_skip != 0 and hasattr(self, 'last_frame'):
                return {
                    'frame': self.last_frame,
                    'detections': self.prev_detections,
                    'motion_status': self.last_motion_status,
                    'fps': self.fps
                }
            
            # Resize frame if needed
            if self.scale_factor != 1.0:
                process_frame = self._resize_frame(frame)
            else:
                process_frame = frame
            
            # Process frame with detector
            processed_frame, detections = self.detector.process_frame(process_frame)
            
            # Store last processed frame
            self.last_frame = processed_frame
            
            # Analyze motion and status changes
            motion_status = self._analyze_motion(detections)
            self.last_motion_status = motion_status
            
            # Update tracking data efficiently
            if detections:
                with self.lock:
                    self.detection_history.append({
                        'timestamp': datetime.now(),
                        'detections': detections,
                        'motion_status': motion_status
                    })
                    
                    # Trim history if needed
                    while len(self.detection_history) > self.max_history:
                        self.detection_history.pop(0)
                        
                    self.prev_detections = detections
            
            # Calculate FPS
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_history:
                self.frame_times.pop(0)
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            
            # Adjust frame skip based on performance
            self._adjust_frame_skip()
            
            return {
                'frame': processed_frame,
                'detections': detections,
                'motion_status': motion_status,
                'fps': self.fps
            }
            
        except Exception as e:
            self.logger.error(f"Error in process_frame: {str(e)}")
            return {
                'frame': frame,
                'detections': [],
                'motion_status': 'error',
                'fps': 0
            }

    def _adjust_frame_skip(self):
        """Adjust frame skip based on performance."""
        avg_time = self.get_processing_time()
        if avg_time > 0:
            target_fps = 15  # Target FPS
            ideal_skip = max(1, int(avg_time * target_fps))
            
            # Gradually adjust frame skip
            if ideal_skip > self.frame_skip:
                self.frame_skip = min(ideal_skip, self.frame_skip + 1)
            elif ideal_skip < self.frame_skip and self.frame_skip > 1:
                self.frame_skip = max(1, self.frame_skip - 1)

    def get_detection_history(self) -> List[Dict]:
        """Get the detection history."""
        with self.lock:
            return self.detection_history.copy()

    def cleanup(self):
        """Clean up resources."""
        try:
            with self.lock:
                self.detection_history.clear()
                self.prev_detections.clear()
                if hasattr(self, 'detector'):
                    self.detector = None
        except Exception as e:
            self.logger.error(f"Error cleaning up tracker: {str(e)}") 