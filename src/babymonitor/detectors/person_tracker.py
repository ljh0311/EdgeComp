"""
Person Tracking Module
---------------------
Implements person tracking with status tracking capabilities.
This module wraps PersonDetector with additional tracking functionality.
"""

import logging
import time
from datetime import datetime
import os
import numpy as np
from collections import deque
from .person_detector import PersonDetector
import cv2
import platform
import psutil
from threading import Lock
from typing import Dict, List, Tuple, Optional, Any
from .base_detector import BaseDetector

# Configure logging
logger = logging.getLogger(__name__)

class PersonTracker(BaseDetector):
    """Tracks person status and history using optimized PersonDetector."""
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 threshold: float = 0.5,
                 force_cpu: Optional[bool] = None):
        """Initialize the person tracker.
        
        Args:
            model_path: Path to the detection model
            threshold: Detection confidence threshold
            force_cpu: Whether to force CPU usage
        """
        super().__init__(threshold=threshold)
        
        # Auto-detect platform and set appropriate defaults
        self.platform = platform.system()
        self.device_type = self._detect_platform()
        
        # Initialize detector with platform-specific settings
        self.detector = PersonDetector(model_path=model_path, threshold=threshold, force_cpu=force_cpu)
        
        # Status tracking with minimal history
        self.detection_history = deque(maxlen=5)   # Minimal history
        self.prev_detections = []
        self.history_size = 2
        
        # Status determination thresholds
        self.lying_ratio_threshold = 1.5
        self.movement_threshold = 20
        self.fall_threshold = 0.6
        self.fall_detection_frames = 2  # Reduced frames needed for fall detection
        
        # Performance optimization
        self.frame_skip = 1
        self.min_frame_skip = 1
        self.max_frame_skip = 3          # Further reduced for smoother video
        
        # Thread safety
        self.lock = Lock()
        
        # Performance monitoring
        self.last_performance_check = time.time()
        self.performance_check_interval = 0.5  # More frequent checks
        
        # Frame processing optimization
        self.scale_factor = 1.0
        self.min_scale_factor = 0.75
        self.max_scale_factor = 1.0
        self.resize_cache = None
        self.last_resize_dims = None
        
        # Last processed results
        self.last_motion_status = 'unknown'
        
        logger.info(f"Person tracker initialized for {self.device_type} platform")
        
    def _detect_platform(self) -> str:
        """Detect platform and set appropriate configuration.
        
        Returns:
            Platform type string
        """
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == 'linux' and ('arm' in machine or 'aarch64' in machine):
            self.target_fps = 10
            self.scale_factor = 0.5
            return "raspberry_pi"
        return "windows" if system == 'windows' else "generic"

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Efficiently resize frame using simplified caching.
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        if self.scale_factor == 1.0:
            return frame
            
        h, w = frame.shape[:2]
        new_dims = (w, h, self.scale_factor)
        
        # Only resize if dimensions changed
        if new_dims != self.last_resize_dims:
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            interpolation = cv2.INTER_AREA if self.scale_factor < 1.0 else cv2.INTER_LINEAR
            self.resize_cache = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
            self.last_resize_dims = new_dims
            
        return self.resize_cache

    def _adjust_frame_skip(self):
        """Dynamically adjust frame skip based on performance metrics."""
        current_time = time.time()
        if current_time - self.last_performance_check < self.performance_check_interval:
            return
            
        self.last_performance_check = current_time
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        current_fps = self.fps
        target_fps = getattr(self, 'target_fps', 25)
        
        # Adjust parameters based on performance
        if current_fps < target_fps * 0.7 or cpu_percent > 80:  # More aggressive adjustment
            self.frame_skip = min(self.frame_skip + 1, self.max_frame_skip)
            if current_fps < target_fps * 0.4:  # More aggressive scaling
                self.scale_factor = max(self.min_scale_factor, self.scale_factor - 0.1)
        elif current_fps > target_fps * 1.3 and cpu_percent < 65:  # More aggressive improvement
            if self.frame_skip > self.min_frame_skip:
                self.frame_skip = max(self.frame_skip - 1, self.min_frame_skip)
            elif self.scale_factor < self.max_scale_factor:
                self.scale_factor = min(self.max_scale_factor, self.scale_factor + 0.1)

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
                    'frame_skip': self.frame_skip,
                    'fps': self.fps
                }
            
            # Resize frame if needed
            if self.scale_factor != 1.0:
                process_frame = self._resize_frame(frame)
            else:
                process_frame = frame
            
            # Process frame with detector
            detector_result = self.detector.process_frame(process_frame)
            
            # Store last processed frame (only the frame with bounding boxes)
            self.last_frame = detector_result['frame']
            
            # Extract detections
            detections = detector_result['detections']
            
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
                'frame': self.last_frame,
                'detections': detections,
                'motion_status': motion_status,
                'frame_skip': self.frame_skip,
                'fps': self.fps
            }
            
        except Exception as e:
            logger.error(f"Error in process_frame: {str(e)}")
            return {
                'frame': frame,
                'detections': [],
                'motion_status': 'error',
                'frame_skip': self.frame_skip,
                'fps': 0
            }

    def _analyze_motion(self, current_detections: List[Dict]) -> str:
        """Analyze motion and status changes between frames.
        
        Args:
            current_detections: List of current detections
            
        Returns:
            Motion status string
        """
        if not self.detection_history or not current_detections:
            return 'no_motion'
            
        try:
            # Get previous detections
            prev_entry = self.detection_history[-1]
            prev_detections = prev_entry['detections']
            
            if not prev_detections:
                return 'new_detection'
            
            # Compare current and previous detections
            max_movement = 0
            status_changed = False
            
            for curr_det in current_detections:
                if 'bbox' not in curr_det:
                    continue
                    
                curr_bbox = curr_det['bbox']
                curr_x = (curr_bbox[0] + curr_bbox[2]) / 2  # Current center x
                curr_y = (curr_bbox[1] + curr_bbox[3]) / 2  # Current center y
                curr_status = curr_det.get('status', 'detected')
                
                # Find closest previous detection
                min_dist = float('inf')
                prev_status = None
                
                for prev_det in prev_detections:
                    if 'bbox' not in prev_det:
                        continue
                        
                    prev_bbox = prev_det['bbox']
                    prev_x = (prev_bbox[0] + prev_bbox[2]) / 2
                    prev_y = (prev_bbox[1] + prev_bbox[3]) / 2
                    
                    # Calculate distance between detections
                    dist = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        prev_status = prev_det.get('status', 'detected')
                
                # Update maximum movement
                max_movement = max(max_movement, min_dist)
                
                # Check if status changed
                if prev_status and prev_status != curr_status:
                    status_changed = True
            
            # Determine motion status
            if max_movement > self.movement_threshold:
                return 'rapid_motion'
            elif status_changed:
                return 'status_changed'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"Error analyzing motion: {str(e)}")
            return 'error'

    def get_status_summary(self, time_range=None) -> Dict[str, float]:
        """Get summary of status history within time range.
        
        Args:
            time_range: Optional time range to consider
            
        Returns:
            Dict with status summary
        """
        if not self.detection_history:
            return {'no_detection': 1.0}
            
        if time_range is None:
            relevant_history = list(self.detection_history)
        else:
            cutoff_time = datetime.now() - time_range
            relevant_history = [h for h in self.detection_history if h['timestamp'] > cutoff_time]
        
        if not relevant_history:
            return {'no_detection': 1.0}
        
        # Count status occurrences
        status_counts = {}
        for entry in relevant_history:
            status = entry['motion_status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Convert to percentages
        total = len(relevant_history)
        return {status: count / total for status, count in status_counts.items()}

    def get_activity_timeline(self, time_range=None) -> List[Dict]:
        """Get timeline of activity within time range.
        
        Args:
            time_range: Optional time range to consider
            
        Returns:
            List of activity entries
        """
        if time_range is None:
            return list(self.detection_history)
        
        cutoff_time = datetime.now() - time_range
        return [h for h in self.detection_history if h['timestamp'] > cutoff_time]

    def cleanup(self):
        """Clean up resources."""
        try:
            # Clean up detector
            if hasattr(self, 'detector') and self.detector is not None:
                self.detector.cleanup()
            
            # Clear history
            self.detection_history.clear()
            self.prev_detections = []
            
            # Clear cache
            self.resize_cache = None
            self.last_resize_dims = None
            
        except Exception as e:
            logger.error(f"Error cleaning up person tracker: {str(e)}")
    
    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup() 