"""
Person Tracking Module
===================
Wraps PersonDetector with status tracking capabilities.
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

class PersonTracker:
    """Tracks person status and history using optimized PersonDetector."""
    
    def __init__(self, model_path=None, force_cpu=None):
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect platform and set appropriate defaults
        self.platform = platform.system()
        self.device = self._detect_platform()
        
        # Initialize detector with platform-specific settings
        self.detector = PersonDetector(model_path=model_path, force_cpu=force_cpu)
        
        # Status tracking with minimal history
        self.detection_history = deque(maxlen=10)   # Reduced history size
        self.prev_detections = []
        self.history_size = 2  # Minimal history size
        
        # Status determination thresholds
        self.lying_ratio_threshold = 1.5
        self.movement_threshold = 20
        self.fall_threshold = 0.6
        self.fall_detection_frames = 3
        
        # Performance optimization
        self.frame_skip = 3              # Initial frame skip
        self.min_frame_skip = 2
        self.max_frame_skip = 15
        self.frame_count = 0
        self.processing_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.target_fps = 15
        
        # Thread safety
        self.lock = Lock()
        
        # Performance monitoring
        self.last_performance_check = time.time()
        self.performance_check_interval = 5.0
        
        # Frame processing optimization
        self.scale_factor = 1.0
        self.last_resize = None
        self.resize_cache = {}
        
    def _detect_platform(self):
        """Detect platform and set appropriate configuration."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == 'linux' and ('arm' in machine or 'aarch64' in machine):
            self.target_fps = 10
            self.scale_factor = 0.5
            return "raspberry_pi"
        return "windows" if system == 'windows' else "generic"

    def _adjust_frame_skip(self, current_fps):
        """Dynamically adjust frame skip based on performance metrics."""
        current_time = time.time()
        if current_time - self.last_performance_check < self.performance_check_interval:
            return
            
        self.last_performance_check = current_time
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        mem_percent = psutil.virtual_memory().percent
        
        # Adjust parameters based on performance
        if current_fps < self.target_fps * 0.7 or cpu_percent > 70:
            self.frame_skip = min(self.frame_skip + 2, self.max_frame_skip)
            if current_fps < self.target_fps * 0.5:
                self.scale_factor = max(0.5, self.scale_factor - 0.1)
        elif current_fps > self.target_fps * 1.3 and cpu_percent < 50:
            self.frame_skip = max(self.frame_skip - 1, self.min_frame_skip)
            self.scale_factor = min(1.0, self.scale_factor + 0.1)

    def _determine_status(self, bbox, prev_bbox=None):
        """Determine person's status based on position and movement."""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        aspect_ratio = width / height if height > 0 else 0
        
        # Quick status checks
        if aspect_ratio > self.lying_ratio_threshold:
            return "lying"
            
        if prev_bbox is None:
            return "seated"
            
        prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
        movement = abs(x1 - prev_x1) + abs(y1 - prev_y1)
        
        if movement > self.movement_threshold:
            if self.frame_count % self.frame_skip == 0:
                prev_height = prev_y2 - prev_y1
                height_change = abs(height - prev_height) / max(height, prev_height)
                position_change = abs(y2 - prev_y2)
                
                if height_change > self.fall_threshold and position_change > self.movement_threshold:
                    return "falling"
            return "moving"
            
        return "seated"

    def _resize_frame(self, frame):
        """Efficiently resize frame using caching."""
        if self.scale_factor == 1.0:
            return frame
            
        h, w = frame.shape[:2]
        cache_key = (w, h, self.scale_factor)
        
        if cache_key != self.last_resize:
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            self.resize_cache = cv2.resize(frame, (new_w, new_h))
            self.last_resize = cache_key
            return self.resize_cache
        
        return cv2.resize(frame, (int(w * self.scale_factor), int(h * self.scale_factor)))

    def process_frame(self, frame):
        """Process a frame and draw detections with status."""
        try:
            if frame is None or not isinstance(frame, np.ndarray):
                return {'frame': frame, 'detections': [], 'status_text': "No valid frame"}

            frame_start = time.time()
            self.frame_count += 1
            
            # Resize frame if needed
            frame = self._resize_frame(frame)
            
            # Step 1: Person Detection
            detections = self.detector.detect(frame)
            
            # Step 2: Process Detections
            processed_detections = []
            frame_copy = frame.copy()  # Single copy for drawing
            
            with self.lock:
                for det in detections:
                    try:
                        x1, y1, x2, y2, conf, cls_id = det
                        
                        # Skip low confidence detections
                        if conf < 0.3:
                            continue
                            
                        # Find matching previous detection
                        prev_bbox = None
                        if self.prev_detections:
                            min_dist = float('inf')
                            for prev in self.prev_detections:
                                px1, py1 = prev[:2]
                                curr_dist = ((x1 - px1) ** 2 + (y1 - py1) ** 2) ** 0.5
                                if curr_dist < min_dist:
                                    min_dist = curr_dist
                                    prev_bbox = prev
                        
                        # Step 3: Status Detection
                        status = self._determine_status([x1, y1, x2, y2], prev_bbox)
                        
                        # Add to processed detections
                        processed_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls_id,
                            'status': status
                        })
                        
                        # Draw detection
                        color = {
                            'falling': (0, 0, 255),
                            'lying': (255, 0, 0),
                            'moving': (0, 255, 0),
                            'seated': (255, 255, 0)
                        }.get(status, (0, 255, 0))
                        
                        cv2.rectangle(frame_copy, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    color, 2)
                        
                        # Only draw labels for important states
                        if status in ['falling', 'lying']:
                            cv2.putText(frame_copy,
                                      f"{status} ({conf:.2f})",
                                      (int(x1), int(y1 - 5)),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5,
                                      color,
                                      1)
                    
                    except Exception as e:
                        self.logger.warning(f"Error processing detection: {str(e)}")
                        continue
                
                # Update tracking data
                if len(processed_detections) > 0:
                    self.detection_history.append({
                        'timestamp': datetime.now(),
                        'detections': processed_detections
                    })
                    self.prev_detections = [d['bbox'] for d in processed_detections][-self.history_size:]
            
            # Calculate FPS and adjust parameters
            processing_time = time.time() - frame_start
            current_fps = 1.0 / processing_time if processing_time > 0 else 0
            self._adjust_frame_skip(current_fps)
            
            # Create minimal status text
            status_text = f"{len(processed_detections)} detected"
            if any(d['status'] == 'falling' for d in processed_detections):
                status_text += " - FALL DETECTED!"
            
            # Add performance overlay
            cv2.putText(frame_copy,
                       f"FPS: {current_fps:.1f} Skip: {self.frame_skip} Scale: {self.scale_factor:.1f}",
                       (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (255, 255, 255),
                       1)
            
            return {
                'frame': frame_copy,
                'detections': processed_detections,
                'status_text': status_text,
                'frame_skip': self.frame_skip,
                'processing_time': processing_time * 1000,  # Convert to ms
                'fps': current_fps
            }
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return {
                'frame': frame,
                'detections': [],
                'status_text': "Error processing frame",
                'frame_skip': self.frame_skip,
                'processing_time': 0,
                'fps': 0
            }

    def get_status_summary(self, time_range=None):
        """Get summary of status history within time range."""
        if not self.detection_history:
            return {'no_detection': 1.0}
            
        if time_range is None:
            relevant_history = self.detection_history
        else:
            cutoff_time = datetime.now() - time_range
            relevant_history = [
                entry for entry in self.detection_history 
                if entry['timestamp'] > cutoff_time
            ]
        
        # Count occurrences of each status
        total_count = len(relevant_history)
        status_counts = {}
        
        for entry in relevant_history:
            statuses = [det['status'] for det in entry['detections']]
            for status in statuses:
                status_counts[status] = status_counts.get(status, 0) + 1
        
        # Convert to percentages
        return {
            status: count / total_count 
            for status, count in status_counts.items()
        }

    def get_activity_timeline(self, time_range=None):
        """Get timeline of activity within time range."""
        if time_range is None:
            return list(self.detection_history)
            
        cutoff_time = datetime.now() - time_range
        return [
            entry for entry in self.detection_history 
            if entry['timestamp'] > cutoff_time
        ]

    def __del__(self):
        """Cleanup resources."""
        try:
            # Clear histories
            self.detection_history.clear()
            self.prev_detections.clear()
        except Exception as e:
            self.logger.error(f"Error cleaning up person tracker: {str(e)}") 