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
from ..models.person_detector import PersonDetector
import cv2

class PersonTracker:
    """Tracks person status and history using optimized PersonDetector."""
    
    def __init__(self, model_path="models/yolov8n.pt", device=None):
        self.logger = logging.getLogger(__name__)
        
        # Ensure model path is absolute
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, model_path)
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.detector = PersonDetector(model_path)
        
        # Status tracking
        self.detection_history = deque(maxlen=100)  # Store last 100 detections
        self.status_history = deque(maxlen=1000)    # Store last 1000 status updates
        self.prev_detections = []
        self.history_size = 5  # For immediate history
        
        # Status determination thresholds
        self.lying_ratio_threshold = 1.5
        self.movement_threshold = 20
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 300  # Cleanup every 5 minutes
        
    def _determine_status(self, bbox, prev_bbox=None):
        """Determine person's status based on position and movement."""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        aspect_ratio = width / height if height > 0 else 0

        # Determine if lying down based on aspect ratio
        if aspect_ratio > self.lying_ratio_threshold:
            return "lying"
        
        # Check for movement if we have previous detection
        if prev_bbox is not None:
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
            movement = abs(x1 - prev_x1) + abs(y1 - prev_y1)
            if movement > self.movement_threshold:
                return "moving"
        
        return "seated"

    def _update_history(self, detections, timestamp=None):
        """Update detection and status history."""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Update detection history
        self.detection_history.append({
            'timestamp': timestamp,
            'count': len(detections),
            'detections': detections
        })
        
        # Update status history with aggregated status
        statuses = [det[6] for det in detections] if detections else ['no_detection']
        status_summary = {
            'timestamp': timestamp,
            'primary_status': max(set(statuses), key=statuses.count) if statuses else 'no_detection',
            'status_counts': {status: statuses.count(status) for status in set(statuses)}
        }
        self.status_history.append(status_summary)

    def _cleanup_old_data(self):
        """Periodically cleanup old data to manage memory."""
        current_time = time.time()
        if current_time - self.last_cleanup_time >= self.cleanup_interval:
            self.last_cleanup_time = current_time
            # Clear old data while keeping the maxlen constraint
            while len(self.detection_history) > self.detection_history.maxlen // 2:
                self.detection_history.popleft()
            while len(self.status_history) > self.status_history.maxlen // 2:
                self.status_history.popleft()

    def detect(self, frame):
        """Detect persons and track their status."""
        # Get raw detections from PersonDetector
        results = self.detector.detect(frame)
        
        # Process each detection to add status
        processed_detections = []
        for det in results:
            bbox = det[:4]  # Get bounding box
            conf = det[4]   # Get confidence
            
            # Find matching previous detection
            prev_bbox = None
            if self.prev_detections:
                min_dist = float('inf')
                for prev in self.prev_detections:
                    px1, py1 = prev[:2]
                    curr_dist = ((bbox[0] - px1) ** 2 + (bbox[1] - py1) ** 2) ** 0.5
                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        prev_bbox = prev
            
            # Determine status
            status = self._determine_status(bbox, prev_bbox)
            
            # Add to processed detections
            processed_detections.append([*bbox, conf, 0, status])
        
        # Update histories
        self._update_history(processed_detections)
        self._cleanup_old_data()
        
        # Update previous detections
        self.prev_detections = [d[:4] for d in processed_detections][-self.history_size:]
        
        return processed_detections

    def get_status_summary(self, time_range=None):
        """Get summary of status history within time range."""
        if not self.status_history:
            return {'no_detection': 1.0}
            
        if time_range is None:
            relevant_history = self.status_history
        else:
            cutoff_time = datetime.now() - time_range
            relevant_history = [
                entry for entry in self.status_history 
                if entry['timestamp'] > cutoff_time
            ]
        
        # Count occurrences of each status
        total_count = len(relevant_history)
        status_counts = {}
        
        for entry in relevant_history:
            status = entry['primary_status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Convert to percentages
        return {
            status: count / total_count 
            for status, count in status_counts.items()
        }

    def get_activity_timeline(self, time_range=None):
        """Get timeline of activity within time range."""
        if time_range is None:
            return list(self.status_history)
            
        cutoff_time = datetime.now() - time_range
        return [
            entry for entry in self.status_history 
            if entry['timestamp'] > cutoff_time
        ]

    def process_frame(self, frame):
        """Process a frame and draw detections with status."""
        try:
            # Run detection
            detections = self.detect(frame)
            
            # Create status text
            status_text = f"{len(detections)} person{'s' if len(detections) != 1 else ''}"
            
            # Draw detections
            frame_copy = frame.copy()
            for x1, y1, x2, y2, conf, cls, status in detections:
                # Draw bounding box
                cv2.rectangle(
                    frame_copy,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )
                
                # Draw label with status
                label = f"Person ({status})"
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame_copy,
                    (int(x1), int(y1) - text_height - 5),
                    (int(x1) + text_width, int(y1)),
                    (0, 255, 0),
                    -1
                )
                cv2.putText(
                    frame_copy,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
            
            # Add detailed status text
            status_details = [det[6] for det in detections]
            if status_details:
                status_text += ": " + ", ".join(status_details)
            
            return frame_copy, status_text
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return frame, "Error"

    def __del__(self):
        """Cleanup resources."""
        try:
            # Clear histories
            self.detection_history.clear()
            self.status_history.clear()
            self.prev_detections.clear()
        except Exception as e:
            self.logger.error(f"Error cleaning up person tracker: {str(e)}") 