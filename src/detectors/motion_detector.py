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

class MotionDetector:
    def __init__(self, config):
        """
        Initialize the motion detector.
        
        Args:
            config (dict): Configuration parameters for motion detection
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.motion_threshold = config.get('motion_threshold', 100)
        self.fall_threshold = config.get('fall_threshold', 1.5)
        self.standing_ratio = config.get('standing_ratio', 0.4)
        self.sitting_ratio = config.get('sitting_ratio', 0.8)
        self.motion_history_size = config.get('motion_history_size', 10)
        self.fall_detection_frames = config.get('fall_detection_frames', 5)
        self.rapid_motion_threshold = config.get('rapid_motion_threshold', 50)
        
        # Initialize state
        self.previous_frame = None
        self.previous_positions = {}
        self.motion_history = collections.deque(maxlen=self.motion_history_size)
        self.position_history = collections.defaultdict(lambda: collections.deque(maxlen=self.fall_detection_frames))
        self.fall_candidates = collections.defaultdict(int)
        
        # Initialize motion detection
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=config.get('bg_history', 500),
            varThreshold=config.get('var_threshold', 16),
            detectShadows=config.get('detect_shadows', True)
        )
    
    def _calculate_motion_score(self, current_frame: np.ndarray) -> float:
        """Calculate motion score using background subtraction and frame difference."""
        # Apply background subtraction
        fg_mask = self.fgbg.apply(current_frame)
        
        # Calculate frame difference if previous frame exists
        if self.previous_frame is not None:
            frame_diff = cv2.absdiff(current_frame, self.previous_frame)
            motion_score = (np.mean(fg_mask) + np.mean(frame_diff)) / 2
        else:
            motion_score = np.mean(fg_mask)
        
        return motion_score
    
    def _detect_rapid_motion(self, motion_score: float) -> bool:
        """Detect rapid motion using motion history."""
        self.motion_history.append(motion_score)
        
        if len(self.motion_history) > 2:
            current_motion = motion_score
            avg_previous_motion = np.mean(list(self.motion_history)[:-1])
            return current_motion > avg_previous_motion * 2
        
        return False
    
    def _analyze_person_movement(self, 
                               person_id: int, 
                               current_box: Tuple[int, int, int, int], 
                               previous_box: Optional[Tuple[int, int, int, int]]) -> Dict:
        """Analyze person movement and position changes."""
        x1, y1, x2, y2 = current_box
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        result = {
            'rapid_motion': False,
            'fall_detected': False,
            'position': 'unknown'
        }
        
        # Determine position based on aspect ratio
        aspect_ratio = width / height if height != 0 else float('inf')
        if aspect_ratio < self.standing_ratio:
            position = "standing"
        elif aspect_ratio < self.sitting_ratio:
            position = "sitting"
        else:
            position = "lying"
        
        result['position'] = position
        
        # Store position history
        self.position_history[person_id].append(position)
        
        if previous_box:
            prev_x1, prev_y1, prev_x2, prev_y2 = previous_box
            prev_center_x = (prev_x1 + prev_x2) // 2
            prev_center_y = (prev_y1 + prev_y2) // 2
            
            # Calculate movement speed
            movement = np.sqrt((center_x - prev_center_x)**2 + (center_y - prev_center_y)**2)
            result['rapid_motion'] = movement > self.rapid_motion_threshold
            
            # Analyze position changes for fall detection
            if len(self.position_history[person_id]) >= 2:
                prev_positions = list(self.position_history[person_id])[-2:]
                
                # Check for sudden position change from standing/sitting to lying
                if (prev_positions[0] in ['standing', 'sitting'] and 
                    prev_positions[1] == 'lying' and 
                    movement > self.motion_threshold):
                    self.fall_candidates[person_id] += 1
                    
                    # Confirm fall if we have enough consecutive detections
                    if self.fall_candidates[person_id] >= 2:
                        result['fall_detected'] = True
                else:
                    self.fall_candidates[person_id] = max(0, self.fall_candidates[person_id] - 1)
        
        return result
    
    def detect(self, frame: np.ndarray, boxes) -> Tuple[np.ndarray, bool, bool]:
        """
        Detect motion and falls in the frame.
        
        Args:
            frame: Current frame
            boxes: Detected person boxes from YOLO
            
        Returns:
            tuple: (annotated_frame, is_rapid_motion, is_fall_detected)
        """
        try:
            # Convert frame to grayscale for motion detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate motion score
            motion_score = self._calculate_motion_score(gray_frame)
            
            # Detect rapid motion
            rapid_motion = self._detect_rapid_motion(motion_score)
            any_fall_detected = False
            
            # Process detected persons
            if boxes is not None:
                for i, box in enumerate(boxes):
                    if box.cls == 0:  # person class
                        # Get current box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        current_box = (x1, y1, x2, y2)
                        
                        # Get previous box if available
                        previous_box = self.previous_positions.get(i)
                        
                        # Analyze movement
                        analysis = self._analyze_person_movement(i, current_box, previous_box)
                        
                        # Update rapid motion flag
                        rapid_motion = rapid_motion or analysis['rapid_motion']
                        any_fall_detected = any_fall_detected or analysis['fall_detected']
                        
                        # Draw bounding box and annotations
                        color = (0, 0, 255) if analysis['fall_detected'] else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add labels
                        labels = [
                            f"Person {i+1}",
                            f"Position: {analysis['position']}",
                            f"Conf: {float(box.conf[0]):.2f}"
                        ]
                        
                        y_offset = y1 - 10
                        for label in labels:
                            cv2.putText(frame, label, (x1, y_offset),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            y_offset -= 20
                        
                        # Store current position for next frame
                        self.previous_positions[i] = current_box
                        
                        # Log events
                        if analysis['fall_detected']:
                            self.logger.warning(f"Fall detected for person {i+1}")
                        if analysis['rapid_motion']:
                            self.logger.warning(f"Rapid movement detected for person {i+1}")
            
            # Update previous frame
            self.previous_frame = gray_frame
            
            return frame, rapid_motion, any_fall_detected
            
        except Exception as e:
            self.logger.error(f"Error in motion and fall detection: {str(e)}")
            self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame, False, False
    
    def reset(self):
        """Reset the detector state."""
        self.previous_frame = None
        self.previous_positions.clear()
        self.motion_history.clear()
        self.position_history.clear()
        self.fall_candidates.clear()
        self.fgbg = cv2.createBackgroundSubtractorMOG2() 