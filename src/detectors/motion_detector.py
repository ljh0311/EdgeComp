"""
Motion Detection Module
=====================
Handles motion detection and fall detection using frame analysis and aspect ratios.
"""

import cv2
import numpy as np
import collections
import logging

class MotionDetector:
    def __init__(self, config):
        """
        Initialize the motion detector.
        
        Args:
            config (dict): Configuration parameters for motion detection
        """
        self.logger = logging.getLogger(__name__)
        self.motion_threshold = config.get('motion_threshold', 100)
        self.fall_threshold = config.get('fall_threshold', 1.5)
        self.previous_frame = None
        self.previous_positions = {}
        self.motion_history = collections.deque(maxlen=10)
    
    def detect(self, frame, boxes):
        """
        Detect motion and falls in the frame.
        
        Args:
            frame: Current frame
            boxes: Detected person boxes from YOLO
            
        Returns:
            tuple: (annotated_frame, is_rapid_motion, is_fall_detected)
        """
        current_positions = {}
        rapid_motion = False
        fall_detected = False
        
        try:
            # Motion detection using frame difference
            if self.previous_frame is not None:
                # Ensure frames are the same size
                if self.previous_frame.shape != frame.shape:
                    self.previous_frame = cv2.resize(self.previous_frame, (frame.shape[1], frame.shape[0]))
                
                # Convert frames to grayscale for more efficient processing
                current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate absolute difference between frames
                frame_diff = cv2.absdiff(current_gray, prev_gray)
                motion_score = np.mean(frame_diff)
                self.motion_history.append(motion_score)
                
                # Detect sudden motion spikes
                if len(self.motion_history) > 2:
                    current_motion = self.motion_history[-1]
                    avg_previous_motion = np.mean(list(self.motion_history)[:-1])
                    if current_motion > avg_previous_motion * 3:  # Adjustable sensitivity
                        rapid_motion = True
                        cv2.putText(frame, "RAPID MOTION DETECTED!", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.logger.warning("Rapid motion detected")
            
            # Process each detected person for fall detection
            if boxes is not None:
                for i, box in enumerate(boxes):
                    if box.cls == 0:  # person class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Calculate position metrics
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        height = y2 - y1
                        width = x2 - x1
                        
                        current_positions[i] = (center_x, center_y, width, height)
                        
                        # Fall detection using aspect ratio changes
                        if i in self.previous_positions:
                            prev_x, prev_y, prev_width, prev_height = self.previous_positions[i]
                            
                            # Calculate and compare aspect ratios
                            current_ratio = width / height if height != 0 else 0
                            previous_ratio = prev_width / prev_height if prev_height != 0 else 0
                            
                            if previous_ratio != 0:
                                # Detect significant changes in aspect ratio
                                if (current_ratio > previous_ratio * self.fall_threshold or
                                    current_ratio < previous_ratio / self.fall_threshold):
                                    fall_detected = True
                                    cv2.putText(frame, "FALL DETECTED!", (10, 90),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    self.logger.warning(f"Potential fall detected for person {i+1}")
                            
                            # Calculate movement speed
                            movement = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                            if movement > self.motion_threshold:
                                rapid_motion = True
                                cv2.putText(frame, f"RAPID MOVEMENT - Person {i+1}!", 
                                          (10, 120 + i*30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                self.logger.warning(f"Rapid movement detected for person {i+1}")
            
            # Update tracking data
            self.previous_positions = current_positions
            self.previous_frame = frame.copy()
            
            return frame, rapid_motion, fall_detected
            
        except Exception as e:
            self.logger.error(f"Error in motion and fall detection: {str(e)}")
            # Reset previous frame on error to prevent cascading issues
            self.previous_frame = frame.copy()
            return frame, False, False
    
    def reset(self):
        """Reset the detector state."""
        self.previous_frame = None
        self.previous_positions.clear()
        self.motion_history.clear() 