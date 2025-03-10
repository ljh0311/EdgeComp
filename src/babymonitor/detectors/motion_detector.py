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
        self.motion_history_size = config.get('motion_history_size', 5)
        self.fall_detection_frames = config.get('fall_detection_frames', 3)
        self.rapid_motion_threshold = config.get('rapid_motion_threshold', 50)
        self.iou_threshold = config.get('iou_threshold', 0.5)  # IOU threshold for tracking
        
        # Initialize state
        self.previous_frame = None
        self.previous_positions = {}
        self.motion_history = collections.deque(maxlen=self.motion_history_size)
        self.position_history = collections.defaultdict(lambda: collections.deque(maxlen=self.fall_detection_frames))
        self.fall_candidates = collections.defaultdict(int)
        
        # Initialize motion detection with optimized parameters
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=config.get('bg_history', 120),
            varThreshold=config.get('var_threshold', 25),
            detectShadows=False
        )
        
        # Pre-allocate buffers
        self.gray_buffer = None
        self.mask_buffer = None
        
        # Track person IDs
        self.next_person_id = 1
        self.person_trackers = {}
        self.active_trackers = set()
        
        # Performance optimization flags
        self.skip_frames = 0
        self.process_every_n = 2
    
    def _calculate_motion_score(self, current_frame: np.ndarray) -> float:
        """Calculate motion score using background subtraction and frame difference."""
        # Apply background subtraction
        fg_mask = self.fgbg.apply(current_frame, learningRate=0.02)
        
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
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        # Calculate areas
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0

    def _match_boxes_to_trackers(self, current_boxes):
        """Match current boxes to existing trackers using IOU."""
        matched_pairs = []
        unmatched_boxes = []
        
        # If no trackers exist, all boxes are unmatched
        if not self.person_trackers:
            return [], list(range(len(current_boxes)))
            
        # Calculate IOU matrix
        iou_matrix = []
        for box in current_boxes:
            ious = []
            for tracker_id, tracker_box in self.person_trackers.items():
                iou = self._calculate_iou(box, tracker_box)
                ious.append((tracker_id, iou))
            iou_matrix.append(ious)
        
        # Match boxes to trackers
        used_trackers = set()
        for box_idx, ious in enumerate(iou_matrix):
            best_iou = self.iou_threshold
            best_tracker = None
            
            for tracker_id, iou in ious:
                if iou > best_iou and tracker_id not in used_trackers:
                    best_iou = iou
                    best_tracker = tracker_id
            
            if best_tracker is not None:
                matched_pairs.append((box_idx, best_tracker))
                used_trackers.add(best_tracker)
            else:
                unmatched_boxes.append(box_idx)
        
        return matched_pairs, unmatched_boxes

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
            boxes: YOLO results object containing detections
            
        Returns:
            tuple: (annotated_frame, is_rapid_motion, is_fall_detected)
        """
        try:
            # Skip frames for performance
            self.skip_frames = (self.skip_frames + 1) % self.process_every_n
            if self.skip_frames != 0:
                return frame, False, False
            
            # Initialize or resize buffers if needed
            if self.gray_buffer is None or self.gray_buffer.shape != (frame.shape[0], frame.shape[1]):
                self.gray_buffer = np.empty((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                self.mask_buffer = np.empty_like(self.gray_buffer)
            
            # Convert frame to grayscale
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self.gray_buffer)
            
            # Calculate motion score
            motion_score = self._calculate_motion_score(self.gray_buffer)
            
            # Detect rapid motion
            rapid_motion = self._detect_rapid_motion(motion_score)
            any_fall_detected = False
            
            # Process detected persons
            current_boxes = []
            if boxes is not None and len(boxes) > 0:
                for result in boxes:
                    if result.boxes is not None:
                        for box in result.boxes:
                            if int(box.cls[0].item()) == 0:  # person class
                                current_boxes.append(tuple(map(int, box.xyxy[0].tolist())))
            
            # Only process person tracking if there are detections
            if current_boxes:
                # Match current boxes with existing trackers
                matched_pairs, unmatched_boxes = self._match_boxes_to_trackers(current_boxes)
                
                # Update trackers and process detections
                new_trackers = {}
                self.active_trackers.clear()
                
                # Process matched pairs
                for box_idx, tracker_id in matched_pairs:
                    current_box = current_boxes[box_idx]
                    new_trackers[tracker_id] = current_box
                    self.active_trackers.add(tracker_id)
                    
                    # Analyze movement
                    previous_box = self.previous_positions.get(tracker_id)
                    analysis = self._analyze_person_movement(tracker_id, current_box, previous_box)
                    
                    # Update state and draw annotations
                    self._update_detection_state(frame, current_box, tracker_id, analysis)
                    rapid_motion = rapid_motion or analysis['rapid_motion']
                    any_fall_detected = any_fall_detected or analysis['fall_detected']
                
                # Process unmatched boxes (new detections)
                for box_idx in unmatched_boxes:
                    current_box = current_boxes[box_idx]
                    tracker_id = self.next_person_id
                    self.next_person_id += 1
                    
                    new_trackers[tracker_id] = current_box
                    self.active_trackers.add(tracker_id)
                    
                    analysis = self._analyze_person_movement(tracker_id, current_box, None)
                    self._update_detection_state(frame, current_box, tracker_id, analysis)
                
                # Update trackers for next frame
                self.person_trackers = new_trackers
            
            # Update previous frame
            np.copyto(self.previous_frame, self.gray_buffer) if self.previous_frame is not None else self.gray_buffer.copy()
            
            return frame, rapid_motion, any_fall_detected
            
        except Exception as e:
            self.logger.error(f"Error in motion and fall detection: {str(e)}")
            return frame, False, False

    def _update_detection_state(self, frame, box, person_id, analysis):
        """Update detection state and draw annotations for a person."""
        x1, y1, x2, y2 = box
        
        # Draw bounding box
        color = (0, 0, 255) if analysis['fall_detected'] else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add labels
        labels = [
            f"Person {person_id}",
            f"Position: {analysis['position']}",
        ]
        
        y_offset = y1 - 10
        for label in labels:
            cv2.putText(frame, label, (x1, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset -= 20
        
        # Store current position for next frame
        self.previous_positions[person_id] = box
        
        # Log events
        if analysis['fall_detected']:
            self.logger.warning(f"Fall detected for person {person_id}")
        if analysis['rapid_motion']:
            self.logger.warning(f"Rapid movement detected for person {person_id}")

    def reset(self):
        """Reset the detector state."""
        self.previous_frame = None
        self.previous_positions.clear()
        self.motion_history.clear()
        self.position_history.clear()
        self.fall_candidates.clear()
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.person_trackers.clear()
        self.active_trackers.clear()
        self.next_person_id = 1 