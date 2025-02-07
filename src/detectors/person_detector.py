"""
Person Detection Module
=====================
Handles person detection, tracking, and position analysis using YOLO.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging

class PersonDetector:
    def __init__(self, model_config):
        """
        Initialize the person detector.
        
        Args:
            model_config (dict): Configuration for the YOLO model
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_config = model_config
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the YOLO model with configuration."""
        try:
            # Initialize model with only the path and device
            self.model = YOLO(self.model_config['path'])
            # Set the device if specified
            if 'device' in self.model_config:
                self.model.to(self.model_config['device'])
            
            self.logger.info("Person detection model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize person detection model: {str(e)}")
            raise
    
    def detect(self, frame):
        """
        Detect people in the frame.
        
        Args:
            frame: Input frame to process
            
        Returns:
            tuple: (annotated_frame, person_count, detections)
        """
        try:
            # Run YOLO detection with confidence and IOU thresholds
            results = self.model(
                frame,
                conf=self.model_config['conf'],
                iou=self.model_config['iou']
            )
            
            person_count = 0
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Filter for person class (class 0 in COCO dataset)
                    if box.cls == 0:  # person class
                        person_count += 1
                        x1, y1, x2, y2 = box.xyxy[0]  # get box coordinates
                        conf = box.conf[0]  # confidence score
                        
                        # Convert coordinates to integers
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # Calculate box dimensions
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Determine if person is standing, sitting, or lying down
                        if height > width * 1.5:  # Person is significantly taller than wide
                            position = "standing"
                        elif width > height * 1.5:  # Person is significantly wider than tall
                            position = "lying down"
                        else:  # Dimensions are relatively similar
                            position = "sitting"
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add confidence label with position
                        label = f'Person {person_count} ({position}): {conf:.2f}'
                        cv2.putText(frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        self.logger.info(f"Person {person_count} detected in {position} position with confidence {conf:.2f}")
            
            # Display total person count
            cv2.putText(frame, f'People in room: {person_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame, person_count, results[0].boxes if results else None
            
        except Exception as e:
            self.logger.error(f"Error in person detection: {str(e)}")
            return frame, 0, None 