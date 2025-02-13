"""
Person Detection Module
=====================
Handles person detection, tracking, and position analysis using YOLO.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
import sys

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
        
        # Create a null device to suppress YOLO's output
        class NullDevice:
            def write(self, s): pass
            def flush(self): pass
        
        # Store original stdout
        self.original_stdout = sys.stdout
        # Create null device for suppressing output
        self.null_device = NullDevice()
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the YOLO model with configuration."""
        try:
            # Temporarily redirect stdout to suppress YOLO's output
            sys.stdout = self.null_device
            
            # Initialize model with only the path and device
            self.model = YOLO(self.model_config['path'])
            # Set the device if specified
            if 'device' in self.model_config:
                self.model.to(self.model_config['device'])
            
            # Restore stdout
            sys.stdout = self.original_stdout
            
            self.logger.info("Person detection model initialized successfully")
        except Exception as e:
            # Restore stdout in case of error
            sys.stdout = self.original_stdout
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
            # Temporarily redirect stdout
            sys.stdout = self.null_device
            
            # Run YOLO detection with confidence and IOU thresholds
            results = self.model(
                frame,
                conf=self.model_config['conf'],
                iou=self.model_config['iou'],
                verbose=False  # Disable verbose output
            )
            
            # Restore stdout
            sys.stdout = self.original_stdout
            
            person_count = 0
            detected_objects = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = result.names[class_id]
                    detected_objects.append(f"{label} ({conf:.2f})")
                    
                    # Filter for person class (class 0 in COCO dataset)
                    if class_id == 0:  # person class
                        person_count += 1
                        x1, y1, x2, y2 = box.xyxy[0]  # get box coordinates
                        
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
                        
                        self.logger.debug(f"Person {person_count} detected in {position} position with confidence {conf:.2f}")
            
            if detected_objects:
                self.logger.info(f"Detected objects: {', '.join(detected_objects)}")
            
            # Display total person count
            cv2.putText(frame, f'People in room: {person_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame, person_count, results[0].boxes if results else None
            
        except Exception as e:
            # Restore stdout in case of error
            sys.stdout = self.original_stdout
            self.logger.error(f"Error in person detection: {str(e)}")
            return frame, 0, None
        finally:
            # Always ensure stdout is restored
            sys.stdout = self.original_stdout 