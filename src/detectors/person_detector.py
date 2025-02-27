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
import os
from pathlib import Path


class PersonDetector:
    def __init__(self, model_path=None):
        """
        Initialize the person detector.

        Args:
            model_path (str, optional): Path to the YOLO model file. If None, uses default.
        """
        self.logger = logging.getLogger(__name__)
        self.model = None

        # Create a null device to suppress YOLO's output
        class NullDevice:
            def write(self, s):
                pass

            def flush(self):
                pass

        # Store original stdout
        self.original_stdout = sys.stdout
        # Create null device for suppressing output
        self.null_device = NullDevice()

        # Use default model path if none provided
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "yolov8n.pt")

        self._initialize_model(model_path)

    def _initialize_model(self, model_path):
        """Initialize the YOLO model."""
        try:
            # Temporarily redirect stdout to suppress YOLO's output
            sys.stdout = self.null_device

            # Initialize model
            self.model = YOLO(model_path)
            self.model.to('cpu')  # Use CPU by default

            # Restore stdout
            sys.stdout = self.original_stdout
            self.logger.info("Person detection model initialized successfully")

        except Exception as e:
            # Restore stdout
            sys.stdout = self.original_stdout
            self.logger.error(f"Error initializing person detection model: {str(e)}")
            raise

    def detect(self, frame):
        """
        Detect persons in a frame.

        Args:
            frame (numpy.ndarray): Input frame

        Returns:
            list: List of detections [x1, y1, x2, y2, conf, cls]
        """
        if self.model is None:
            return []

        try:
            # Run detection
            results = self.model(frame, verbose=False)
            
            # Filter for person class (class 0 in COCO dataset)
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if box.cls == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        detections.append([x1, y1, x2, y2, conf, cls])

            return detections

        except Exception as e:
            self.logger.error(f"Error during person detection: {str(e)}")
            return []

    def process_frame(self, frame):
        """
        Process a frame and draw detections on it.

        Args:
            frame (numpy.ndarray): Input frame

        Returns:
            numpy.ndarray: Frame with detections drawn
        """
        if self.model is None:
            return frame

        try:
            # Run detection
            detections = self.detect(frame)
            
            # Draw detections
            frame_copy = frame.copy()
            for x1, y1, x2, y2, conf, cls in detections:
                # Draw bounding box
                cv2.rectangle(
                    frame_copy,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),  # Green color
                    2
                )
                
                # Draw label
                label = f"Person {conf:.2f}"
                cv2.putText(
                    frame_copy,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
            return frame_copy
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return frame

    def __del__(self):
        """Cleanup when object is deleted."""
        # Restore stdout
        if hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout
