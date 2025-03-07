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
import torch


class PersonDetector:
    """Person detector using YOLOv5."""

    def __init__(self, model_path=None, device=None):
        """Initialize the person detector."""
        self.logger = logging.getLogger(__name__)
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self._initialize_model(model_path)
            self.logger.info(f"Person detector initialized on {self.device}")
        except Exception as e:
            self.logger.error(f"Error initializing person detector: {str(e)}")
            raise

    def _initialize_model(self, model_path=None):
        """Initialize the YOLO model."""
        try:
            if model_path:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            else:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            
            # Move model to appropriate device and set inference mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Suppress model output
            self.model.conf = 0.5  # Confidence threshold
            self.model.iou = 0.45  # NMS IoU threshold
            self.model.classes = [0]  # Only detect persons (class 0 in COCO)
            
            if self.device.type == 'cuda':
                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {str(e)}")
            raise

    def detect(self, frame):
        """Detect persons in a frame."""
        try:
            # Skip processing if model not loaded
            if self.model is None:
                return []

            # Convert frame to RGB (YOLO expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to a fixed size for faster processing
            input_size = 640
            orig_shape = frame_rgb.shape[:2]
            frame_resized = cv2.resize(frame_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            
            # Move input to device and perform inference
            with torch.no_grad():  # Disable gradient calculation for inference
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=torch.float16):  # Use FP16 for faster inference
                        results = self.model(frame_resized)
                else:
                    results = self.model(frame_resized)
            
            # Extract detections for persons (class 0)
            detections = []
            if len(results.pred) > 0 and len(results.pred[0]) > 0:
                # Get detections from first image
                pred = results.pred[0]
                
                # Filter for person class (0) and convert to CPU if needed
                scale_x = orig_shape[1] / input_size
                scale_y = orig_shape[0] / input_size
                
                for *xyxy, conf, cls in pred:
                    if int(cls) == 0 and float(conf) > 0.3:  # Person class with confidence threshold
                        # Convert to CPU and scale coordinates back to original image size
                        x1, y1, x2, y2 = [
                            float(coord.cpu() if coord.is_cuda else coord) for coord in xyxy
                        ]
                        # Scale coordinates back to original image size
                        bbox = [
                            x1 * scale_x,
                            y1 * scale_y,
                            x2 * scale_x,
                            y2 * scale_y
                        ]
                        confidence = float(conf.cpu() if conf.is_cuda else conf)
                        detections.append([*bbox, confidence, 0])  # [x1, y1, x2, y2, conf, cls]
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in person detection: {str(e)}")
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
        """Cleanup resources."""
        try:
            # Clear CUDA cache if using GPU
            if hasattr(self, 'device') and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Error cleaning up person detector: {str(e)}")
