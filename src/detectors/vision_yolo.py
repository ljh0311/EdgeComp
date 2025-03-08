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
    """Person detector using YOLOv8."""

    def __init__(self, model_path=None, device=None):
        """Initialize the person detector."""
        self.logger = logging.getLogger(__name__)
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Track previous detections for motion analysis
        self.prev_detections = []
        self.detection_history = []
        self.history_size = 5
        
        # Detection smoothing
        self.smooth_factor = 0.7  # Higher value means more smoothing
        self.min_detection_confidence = 0.3
        
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
                self.model = YOLO(model_path)
            else:
                # Use YOLOv8n by default
                self.model = YOLO('yolov8n.pt')
            
            # Move model to appropriate device
            self.model.to(self.device)
            
            # Set model parameters
            self.model.conf = self.min_detection_confidence  # Confidence threshold
            self.model.iou = 0.45  # NMS IoU threshold
            self.model.classes = [0]  # Only detect persons (class 0 in COCO)
            self.model.max_det = 10  # Maximum detections per image
            
            if self.device.type == 'cuda':
                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {str(e)}")
            raise

    def _smooth_detections(self, current_detections):
        """Smooth detection boxes using previous detections."""
        if not self.prev_detections:
            return current_detections

        smoothed_detections = []
        for curr_det in current_detections:
            # Find matching previous detection
            matched = False
            curr_box = curr_det[:4]
            for prev_det in self.prev_detections:
                prev_box = prev_det[:4]
                # Calculate IoU between current and previous detection
                intersection_x1 = max(curr_box[0], prev_box[0])
                intersection_y1 = max(curr_box[1], prev_box[1])
                intersection_x2 = min(curr_box[2], prev_box[2])
                intersection_y2 = min(curr_box[3], prev_box[3])
                
                if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                    # Boxes overlap, smooth the coordinates
                    smoothed_box = [
                        prev_box[i] * self.smooth_factor + curr_box[i] * (1 - self.smooth_factor)
                        for i in range(4)
                    ]
                    smoothed_det = [*smoothed_box, curr_det[4], curr_det[5], curr_det[6]]
                    smoothed_detections.append(smoothed_det)
                    matched = True
                    break
            
            if not matched:
                # No match found, use current detection as is
                smoothed_detections.append(curr_det)

        return smoothed_detections

    def _determine_status(self, bbox, prev_bbox=None):
        """Determine person's status based on position and movement."""
        x1, y1, x2, y2 = bbox[:4]
        height = y2 - y1
        width = x2 - x1
        aspect_ratio = width / height if height > 0 else 0

        # Determine if lying down based on aspect ratio
        if aspect_ratio > 1.5:
            return "lying"
        
        # If we have previous detection, check for movement
        if prev_bbox is not None:
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox[:4]
            movement = abs(x1 - prev_x1) + abs(y1 - prev_y1)
            if movement > 20:  # Threshold for movement detection
                return "moving"
        
        return "seated"

    def detect(self, frame):
        """Detect persons in a frame."""
        try:
            # Skip processing if model not loaded
            if self.model is None:
                return []

            # Convert frame to RGB (YOLO expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to a fixed size for faster processing
            input_size = 416  # Reduced from 640 for better performance
            orig_shape = frame_rgb.shape[:2]
            frame_resized = cv2.resize(frame_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            
            # Perform inference
            with torch.no_grad():  # Disable gradient calculation for inference
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=torch.float16):  # Use FP16 for faster inference
                        results = self.model(frame_resized)
                else:
                    results = self.model(frame_resized)
            
            # Extract detections for persons (class 0)
            detections = []
            if len(results) > 0:
                # Get detections from first image
                result = results[0]
                
                # Scale coordinates back to original image size
                scale_x = orig_shape[1] / input_size
                scale_y = orig_shape[0] / input_size
                
                # Process each detection
                for box in result.boxes:
                    if box.cls == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        
                        # Scale coordinates back to original image size
                        bbox = [
                            x1 * scale_x,
                            y1 * scale_y,
                            x2 * scale_x,
                            y2 * scale_y
                        ]
                        
                        # Find matching previous detection for status
                        prev_bbox = None
                        if self.prev_detections:
                            # Find closest previous detection
                            min_dist = float('inf')
                            for prev in self.prev_detections:
                                px1, py1 = prev[:2]
                                curr_dist = ((x1 * scale_x - px1) ** 2 + (y1 * scale_y - py1) ** 2) ** 0.5
                                if curr_dist < min_dist:
                                    min_dist = curr_dist
                                    prev_bbox = prev
                        
                        # Determine status
                        status = self._determine_status(bbox, prev_bbox)
                        
                        detections.append([*bbox, confidence, 0, status])  # [x1, y1, x2, y2, conf, cls, status]
            
            # Smooth detections
            smoothed_detections = self._smooth_detections(detections)
            
            # Update previous detections
            self.prev_detections = [d[:4] for d in smoothed_detections]
            
            # Update detection history
            self.detection_history.append(len(smoothed_detections))
            if len(self.detection_history) > self.history_size:
                self.detection_history.pop(0)
            
            return smoothed_detections
            
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
            str: Status text in format "<person number> <status>"
        """
        if self.model is None:
            return frame, "0 persons"

        try:
            # Run detection
            detections = self.detect(frame)
            
            # Create status text
            status_text = f"{len(detections)} person{'s' if len(detections) != 1 else ''}"
            
            # Draw detections
            frame_copy = frame.copy()
            for x1, y1, x2, y2, conf, cls, status in detections:
                # Draw bounding box with thicker lines
                cv2.rectangle(
                    frame_copy,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),  # Green color
                    3  # Thicker line
                )
                
                # Draw filled background for text
                label = f"Person ({status})"
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    frame_copy,
                    (int(x1), int(y1) - text_height - 10),
                    (int(x1) + text_width, int(y1)),
                    (0, 255, 0),
                    -1  # Filled rectangle
                )
                
                # Draw label with status
                cv2.putText(
                    frame_copy,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,  # Larger font
                    (0, 0, 0),  # Black text
                    2  # Thicker text
                )
            
            # Add detailed status text
            status_details = []
            for det in detections:
                status_details.append(det[6])  # Get status
            if status_details:
                status_text += ": " + ", ".join(status_details)
                
            return frame_copy, status_text
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return frame, "Error"

    def __del__(self):
        """Cleanup resources."""
        try:
            # Clear CUDA cache if using GPU
            if hasattr(self, 'device') and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Error cleaning up person detector: {str(e)}")
