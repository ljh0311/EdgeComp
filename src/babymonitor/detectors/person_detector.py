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
import torch
from pathlib import Path
import time


class PersonDetector:
    def __init__(self, model_path=None, force_cpu=False, max_retries=3):
        """
        Initialize the person detector.

        Args:
            model_path (str, optional): Path to the YOLO model file. If None, uses default.
            force_cpu (bool, optional): Force CPU usage even if CUDA is available.
            max_retries (int, optional): Maximum number of retries for model initialization.
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.max_retries = max_retries
        
        # Device selection with CPU fallback
        self.device = "cpu"
        if not force_cpu:
            try:
                if torch.cuda.is_available():
                    self.device = "cuda"
                    torch.backends.cudnn.benchmark = True
            except Exception as e:
                self.logger.warning(f"CUDA initialization failed, falling back to CPU: {e}")
                self.device = "cpu"
        
        self.logger.info(f"Using device: {self.device}")

        # Use default model path if none provided
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "yolov8n.pt")
            if not os.path.exists(model_path):
                # Try downloading the model if it doesn't exist
                try:
                    self.logger.info("Model not found, attempting to download...")
                    model_path = "yolov8n.pt"  # This will trigger auto-download
                except Exception as e:
                    self.logger.error(f"Failed to download model: {e}")
                    raise FileNotFoundError("Model file not found and download failed")

        self._initialize_model_with_retry(model_path)

    def _initialize_model_with_retry(self, model_path):
        """Initialize the YOLO model with retry mechanism."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempting to initialize model (attempt {attempt + 1}/{self.max_retries})")
                
                # Initialize model with optimized settings
                self.model = YOLO(model_path)
                self.model.to(self.device)
                
                # Set model parameters for faster inference
                self.model.conf = 0.3
                self.model.iou = 0.3
                self.model.max_det = 4
                
                # Verify model works with a test input
                test_frame = np.zeros((32, 32, 3), dtype=np.uint8)
                _ = self.model(test_frame, verbose=False)
                
                self.logger.info(f"Model initialized successfully on {self.device}")
                return
                
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Model initialization attempt {attempt + 1} failed: {e}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                time.sleep(1)  # Wait before retrying
        
        error_msg = f"Failed to initialize model after {self.max_retries} attempts. Last error: {last_error}"
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)

    def detect(self, frame):
        """
        Detect persons in a frame.

        Args:
            frame (numpy.ndarray): Input frame

        Returns:
            list: YOLO results object containing detections
        """
        if self.model is None:
            self.logger.error("Model not initialized")
            return []
            
        if frame is None or not isinstance(frame, np.ndarray):
            self.logger.warning("Invalid frame input")
            return []

        try:
            # Store original dimensions
            original_h, original_w = frame.shape[:2]
            
            # Scale down frame for faster processing if it's large
            target_size = (640, 640)
            if original_h > 640 or original_w > 640:
                scale = min(640/original_w, 640/original_h)
                new_w = int(original_w * scale)
                new_h = int(original_h * scale)
                processed_frame = cv2.resize(frame, (new_w, new_h))
            else:
                processed_frame = frame
                scale = 1.0

            # Run detection with optimized settings
            results = self.model(processed_frame, verbose=False)
            
            # If we scaled the image, adjust the detection coordinates
            if scale != 1.0:
                for r in results:
                    if r.boxes is not None:
                        # Scale coordinates back to original size
                        r.boxes.xyxy = r.boxes.xyxy / scale
            
            return results

        except Exception as e:
            self.logger.error(f"Error during person detection: {str(e)}")
            if self.device == "cuda":
                torch.cuda.empty_cache()  # Try to recover from CUDA errors
            return []

    def process_frame(self, frame):
        """
        Process a frame and draw detections on it.

        Args:
            frame (numpy.ndarray): Input frame

        Returns:
            numpy.ndarray: Frame with detections drawn
        """
        if frame is None or not isinstance(frame, np.ndarray):
            return frame

        try:
            # Create copy only if we have detections
            detections = self.detect(frame)
            if not detections or len(detections[0].boxes) == 0:
                return frame
            
            frame_copy = frame.copy()
            
            for r in detections:
                boxes = r.boxes
                for box in boxes:
                    if box.cls == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Optimize drawing operations
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        
                        # Draw more visible bounding box
                        # Draw outer black box for contrast
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        # Draw inner green box
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw efficient label with better visibility
                        label = f"Person {conf:.2f}"
                        font_scale = 0.6
                        thickness = 2
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        
                        # Get text size
                        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # Draw label background
                        cv2.rectangle(frame_copy, 
                                    (x1, y1 - text_h - 8),
                                    (x1 + text_w + 4, y1),
                                    (0, 0, 0), -1)  # Black background
                        cv2.rectangle(frame_copy, 
                                    (x1, y1 - text_h - 8),
                                    (x1 + text_w + 4, y1),
                                    (0, 255, 0), 1)  # Green border
                        
                        # Draw text
                        cv2.putText(frame_copy, label,
                                  (x1 + 2, y1 - 5),
                                  font,
                                  font_scale,
                                  (255, 255, 255),  # White text
                                  thickness)

            return frame_copy

        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return frame

    def __del__(self):
        """Cleanup when object is deleted."""
        if hasattr(self, "model") and self.model is not None:
            try:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
