"""
Person Detection Module
=====================
Handles person detection, tracking, and position analysis using YOLO.
Supports both CPU and GPU execution.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
import sys
import os
import time
from pathlib import Path
import requests
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as nullcontext


class PersonDetector:
    # Default model URL and name for automatic download
    DEFAULT_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    DEFAULT_MODEL_NAME = "yolov8n.pt"
    
    def __init__(self, model_path=None, max_retries=3, force_cpu=False):
        """
        Initialize the person detector.

        Args:
            model_path (str, optional): Path to the YOLO model file. If None, uses default.
            max_retries (int, optional): Maximum number of retries for model initialization.
            force_cpu (bool, optional): Force CPU usage even if GPU is available.
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Initialize metrics
        self._fps = 0
        self._processing_time = 0
        self._last_frame_time = time.time()
        self._frame_count = 0
        self._start_time = time.time()

        self.model = None
        self.max_retries = max_retries
        self.force_cpu = force_cpu
        self._select_device()

        # Use default model path if none provided
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / self.DEFAULT_MODEL_NAME)
        
        # Ensure model exists or download it
        if not os.path.exists(model_path):
            try:
                model_path = self._download_model(model_path)
            except Exception as e:
                self.logger.error(f"Failed to download model: {e}")
                raise FileNotFoundError("Model file not found and download failed")

        self._initialize_model_with_retry(model_path)

    def _select_device(self):
        """Select the appropriate device (CPU/GPU) based on availability and settings."""
        if self.force_cpu:
            self.device = "cpu"
            self.logger.info("Forced CPU usage")
            return

        try:
            if torch.cuda.is_available():
                self.device = "cuda"
                torch.backends.cudnn.benchmark = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
                self.logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                self.device = "cpu"
                self.logger.info("No GPU detected, using CPU")
        except Exception as e:
            self.device = "cpu"
            self.logger.warning(f"Error detecting GPU, falling back to CPU: {e}")

    def switch_device(self, force_cpu=None):
        """
        Switch between CPU and GPU execution.
        
        Args:
            force_cpu (bool): If True, force CPU usage. If False, try to use GPU.
        
        Returns:
            bool: True if switch was successful, False otherwise
        """
        if force_cpu is not None:
            self.force_cpu = force_cpu
        
        old_device = self.device
        self._select_device()
        
        if old_device != self.device:
            try:
                if self.model is not None:
                    # Clear CUDA cache if switching from GPU
                    if old_device == "cuda":
                        torch.cuda.empty_cache()
                    
                    # Move model to new device
                    self.model.to(self.device)
                    
                    # Optimize for GPU if applicable
                    if self.device == "cuda":
                        self.model.model.half()  # Use FP16 for GPU
                    
                    self.logger.info(f"Successfully switched to {self.device}")
                return True
            except Exception as e:
                self.logger.error(f"Error switching device: {e}")
                self.device = old_device  # Revert to old device
                return False
        return True

    def get_device_info(self):
        """
        Get current device information.
        
        Returns:
            dict: Device information including type, name, and memory if available
        """
        info = {"current_device": self.device}
        
        if self.device == "cuda":
            try:
                info.update({
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB",
                    "gpu_memory_used": f"{torch.cuda.memory_allocated() / (1024**3):.1f}GB"
                })
            except Exception as e:
                self.logger.error(f"Error getting GPU info: {e}")
        
        return info

    def get_fps(self):
        """Get the current FPS."""
        return self._fps

    def get_processing_time(self):
        """Get the current frame processing time in milliseconds."""
        return self._processing_time

    def get_memory_usage(self):
        """Get current memory usage percentage."""
        try:
            import psutil
            return psutil.Process().memory_percent()
        except:
            return 0.0

    def _download_model(self, model_path):
        """Download the YOLO model if it doesn't exist."""
        self.logger.info(f"Model not found at {model_path}, attempting to download...")
        try:
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Download with progress bar
            response = requests.get(self.DEFAULT_MODEL_URL, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            self.logger.info("Downloading YOLOv8n model...")
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            
            with open(model_path, 'wb') as f:
                for data in response.iter_content(chunk_size=1024):
                    progress_bar.update(len(data))
                    f.write(data)
            
            progress_bar.close()
            self.logger.info(f"Model downloaded successfully to {model_path}")
            return model_path

        except Exception as e:
            self.logger.error(f"Error downloading model: {e}")
            raise

    def _initialize_model_with_retry(self, model_path):
        """Initialize the YOLO model with retry mechanism."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempting to initialize model (attempt {attempt + 1}/{self.max_retries})")
                
                # Initialize model with optimized settings
                self.model = YOLO(model_path)
                
                # Configure model parameters before moving to device
                self.model.conf = 0.3  # Lower confidence threshold for better detection
                self.model.iou = 0.3   # Lower IoU threshold
                self.model.max_det = 4  # Limit detections for better performance
                
                # Handle device-specific configurations
                if self.device == "cuda":
                    # Move model to GPU first
                    self.model.to(self.device)
                    
                    # Enable automatic mixed precision
                    self.model.amp = True
                    
                    # Configure for GPU inference
                    self.model.model.float()  # Ensure model is in float32 first
                    torch.backends.cudnn.benchmark = True
                    if torch.cuda.is_available():
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                else:
                    # CPU configuration
                    self.model.to(self.device)
                
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
        if frame is None or not isinstance(frame, np.ndarray):
            self.logger.warning("Invalid frame input")
            return []

        try:
            start_time = time.time()
            
            # Store original dimensions
            original_h, original_w = frame.shape[:2]
            
            # Scale down frame for faster processing if it's large
            target_size = (416, 416)  # Smaller size for Raspberry Pi
            if original_h > 416 or original_w > 416:
                scale = min(416/original_w, 416/original_h)
                new_w = int(original_w * scale)
                new_h = int(original_h * scale)
                processed_frame = cv2.resize(frame, (new_w, new_h))
            else:
                processed_frame = frame
                scale = 1.0

            # Run detection with optimized settings
            with torch.cuda.amp.autocast() if self.device == "cuda" else nullcontext():
                results = self.model(processed_frame, verbose=False)
            
            # If we scaled the image, adjust the detection coordinates
            if scale != 1.0:
                # Create new scaled boxes instead of modifying the original
                for r in results:
                    if r.boxes is not None and len(r.boxes.xyxy) > 0:
                        # Scale coordinates back to original size
                        scaled_boxes = r.boxes.xyxy.clone()
                        scaled_boxes = scaled_boxes / scale
                        # Store scaled coordinates in a custom attribute
                        r.boxes.scaled_xyxy = scaled_boxes
            
            # Update metrics
            end_time = time.time()
            self._processing_time = (end_time - start_time) * 1000  # Convert to ms
            self._frame_count += 1
            
            elapsed = end_time - self._start_time
            if elapsed >= 1.0:  # Update FPS every second
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._start_time = end_time
            
            return results

        except Exception as e:
            self.logger.error(f"Error during person detection: {str(e)}")
            return []

    def process_frame(self, frame):
        """Process a frame and draw detections."""
        if frame is None or not isinstance(frame, np.ndarray):
            return frame

        try:
            # Create copy only if we have detections
            detections = self.detect(frame)
            if not detections or len(detections[0].boxes) == 0:
                return frame

            frame_copy = frame.copy()
            original_h, original_w = frame.shape[:2]
            
            for r in detections:
                boxes = r.boxes
                for i in range(len(boxes)):
                    if boxes.cls[i] == 0:  # Person class
                        # Get coordinates and scale them if needed
                        if hasattr(boxes, 'scaled_xyxy'):
                            box_coords = boxes.scaled_xyxy[i]
                        else:
                            box_coords = boxes.xyxy[i]
                            # Scale coordinates if frame was resized
                            if original_h > 416 or original_w > 416:
                                scale_x = original_w / 416
                                scale_y = original_h / 416
                                box_coords[0] *= scale_x
                                box_coords[1] *= scale_y
                                box_coords[2] *= scale_x
                                box_coords[3] *= scale_y
                        
                        x1, y1, x2, y2 = map(int, box_coords.cpu().numpy())
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        # Draw high-visibility bounding box
                        # Draw thick red outline for better visibility
                        cv2.rectangle(frame_copy, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 4)
                        # Draw yellow inner box
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        
                        # Draw efficient label with better visibility
                        label = f"Person {conf:.2f}"
                        font_scale = 0.8  # Larger font for better visibility
                        thickness = 2
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        
                        # Get text size
                        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # Draw label background (semi-transparent)
                        sub_img = frame_copy[y1-text_h-10:y1, x1:x1+text_w+6]
                        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                        alpha = 0.7
                        frame_copy[y1-text_h-10:y1, x1:x1+text_w+6] = cv2.addWeighted(sub_img, 1-alpha, black_rect, alpha, 0)
                        
                        # Draw white text for maximum visibility
                        cv2.putText(frame_copy, label,
                                  (x1+2, y1-8),
                                  font,
                                  font_scale,
                                  (255, 255, 255),
                                  thickness)

            return frame_copy

        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return frame

    def __del__(self):
        """Cleanup when object is deleted."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if hasattr(self, 'device') and self.device == "cuda":
                torch.cuda.empty_cache()
