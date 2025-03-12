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
from torch.cuda.amp import autocast


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
        self._warmup_done = False

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
        
        # Warmup the model
        self._warmup_model()

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
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
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

    def _warmup_model(self):
        """Warm up the model with a few inference passes."""
        if self._warmup_done:
            return
            
        self.logger.info("Warming up model...")
        dummy_input = torch.zeros((1, 3, 640, 640)).to(self.device)
        if self.device == "cuda":
            dummy_input = dummy_input.half()
            
        try:
            with torch.no_grad():
                for _ in range(3):  # 3 warmup iterations
                    if self.device == "cuda":
                        with autocast():
                            _ = self.model(dummy_input)
                    else:
                        _ = self.model(dummy_input)
            self._warmup_done = True
            torch.cuda.synchronize() if self.device == "cuda" else None
        except Exception as e:
            self.logger.warning(f"Warmup failed: {e}")

    def _initialize_model_with_retry(self, model_path):
        """Initialize the YOLO model with retry mechanism."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempting to initialize model (attempt {attempt + 1}/{self.max_retries})")
                
                # Initialize model with optimized settings
                self.model = YOLO(model_path)
                
                # Optimize model parameters
                self.model.conf = 0.3
                self.model.iou = 0.3
                self.model.max_det = 4
                self.model.agnostic_nms = True  # Class-agnostic NMS
                
                # Move model to device and optimize
                self.model.to(self.device)
                if self.device == "cuda":
                    self.model.model.half()  # FP16 for GPU
                    self.model.amp = True
                    torch.backends.cudnn.benchmark = True
                
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
            list: List of detections, each containing [x1, y1, x2, y2, confidence, class_id]
        """
        if frame is None or not isinstance(frame, np.ndarray):
            self.logger.warning("Invalid frame input")
            return []

        try:
            start_time = time.time()
            
            # Store original dimensions
            original_h, original_w = frame.shape[:2]
            
            # Scale down frame for faster processing if it's large
            target_size = (416, 416)  # Smaller size for better performance
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
            
            # Process results into a standardized format
            detections = []
            if len(results) > 0:  # Check if we have any results
                result = results[0]  # Get first result
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes = result.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        # Only include person detections (class 0)
                        if cls_id == 0 and conf > 0.3:  # Confidence threshold
                            # Scale coordinates back to original size if needed
                            if scale != 1.0:
                                x1, y1, x2, y2 = [coord / scale for coord in [x1, y1, x2, y2]]
                            
                            detections.append([x1, y1, x2, y2, conf, cls_id])
            
            # Update metrics
            end_time = time.time()
            self._processing_time = (end_time - start_time) * 1000  # Convert to ms
            self._frame_count += 1
            
            elapsed = end_time - self._start_time
            if elapsed >= 1.0:  # Update FPS every second
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._start_time = end_time
            
            return detections

        except Exception as e:
            self.logger.error(f"Error during person detection: {str(e)}")
            return []

    def process_frame(self, frame):
        """
        Process a frame and return detections without drawing.

        Args:
            frame (numpy.ndarray): Input frame

        Returns:
            tuple: (frame, detections) where detections is a list of [x1, y1, x2, y2, confidence, class_id]
        """
        if frame is None or not isinstance(frame, np.ndarray):
            return frame, []

        try:
            detections = self.detect(frame)
            return frame, detections
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return frame, []

    def __del__(self):
        """Cleanup when object is deleted."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if hasattr(self, 'device') and self.device == "cuda":
                torch.cuda.empty_cache()
