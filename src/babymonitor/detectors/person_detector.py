"""
Person Detection Module
===================
Handles person detection using YOLOv8.
"""

import os
import time
import logging
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional
from .base_detector import BaseDetector

# Configure logging
logger = logging.getLogger(__name__)

class PersonDetector(BaseDetector):
    """Handles person detection and motion analysis."""
    
    def __init__(self, 
                 model_path: str = "models/yolov8n.pt",
                 threshold: float = 0.5,
                 force_cpu: Optional[bool] = None):
        """Initialize the person detector.
        
        Args:
            model_path: Path to the YOLOv8 model
            threshold: Detection confidence threshold
            force_cpu: Force CPU usage even if GPU is available
        """
        super().__init__(threshold=threshold)
        
        self.model_path = self._resolve_model_path(model_path)
        self.device = self._detect_device(force_cpu)
        self.model = None
        self.motion_threshold = 0.05
        self.falling_threshold = 0.15
        self.lying_ratio_threshold = 1.5
        
        # Performance optimization
        self.frame_skip = 1
        self.process_resolution = (320, 320)  # Lower resolution for processing
        self.original_resolution = None
        self.processing_scale = 1.0
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"Person detector initialized successfully")
    
    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve the model path.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Resolved model path
        """
        # Check if path is absolute
        if os.path.isabs(model_path):
            if os.path.exists(model_path):
                return model_path
        
        # Check relative to current directory
        if os.path.exists(model_path):
            return model_path
        
        # Check relative to src directory
        src_path = os.path.join("src", model_path)
        if os.path.exists(src_path):
            return src_path
        
        # Check relative to project root
        root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), model_path)
        if os.path.exists(root_path):
            return root_path
        
        # Default to the provided path and let the model loader handle it
        logger.warning(f"Model path {model_path} not found, using as is")
        return model_path
    
    def _detect_device(self, force_cpu: Optional[bool]) -> str:
        """Detect the device to use for inference.
        
        Args:
            force_cpu: Force CPU usage even if GPU is available
            
        Returns:
            Device string
        """
        if force_cpu:
            logger.info("Forcing CPU usage")
            return "cpu"
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
            logger.info(f"Using GPU: {device_name} ({memory:.1f}GB)")
            return "cuda:0"
        else:
            logger.info("No GPU detected, using CPU")
            return "cpu"
    
    def _initialize_model(self):
        """Initialize the YOLOv8 model."""
        try:
            # Import here to avoid dependency if not used
            from ultralytics import YOLO
            
            # Try to load the model with retries
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Attempting to initialize model (attempt {attempt}/{max_retries})")
                    self.model = YOLO(self.model_path)
                    
                    # Set model parameters
                    self.model.to(self.device)
                    logger.info("Model initialized")
                    break
                except Exception as e:
                    logger.error(f"Error initializing model (attempt {attempt}/{max_retries}): {str(e)}")
                    if attempt == max_retries:
                        raise
                    time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Dict containing processed frame and detection results
        """
        start_time = time.time()
        
        # Skip frames for performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0 and self.last_frame is not None:
            return {
                'frame': self.last_frame,
                'detections': self.last_detections or [],
                'fps': self.fps
            }
        
        # Store original frame dimensions
        if self.original_resolution is None:
            self.original_resolution = (frame.shape[1], frame.shape[0])
            self.processing_scale = min(
                self.process_resolution[0] / self.original_resolution[0],
                self.process_resolution[1] / self.original_resolution[1]
            )
        
        # Create a working copy of the frame
        display_frame = frame.copy()
        
        # Resize frame for processing
        if self.processing_scale < 1.0:
            process_width = int(frame.shape[1] * self.processing_scale)
            process_height = int(frame.shape[0] * self.processing_scale)
            process_frame = cv2.resize(frame, (process_width, process_height))
        else:
            process_frame = frame
        
        # Run detection with optimal settings
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device != "cpu"):
            results = self.model(process_frame, conf=self.threshold, iou=0.45, classes=[0])  # Only detect persons (class 0)
            
        # Process detections
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates in the processing frame
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                # Scale coordinates back to original frame
                if self.processing_scale < 1.0:
                    scale = 1.0 / self.processing_scale
                    x1 *= scale
                    y1 *= scale
                    x2 *= scale
                    y2 *= scale
                
                # Get motion status
                status = self._analyze_motion(frame, (x1, y1, x2, y2))
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'status': status
                })
                
                # Draw bounding box and status
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{status} ({confidence:.2f})", 
                          (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 0), 2)
        
        # Update frame history
        self.last_frame = display_frame
        self.last_detections = detections
        
        # Calculate FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        # Adjust frame skip based on performance
        self._adjust_frame_skip()
        
        return {
            'frame': display_frame,
            'detections': detections,
            'fps': self.fps
        }
    
    def _adjust_frame_skip(self):
        """Adjust frame skip based on performance."""
        avg_time = self.get_processing_time()
        if avg_time > 0:
            target_fps = 15  # Target FPS
            ideal_skip = max(1, int(avg_time * target_fps))
            
            # Gradually adjust frame skip
            if ideal_skip > self.frame_skip:
                self.frame_skip = min(ideal_skip, self.frame_skip + 1)
            elif ideal_skip < self.frame_skip and self.frame_skip > 1:
                self.frame_skip = max(1, self.frame_skip - 1)
    
    def _analyze_motion(self, current_frame: np.ndarray, bbox: Tuple) -> str:
        """Analyze motion for a detected person.
        
        Args:
            current_frame: Current frame
            bbox: Bounding box coordinates
            
        Returns:
            Motion status string
        """
        if self.last_frame is None or self.last_detections is None:
            return "standing"
            
        try:
            x1, y1, x2, y2 = map(int, bbox)  # Convert to integers
            h, w = current_frame.shape[:2]
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Extract ROIs
            current_roi = current_frame[y1:y2, x1:x2]
            last_roi = self.last_frame[y1:y2, x1:x2]
            
            if current_roi.size == 0 or last_roi.size == 0:
                return "standing"
                
            # Calculate motion score
            diff = cv2.absdiff(current_roi, last_roi)
            motion_score = np.mean(diff) / 255.0
            
            # Determine status based on motion
            if motion_score > self.motion_threshold:
                # Check for falling motion
                if self._is_falling_motion(current_roi, last_roi):
                    return "falling"
                return "moving"
            else:
                # Check for lying position
                if self._is_lying_position(current_roi):
                    return "lying"
                return "standing"
                
        except Exception as e:
            logger.error(f"Error in motion analysis: {e}")
            return "standing"
    
    def _is_falling_motion(self, current_roi: np.ndarray, last_roi: np.ndarray) -> bool:
        """Check if the motion indicates falling.
        
        Args:
            current_roi: Current ROI
            last_roi: Last ROI
            
        Returns:
            True if falling motion detected
        """
        try:
            # Simple implementation - check for rapid vertical movement
            h_current, w_current = current_roi.shape[:2]
            h_last, w_last = last_roi.shape[:2]
            
            # Check for significant height change
            if h_current > 0 and h_last > 0:
                height_ratio = h_current / h_last
                return height_ratio < 0.8 or height_ratio > 1.2
            
            return False
        except Exception as e:
            logger.error(f"Error in falling motion detection: {e}")
            return False
    
    def _is_lying_position(self, roi: np.ndarray) -> bool:
        """Check if the person is in a lying position.
        
        Args:
            roi: Region of interest
            
        Returns:
            True if lying position detected
        """
        try:
            h, w = roi.shape[:2]
            
            # Check if width is significantly larger than height
            return w > 0 and h > 0 and w / h > self.lying_ratio_threshold
        except Exception as e:
            logger.error(f"Error in lying position detection: {e}")
            return False
    
    def switch_device(self, force_cpu: bool = False) -> bool:
        """Switch between CPU and GPU.
        
        Args:
            force_cpu: Force CPU usage even if GPU is available
            
        Returns:
            True if switch successful
        """
        new_device = "cpu" if force_cpu else ("cuda:0" if torch.cuda.is_available() else "cpu")
        if new_device == self.device:
            return True
            
        try:
            self.device = new_device
            if self.model:
                self.model.to(self.device)
            return True
        except Exception as e:
            logger.error(f"Error switching device: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear CUDA cache
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Remove model reference
            self.model = None
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")
