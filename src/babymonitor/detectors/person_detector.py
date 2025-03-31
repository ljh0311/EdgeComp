"""
Person Detector Module
===================
Efficient person detection using YOLOv8 for accurate person detection.
"""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
import time
import torch
from typing import Dict, List, Any, Tuple
from .base_detector import BaseDetector

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()

class PersonDetector(BaseDetector):
    """Person detector using YOLOv8 for accurate and efficient person detection."""

    def __init__(
        self,
        model_path: str = None,
        threshold: float = 0.5,
        target_size: Tuple[int, int] = (640, 480),
        force_cpu: bool = False,
        max_retries: int = 3,
        frame_skip: int = 0,
        process_resolution: Tuple[int, int] = None,
        keep_history: bool = True,
        max_history: int = 100
    ):
        """Initialize the person detector.

        Args:
            model_path: Path to YOLOv8 model file
            threshold: Detection confidence threshold
            target_size: Target frame size for processing
            force_cpu: Force CPU usage even if CUDA is available
            max_retries: Maximum number of retries for model loading
            frame_skip: Number of frames to skip between detections
            process_resolution: Resolution for processing frames
            keep_history: Whether to keep detection history
            max_history: Maximum number of history entries to keep
        """
        super().__init__(threshold=threshold)

        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
        self.force_cpu = force_cpu
        self.max_retries = max_retries
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.process_resolution = process_resolution
        self.keep_history = keep_history
        self.max_history = max_history
        
        # Default model path if none provided
        if model_path is None:
            # Look for model in standard locations
            possible_paths = [
                os.path.join(os.getcwd(), "yolov8n.pt"),
                os.path.join(os.getcwd(), "models", "yolov8n.pt"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "yolov8n.pt")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    self.logger.info(f"Found YOLOv8 model at: {model_path}")
                    break
        
        self.model_path = model_path
        
        # Initialize the YOLOv8 detector
        self._initialize_detector()
        
        # Initialize state
        self.last_detection_time = time.time()
        self.detection_history = []
        self.max_history_size = 10
        
        # Store last result for web interface
        self.last_result = None

    def _initialize_detector(self):
        """Initialize the YOLOv8 model for person detection."""
        self.logger.info("Initializing YOLOv8 detector")
        
        # Import YOLO only when needed to avoid loading at module import time
        try:
            from ultralytics import YOLO
            self.logger.info("Successfully imported YOLO from ultralytics")
        except ImportError as e:
            self.logger.error(f"Failed to import YOLO: {str(e)}")
            self.logger.warning("YOLOv8 not available. Try installing with: pip install ultralytics")
            self.model = None
            return
            
        try:
            # Check if model path exists
            if not self.model_path:
                self.logger.error("Model path is not set")
                self.model = None
                return
                
            if os.path.exists(self.model_path):
                self.logger.info(f"Found model at path: {self.model_path}")
                
                # Try loading model with retries
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    try:
                        self.logger.info(f"Loading YOLOv8 model (attempt {attempt}/{max_retries})")
                        self.model = YOLO(self.model_path)
                        
                        # Validate model loaded correctly by trying a dummy prediction
                        dummy_input = np.zeros((100, 100, 3), dtype=np.uint8)
                        _ = self.model(dummy_input, verbose=False)
                        
                        # Log model information
                        if hasattr(self.model, 'names'):
                            self.logger.info(f"Model loaded successfully with classes: {self.model.names}")
                        else:
                            self.logger.warning("Model loaded but class names not found")
                        
                        break  # Successfully loaded
                    except Exception as e:
                        self.logger.error(f"Error loading model (attempt {attempt}/{max_retries}): {str(e)}")
                        if attempt == max_retries:
                            self.logger.error("Maximum retries reached. Failed to load model.")
                            self.model = None
                        else:
                            time.sleep(1)  # Wait before retrying
            else:
                self.logger.warning(f"Model not found at path: {self.model_path}")
                self.logger.info("Attempting to download pre-trained YOLOv8 model")
                
                try:
                    # Use a pre-trained model as fallback
                    self.model = YOLO("yolov8n.pt")
                    self.logger.info("Downloaded pre-trained YOLOv8n model")
                except Exception as e:
                    self.logger.error(f"Failed to download pre-trained model: {str(e)}")
                    self.model = None
        except Exception as e:
            self.logger.error(f"Error initializing detector: {str(e)}")
            self.model = None
            
        # Final check
        if self.model is None:
            self.logger.warning("Setting up dummy model. Person detection will not work correctly.")
            # Create a dummy model for graceful degradation
            class DummyModel:
                def __init__(self):
                    self.names = {0: "person"}
                    
                def __call__(self, *args, **kwargs):
                    return []  # Return empty results
                    
            self.model = DummyModel()

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a frame to detect persons.

        Args:
            frame: Input frame to process

        Returns:
            Dict with detection results including:
                - person_detected (bool): Whether a person was detected
                - confidence (float): Confidence score of the detection
                - detections (list): List of detection information including bounding boxes
        """
        # Check if frame is valid
        if frame is None or frame.size == 0:
            self.logger.warning("Invalid frame provided to person detector")
            return {
                'person_detected': False,
                'confidence': 0.0,
                'detections': []
            }
            
        # Initialize response
        result = {
            'person_detected': False,
            'confidence': 0.0,
            'detections': [],
            'processing_time': 0.0,
            'detection_type': 'none'
        }

        start_time = time.time()
        
        try:
            # Skip if model not available
            if self.model is None:
                self.logger.warning("Person detection model not available")
                return result
                
            # Run inference
            detections = self.model(frame, verbose=False)
            
            # Process results
            if detections:
                person_boxes = []
                max_confidence = 0.0
                
                for detection in detections:
                    if hasattr(detection, 'boxes') and len(detection.boxes) > 0:
                        self.logger.debug(f"Found {len(detection.boxes)} boxes in detection")
                        
                        # Process each box
                        for i in range(len(detection.boxes)):
                            # Get box data
                            box = detection.boxes[i]
                            
                            # Extract class ID, confidence, and coordinates
                            if hasattr(box, 'cls'):
                                cls_id = int(box.cls[0].item() if hasattr(box.cls[0], 'item') else box.cls[0])
                            else:
                                self.logger.warning("Box missing class information")
                                continue
                                
                            # Check if person class (typically 0 in COCO)
                            if cls_id != 0:  # Not a person
                                continue
                                
                            # Get confidence
                            if hasattr(box, 'conf'):
                                confidence = float(box.conf[0].item() if hasattr(box.conf[0], 'item') else box.conf[0])
                            else:
                                self.logger.warning("Box missing confidence information")
                                continue
                                
                            # Skip low confidence detections
                            if confidence < self.threshold:
                                continue
                                
                            # Get coordinates
                            if hasattr(box, 'xyxy'):
                                # [x1, y1, x2, y2] format (top-left, bottom-right corners)
                                coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                                x1, y1, x2, y2 = map(int, coords)
                            elif hasattr(box, 'xywh'):
                                # [x, y, w, h] format (center x, center y, width, height)
                                coords = box.xywh[0].cpu().numpy() if hasattr(box.xywh[0], 'cpu') else box.xywh[0]
                                x, y, w, h = map(int, coords)
                                x1, y1 = int(x - w/2), int(y - h/2)
                                x2, y2 = int(x + w/2), int(y + h/2)
                            else:
                                self.logger.warning("Box missing coordinate information")
                                continue
                                
                            # Save detection
                            person_boxes.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': cls_id,
                                'class_name': 'person'
                            })
                            
                            # Update max confidence
                            max_confidence = max(max_confidence, confidence)
                
                # Add detections to result
                if person_boxes:
                    result['person_detected'] = True
                    result['confidence'] = max_confidence
                    result['detections'] = person_boxes
                    result['detection_type'] = 'yolov8'
                    
                    # Add to detection history
                    if self.keep_history:
                        self.detection_history.append({
                            'timestamp': time.time(),
                            'confidence': max_confidence
                        })
                        # Trim history if needed
                        if len(self.detection_history) > self.max_history:
                            self.detection_history.pop(0)
        except Exception as e:
            self.logger.error(f"Error detecting persons: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        # Add processing time
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        return result
    
    def _draw_detection(self, frame, bbox, label, color):
        """Draw detection bounding box and label on frame.
        
        Args:
            frame: Frame to draw on
            bbox: Bounding box (x, y, w, h)
            label: Label to display
            color: Color for the bounding box and label
        """
        x, y, w, h = bbox
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Calculate label size for background
        label_size, base_line = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Ensure label is visible (not cut off at top of frame)
        label_y = max(y - 7, label_size[1] + 5)
        bg_y1 = max(y - label_size[1] - 10, 0)
        
        # Draw semi-transparent background for label
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, bg_y1),
            (x + label_size[0], label_y + 7),
            (0, 0, 0),
            -1
        )
        # Apply transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA  # Anti-aliased line for smoother text
        )
    
    def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up Person Detector")
        # No specific cleanup needed for YOLOv8 model
        pass
        
    def reset(self):
        """Reset the detector state."""
        self.logger.info("Resetting Person Detector")
        self.frame_count = 0
        self.last_result = None
        self.detection_history = []
        self.last_detection_time = time.time()

    def check_model(self) -> Dict[str, Any]:
        """Check if the YOLOv8 model is loaded and working properly.
        
        Returns:
            Dict containing model status information
        """
        result = {
            "status": "unknown",
            "model_path": self.model_path,
            "model_loaded": self.model is not None,
            "cuda_available": CUDA_AVAILABLE,
            "using_cuda": CUDA_AVAILABLE and not self.force_cpu,
            "errors": []
        }
        
        # Check if model path exists
        if self.model_path:
            if os.path.exists(self.model_path):
                result["model_exists"] = True
                result["model_size"] = os.path.getsize(self.model_path) / (1024 * 1024)  # Size in MB
            else:
                result["model_exists"] = False
                result["errors"].append(f"Model file not found: {self.model_path}")
        else:
            result["model_exists"] = False
            result["errors"].append("No model path specified")
            
        # Check if model is loaded
        if self.model is not None:
            # Check model classes
            if hasattr(self.model, 'names') and self.model.names:
                result["classes"] = self.model.names
                if 0 in self.model.names and "person" in self.model.names[0].lower():
                    result["has_person_class"] = True
                else:
                    result["has_person_class"] = False
                    result["errors"].append("Model doesn't have 'person' class at index 0")
            else:
                result["has_person_class"] = False
                result["errors"].append("Model doesn't have class names attribute")
            
            # Test model with dummy input
            try:
                dummy_input = np.zeros((100, 100, 3), dtype=np.uint8)
                test_result = self.model(dummy_input, verbose=False)
                result["model_test"] = "success"
                
                # Check if result has expected structure
                if test_result and len(test_result) > 0:
                    if hasattr(test_result[0], 'boxes'):
                        result["result_structure"] = "valid"
                    else:
                        result["result_structure"] = "unexpected"
                        result["errors"].append("Model results don't have expected structure")
                else:
                    result["result_structure"] = "empty"
            except Exception as e:
                result["model_test"] = "failed"
                result["errors"].append(f"Error testing model: {str(e)}")
        else:
            result["model_test"] = "skipped"
            result["errors"].append("Model not loaded")
        
        # Set overall status
        if not result.get("model_exists", False):
            result["status"] = "missing"
        elif not result.get("model_loaded", False):
            result["status"] = "not_loaded"
        elif result.get("model_test") == "failed":
            result["status"] = "not_working"
        elif result.get("has_person_class", False) and result.get("result_structure") == "valid":
            result["status"] = "working"
        else:
            result["status"] = "partial"
            
        # Add performance suggestion
        if CUDA_AVAILABLE and self.force_cpu:
            result["performance_suggestion"] = "CUDA is available but not being used. Consider enabling GPU acceleration."
            
        return result
        
    def reload_model(self) -> Dict[str, Any]:
        """Reload the YOLOv8 model.
        
        Returns:
            Dict containing reload status
        """
        self.logger.info("Attempting to reload YOLOv8 model")
        
        try:
            # Save original state
            original_model = self.model
            
            # Re-initialize detector
            self._initialize_detector()
            
            # Check if model loaded successfully
            if self.model is not None:
                # Test model
                try:
                    dummy_input = np.zeros((100, 100, 3), dtype=np.uint8)
                    _ = self.model(dummy_input, verbose=False)
                    self.logger.info("Model reloaded and tested successfully")
                    return {
                        "status": "success",
                        "message": "Model reloaded and tested successfully"
                    }
                except Exception as e:
                    # Restore original model if test fails
                    self.model = original_model
                    self.logger.error(f"Reloaded model failed testing: {str(e)}")
                    return {
                        "status": "error",
                        "message": f"Reloaded model failed testing: {str(e)}"
                    }
            else:
                # Restore original model
                self.model = original_model
                self.logger.error("Failed to reload model")
                return {
                    "status": "error",
                    "message": "Failed to reload model"
                }
                
        except Exception as e:
            self.logger.error(f"Error reloading model: {str(e)}")
            return {
                "status": "error",
                "message": f"Error reloading model: {str(e)}"
            }
