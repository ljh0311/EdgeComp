"""
Person Detector Module
===================
Efficient person detection using YOLOv8 for accurate person detection and state tracking.
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
        state_history_size: int = 10  # Number of frames to keep for state detection
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
            state_history_size: Number of frames to keep for state detection
        """
        super().__init__(threshold=threshold)

        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
        self.force_cpu = force_cpu
        self.max_retries = max_retries
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.process_resolution = process_resolution
        
        # State tracking
        self.state_history_size = state_history_size
        self.position_history = []
        self.last_positions = []
        self.current_state = 'unknown'
        self.state_confidence = 0.0
        self.movement_threshold = 20  # Pixels
        self.lying_aspect_ratio_threshold = 2.0  # Width/Height ratio for lying detection
        
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
        """Initialize the YOLOv8 detector."""
        try:
            # Import YOLO here to avoid loading at module import time
            from ultralytics import YOLO
            
            # Retry mechanism for model loading
            retries = 0
            while retries < self.max_retries:
                try:
                    # Determine device
                    device = 'cpu'
                    if CUDA_AVAILABLE and not self.force_cpu:
                        device = 0  # Use first CUDA device
                        self.logger.info("CUDA is available, using GPU for inference")
                    else:
                        self.logger.info("Using CPU for inference")
                    
                    # Load the YOLOv8 model
                    if self.model_path and os.path.exists(self.model_path):
                        self.model = YOLO(self.model_path)
                        self.model.to(device)
                        self.logger.info(f"YOLOv8 model loaded from {self.model_path}")
                    else:
                        # Fall back to downloading the model if not found
                        self.logger.warning(f"Model not found at {self.model_path}, loading from Ultralytics")
                        self.model = YOLO('yolov8n.pt')
                        self.model.to(device)
                    
                    # Set class names for filtering
                    self.class_names = self.model.names
                    self.person_class_id = 0  # In COCO dataset, 0 is person class
                    
                    self.logger.info("YOLOv8 detector initialized successfully")
                    return
                
                except Exception as e:
                    retries += 1
                    self.logger.error(f"Error loading model (attempt {retries}/{self.max_retries}): {str(e)}")
                    time.sleep(1)  # Wait before retrying
            
            raise RuntimeError(f"Failed to load YOLOv8 model after {self.max_retries} attempts")
                
        except ImportError as e:
            self.logger.error(f"Error importing YOLO: {str(e)}. Please install the ultralytics package.")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing detector: {str(e)}")
            raise

    def _analyze_state(self, detection: Dict) -> Tuple[str, float]:
        """Analyze the state of a detected person based on their position and movement.
        
        Args:
            detection: Dictionary containing detection information
            
        Returns:
            Tuple of (state, confidence)
        """
        x, y, w, h = detection['box']
        
        # Calculate aspect ratio for lying detection
        aspect_ratio = w / h if h > 0 else 0
        
        # Add current position to history
        self.last_positions.append((x + w/2, y + h/2))
        if len(self.last_positions) > self.state_history_size:
            self.last_positions.pop(0)
            
        # Calculate movement
        movement = 0
        if len(self.last_positions) > 1:
            for i in range(1, len(self.last_positions)):
                prev_x, prev_y = self.last_positions[i-1]
                curr_x, curr_y = self.last_positions[i]
                movement += ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
                
        avg_movement = movement / len(self.last_positions) if self.last_positions else 0
        
        # Determine state
        if aspect_ratio > self.lying_aspect_ratio_threshold:
            state = 'lying'
            confidence = min(1.0, aspect_ratio / self.lying_aspect_ratio_threshold)
        elif avg_movement > self.movement_threshold:
            state = 'moving'
            confidence = min(1.0, avg_movement / self.movement_threshold)
        else:
            state = 'seated'
            confidence = 1.0 - (avg_movement / self.movement_threshold)
            
        return state, confidence

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None:
            return {"frame": frame, "detections": [], "fps": self.fps}

        start_time = time.time()
        frame_with_detections = frame.copy()
        detections = []

        try:
            # Implement frame skipping for better performance
            self.frame_count += 1
            
            if self.frame_skip > 0 and self.frame_count % (self.frame_skip + 1) != 0:
                if self.last_result is not None:
                    self.last_result["frame"] = frame_with_detections
                    return self.last_result
            
            # Resize frame if process_resolution is specified
            if self.process_resolution:
                input_frame = cv2.resize(frame, self.process_resolution, interpolation=cv2.INTER_AREA)
            else:
                input_frame = frame
            
            # Run YOLOv8 inference
            results = self.model(input_frame, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes.data.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2, conf, class_id = box
                    
                    if int(class_id) == self.person_class_id and conf >= self.threshold:
                        # Create detection with box coordinates
                        detection = {
                            "box": (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                            "confidence": float(conf),
                            "class": "person"
                        }
                        
                        # Analyze state
                        state, state_conf = self._analyze_state(detection)
                        detection["state"] = state
                        detection["state_confidence"] = state_conf
                        
                        detections.append(detection)
                        
                        # Draw the detection with state information
                        self._draw_detection(
                            frame_with_detections,
                            detection["box"],
                            f"Person ({state}): {conf:.2f}",
                            (0, 255, 0)  # Green color for person
                        )
            
            # Calculate FPS
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if elapsed_time > 0:
                current_fps = 1.0 / elapsed_time
                self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps
            
            # Draw FPS on frame
            cv2.putText(
                frame_with_detections,
                f"FPS: {self.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Store the result
            self.last_result = {
                "frame": frame_with_detections,
                "detections": detections,
                "fps": self.fps,
                "timestamp": time.time()
            }
            
            return self.last_result
            
        except Exception as e:
            self.logger.error(f"Error in process_frame: {str(e)}")
            return {
                "frame": frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                "detections": [],
                "fps": self.fps,
                "timestamp": time.time()
            }
    
    def _draw_detection(self, frame, bbox, label, color):
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
