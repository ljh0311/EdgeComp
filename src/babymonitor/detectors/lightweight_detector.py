"""
Lightweight Detector Module
--------------------------
Implements a lightweight TensorFlow Lite-based detection system for efficient object detection.
This module provides efficient object detection with minimal resource usage, optimized for CPU.
"""

import os
import sys
import time
import cv2
import numpy as np
from threading import Thread
import importlib.util
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import psutil
from .base_detector import BaseDetector

# Configure logging
logger = logging.getLogger(__name__)

class VideoStream:
    """Threaded camera stream optimized for resource-constrained devices."""
    
    def __init__(self, resolution=(640, 480), framerate=30, camera_index=0, buffer_size=2):
        """Initialize the video stream.
        
        Args:
            resolution: Tuple of (width, height) for the camera resolution
            framerate: Target framerate for the camera
            camera_index: Index of the camera to use (default: 0 for first camera)
            buffer_size: Size of the frame buffer (smaller values reduce latency)
        """
        self.stream = cv2.VideoCapture(camera_index)
        
        # Optimize for performance
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, framerate)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        
        # Try to set additional performance parameters
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.frame_count = 0
        self.fps_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.skip_frames = 0  # Number of frames to skip between processing
        self.skip_counter = 0
        self.motion_detector = None
        
    def start(self):
        """Start the video stream thread."""
        Thread(target=self.update, args=(), daemon=True).start()
        return self
        
    def update(self):
        """Update the video stream continuously."""
        try:
            while not self.stopped:
                # Read the next frame
                (self.grabbed, frame) = self.stream.read()
                
                # Calculate FPS
                self.frame_count += 1
                self.fps_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.fps_count
                    self.fps_count = 0
                    self.last_fps_time = current_time
                
                # Skip frames if needed
                if self.skip_frames > 0:
                    self.skip_counter = (self.skip_counter + 1) % (self.skip_frames + 1)
                    if self.skip_counter != 0:
                        continue
                
                # Store the frame
                if self.grabbed:
                    self.frame = frame
                else:
                    self.stop()
                    break
        except Exception as e:
            logger.error(f"Error in video stream: {e}")
            self.stop()
            
    def read(self):
        """Read the current frame."""
        return self.grabbed, self.frame, False
        
    def get_fps(self):
        """Get the current FPS."""
        return self.fps
        
    def set_skip_frames(self, skip_frames):
        """Set the number of frames to skip between processing."""
        self.skip_frames = max(0, skip_frames)
        self.skip_counter = 0
        logger.info(f"Set frame skip to {self.skip_frames}")
        
    def stop(self):
        """Stop the video stream."""
        self.stopped = True
        self.stream.release()

class LightweightDetector(BaseDetector):
    """Lightweight TensorFlow Lite detector optimized for CPU."""
    
    def __init__(self, 
                 model_path: str = "models/person_detection_model.tflite",
                 label_path: str = "models/person_labels.txt",
                 threshold: float = 0.5,
                 resolution: Tuple[int, int] = (320, 320),
                 num_threads: int = 4):
        """Initialize the lightweight detector.
        
        Args:
            model_path: Path to the TFLite model
            label_path: Path to the labels file
            threshold: Detection confidence threshold
            resolution: Input resolution for the model
            num_threads: Number of threads for inference
        """
        super().__init__(threshold=threshold)
        
        self.model_path = self._resolve_model_path(model_path)
        self.label_path = self._resolve_model_path(label_path)
        self.input_resolution = resolution
        self.num_threads = num_threads
        
        # TFLite model variables
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = []
        
        # Performance optimization
        self.frame_skip = 1
        self.process_resolution = resolution
        self.original_resolution = None
        self.processing_scale = 1.0
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"Lightweight detector initialized successfully")
    
    def _resolve_model_path(self, file_path: str) -> str:
        """Resolve the model path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Resolved file path
        """
        # Check if path is absolute
        if os.path.isabs(file_path):
            if os.path.exists(file_path):
                return file_path
        
        # Check relative to current directory
        if os.path.exists(file_path):
            return file_path
        
        # Check relative to src directory
        src_path = os.path.join("src", file_path)
        if os.path.exists(src_path):
            return src_path
        
        # Check relative to project root
        root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), file_path)
        if os.path.exists(root_path):
            return root_path
        
        # Default to the provided path and let the model loader handle it
        logger.warning(f"File path {file_path} not found, using as is")
        return file_path
    
    def _initialize_model(self):
        """Initialize the TFLite model."""
        try:
            # Check if TensorFlow Lite interpreter is available
            pkg = importlib.util.find_spec('tflite_runtime')
            if pkg:
                from tflite_runtime.interpreter import Interpreter
                logger.info("Using TFLite Runtime")
            else:
                from tensorflow.lite.python.interpreter import Interpreter
                logger.info("Using TensorFlow Lite")
            
            # Load labels
            try:
                with open(self.label_path, 'r') as f:
                    self.labels = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(self.labels)} labels")
            except Exception as e:
                logger.error(f"Error loading labels: {e}")
                self.labels = ["person"]  # Default label
            
            # Load model
            try:
                self.interpreter = Interpreter(model_path=self.model_path, num_threads=self.num_threads)
                self.interpreter.allocate_tensors()
                
                # Get model details
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                # Get model input shape
                input_shape = self.input_details[0]['shape']
                self.input_height = input_shape[1]
                self.input_width = input_shape[2]
                
                logger.info(f"Model loaded with input shape: {input_shape}")
                logger.info(f"Using {self.num_threads} threads for inference")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
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
                self.input_width / self.original_resolution[0],
                self.input_height / self.original_resolution[1]
            )
        
        # Create a working copy of the frame
        display_frame = frame.copy()
        
        # Preprocess the frame
        try:
            # Resize frame to model input size
            input_frame = cv2.resize(frame, (self.input_width, self.input_height))
            
            # Convert to RGB (TFLite models typically expect RGB)
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values if needed
            input_frame = input_frame.astype(np.float32) / 255.0
            
            # Add batch dimension
            input_tensor = np.expand_dims(input_frame, axis=0)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get detection results
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            
            # Process detections
            detections = []
            for i in range(len(scores)):
                if scores[i] >= self.threshold:
                    # Get class label
                    class_id = int(classes[i])
                    if class_id < len(self.labels):
                        label = self.labels[class_id]
                    else:
                        label = f"Class {class_id}"
                    
                    # Get bounding box
                    ymin, xmin, ymax, xmax = boxes[i]
                    
                    # Convert normalized coordinates to pixel coordinates
                    xmin = int(xmin * frame.shape[1])
                    xmax = int(xmax * frame.shape[1])
                    ymin = int(ymin * frame.shape[0])
                    ymax = int(ymax * frame.shape[0])
                    
                    # Add detection
                    detections.append({
                        'bbox': (xmin, ymin, xmax, ymax),
                        'confidence': float(scores[i]),
                        'class': label,
                        'status': 'detected'
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    
                    # Draw label
                    label_text = f"{label}: {scores[i]:.2f}"
                    cv2.putText(display_frame, label_text, (xmin, ymin - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
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
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {
                'frame': frame,
                'detections': [],
                'fps': 0
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
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Remove interpreter reference
            self.interpreter = None
            self.input_details = None
            self.output_details = None
        except Exception as e:
            logger.error(f"Error cleaning up: {e}") 