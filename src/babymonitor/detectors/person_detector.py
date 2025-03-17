"""
Person Detector Module
===================
Efficient person detection using Haar Cascade classifiers for face and body detection.
"""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Any, Tuple
from .base_detector import BaseDetector


class PersonDetector(BaseDetector):
    """Person detector using Haar Cascade classifiers for face and body detection."""

    def __init__(
        self,
        model_path: str = None,
        threshold: float = 0.7,  # Increased threshold for better stability
        target_size: Tuple[int, int] = (640, 480),
    ):
        """Initialize the person detector.

        Args:
            model_path: Path to model directory (not used for Haar Cascade)
            threshold: Detection confidence threshold (used for filtering)
            target_size: Target frame size for processing
        """
        super().__init__(threshold=threshold)

        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors
        self._initialize_detector()
        
        # Initialize state
        self.last_detection_time = time.time()
        self.detection_history = []
        self.max_history_size = 10
        
        # Store last result for web interface
        self.last_result = None
        
        # Lower threshold for initial detection to improve sensitivity
        self.initial_detection_threshold = max(0.4, self.threshold - 0.2)

    def _initialize_detector(self):
        """Initialize the Haar Cascade detectors for face and body."""
        try:
            # Load the pre-trained face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Load the pre-trained body detectors
            self.upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
            self.full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            self.lower_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')
            
            # Check if classifiers loaded successfully
            if (self.face_cascade.empty() or self.upper_body_cascade.empty() or 
                self.full_body_cascade.empty() or self.lower_body_cascade.empty()):
                self.logger.error("Error loading Haar Cascade classifiers")
                raise RuntimeError("Failed to load Haar Cascade classifiers")
                
            self.logger.info("Haar Cascade classifiers loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing detector: {str(e)}")
            raise

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a frame to detect people.

        Args:
            frame: Input frame

        Returns:
            Dict containing processed frame and detections
        """
        if frame is None:
            return {"frame": frame, "detections": [], "fps": self.fps}

        start_time = time.time()

        try:
            # Make a copy of the frame for drawing
            frame_with_detections = frame.copy()
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization to improve detection in different lighting
            gray = cv2.equalizeHist(gray)
            
            # Detect faces with lower initial threshold for better sensitivity
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,  # Slightly reduced for better detection
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Detect upper bodies
            upper_bodies = self.upper_body_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(80, 80),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Detect full bodies
            full_bodies = self.full_body_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(80, 200),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Detect lower bodies
            lower_bodies = self.lower_body_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(60, 120),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Process detections
            results = []
            
            # Define colors for different detection types (BGR format)
            colors = {
                "face": (0, 255, 0),      # Green
                "upper_body": (255, 0, 0), # Blue
                "full_body": (0, 0, 255),  # Red
                "lower_body": (0, 255, 255) # Yellow
            }
            
            # Helper function to draw bounding boxes with consistent style
            def draw_detection(frame, bbox, label, color):
                x, y, w, h = bbox
                # Draw rectangle with specified color
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
            
            # Process face detections
            for (x, y, w, h) in faces:
                # Calculate confidence (just a placeholder since Haar doesn't provide confidence)
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                confidence = min(0.95, max(0.5, face_area / frame_area * 10))
                
                # Only include detections above initial threshold
                if confidence >= self.initial_detection_threshold:
                    # Create detection result
                    detection_result = {
                        "bbox": (x, y, x + w, y + h),
                        "confidence": float(confidence),  # Ensure it's a float for JSON serialization
                        "class": "face"
                    }
                    results.append(detection_result)
                    
                    # Draw the face detection
                    draw_detection(
                        frame_with_detections,
                        (x, y, w, h),
                        f"Face: {confidence:.2f}",
                        colors["face"]
                    )
            
            # Process upper body detections
            for (x, y, w, h) in upper_bodies:
                # Calculate confidence
                body_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                confidence = min(0.9, max(0.5, body_area / frame_area * 8))
                
                # Only include detections above initial threshold
                if confidence >= self.initial_detection_threshold:
                    # Create detection result
                    detection_result = {
                        "bbox": (x, y, x + w, y + h),
                        "confidence": float(confidence),  # Ensure it's a float for JSON serialization
                        "class": "upper_body"
                    }
                    results.append(detection_result)
                    
                    # Draw the body detection
                    draw_detection(
                        frame_with_detections,
                        (x, y, w, h),
                        f"Upper: {confidence:.2f}",
                        colors["upper_body"]
                    )
            
            # Process full body detections
            for (x, y, w, h) in full_bodies:
                # Calculate confidence
                body_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                confidence = min(0.9, max(0.6, body_area / frame_area * 10))
                
                # Only include detections above initial threshold
                if confidence >= self.initial_detection_threshold:
                    # Create detection result
                    detection_result = {
                        "bbox": (x, y, x + w, y + h),
                        "confidence": float(confidence),  # Ensure it's a float for JSON serialization
                        "class": "full_body"
                    }
                    results.append(detection_result)
                    
                    # Draw the body detection
                    draw_detection(
                        frame_with_detections,
                        (x, y, w, h),
                        f"Full: {confidence:.2f}",
                        colors["full_body"]
                    )
            
            # Process lower body detections
            for (x, y, w, h) in lower_bodies:
                # Calculate confidence
                body_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                confidence = min(0.85, max(0.5, body_area / frame_area * 8))
                
                # Only include detections above initial threshold
                if confidence >= self.initial_detection_threshold:
                    # Create detection result
                    detection_result = {
                        "bbox": (x, y, x + w, y + h),
                        "confidence": float(confidence),  # Ensure it's a float for JSON serialization
                        "class": "lower_body"
                    }
                    results.append(detection_result)
                    
                    # Draw the body detection
                    draw_detection(
                        frame_with_detections,
                        (x, y, w, h),
                        f"Lower: {confidence:.2f}",
                        colors["lower_body"]
                    )
            
            # Update detection history
            if len(results) > 0:
                self.last_detection_time = time.time()
                self.detection_history.append(len(results))
                if len(self.detection_history) > self.max_history_size:
                    self.detection_history.pop(0)
            
            # Draw status information with consistent style
            # Create semi-transparent background for status info
            overlay = frame_with_detections.copy()
            cv2.rectangle(
                overlay,
                (5, 5),
                (250, 130),
                (0, 0, 0),
                -1
            )
            # Apply transparency
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame_with_detections, 1 - alpha, 0, frame_with_detections)
            
            # Add detection count to frame
            cv2.putText(
                frame_with_detections,
                f"Persons: {len(results)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            
            # Update FPS
            self.update_fps(time.time() - start_time)
            
            # Add FPS to frame
            cv2.putText(
                frame_with_detections,
                f"FPS: {self.fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            
            # Add detection history
            if self.detection_history:
                avg_detections = sum(self.detection_history) / len(self.detection_history)
                cv2.putText(
                    frame_with_detections,
                    f"Avg Detections: {avg_detections:.1f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
            
            # Add threshold info
            cv2.putText(
                frame_with_detections,
                f"Threshold: {self.threshold:.1f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            
            # Store the result for web interface
            result = {
                "frame": frame_with_detections,
                "detections": results,
                "fps": float(self.fps)  # Ensure it's a float for JSON serialization
            }
            self.last_result = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"frame": frame, "detections": [], "fps": float(self.fps)}  # Ensure it's a float for JSON serialization

    def cleanup(self):
        """Clean up resources."""
        self.face_cascade = None
        self.upper_body_cascade = None
        self.full_body_cascade = None
        self.lower_body_cascade = None
