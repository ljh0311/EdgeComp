"""
Detector Factory Module
=====================
Factory for creating different types of detectors.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional
from .person_detector import PersonDetector
from .emotion_detector import EmotionDetector

# Configure logging
logger = logging.getLogger(__name__)

class DetectorType(Enum):
    """Supported detector types."""
    PERSON = "person"
    YOLOV8 = "yolov8"
    EMOTION = "emotion"

class DetectorFactory:
    """Factory for creating different types of detectors."""
    
    @staticmethod
    def create_detector(detector_type: str = "person",
                       model_path: Optional[str] = None,
                       threshold: float = 0.5,
                       force_cpu: bool = False,
                       config: Optional[Dict[str, Any]] = None,
                       **kwargs) -> PersonDetector:
        """Create a detector instance.
        
        Args:
            detector_type: Type of detector ("person", "yolov8")
            model_path: Optional path to model files
            threshold: Detection confidence threshold
            force_cpu: Force CPU usage
            config: Optional configuration dictionary
            **kwargs: Additional keyword arguments for detector
            
        Returns:
            PersonDetector instance
            
        Raises:
            ValueError: If detector_type is not supported
        """
        detector_type = detector_type.lower()
        
        # If config is provided, use it to update parameters
        if config is not None:
            model_path = config.get("model_path", model_path)
            threshold = config.get("threshold", threshold)
            force_cpu = config.get("force_cpu", force_cpu)
            # Add any other kwargs from config
            for key, value in config.items():
                if key not in ["model_path", "threshold", "force_cpu"]:
                    kwargs[key] = value
        
        if detector_type == "person" or detector_type == "yolov8":
            return PersonDetector(
                model_path=model_path,
                threshold=threshold,
                force_cpu=force_cpu,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")

    @staticmethod
    def create_emotion_detector(config: Dict[str, Any]) -> EmotionDetector:
        """Create an emotion detector instance.
        
        Args:
            config: Configuration for the emotion detector
            
        Returns:
            EmotionDetector instance
        """
        return EmotionDetector(
            model_path=config.get("model_path"),
            threshold=config.get("threshold", 0.5),
            device=config.get("device", "cpu")
        )

    @staticmethod
    def create_detector_instance(detector_type: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create a detector instance.
        
        Args:
            detector_type: Type of detector to create
            config: Configuration for the detector
            
        Returns:
            Detector instance
        """
        if config is None:
            config = {}
            
        detector_type = detector_type.lower()
        
        try:
            if detector_type == DetectorType.PERSON.value or detector_type == DetectorType.YOLOV8.value:
                return PersonDetector(
                    model_path=config.get("model_path"),
                    threshold=config.get("threshold", 0.5),
                    force_cpu=config.get("force_cpu", False),
                    max_retries=config.get("max_retries", 3)
                )
                
            elif detector_type == DetectorType.EMOTION.value:
                return EmotionDetector(
                    model_path=config.get("model_path"),
                    threshold=config.get("threshold", 0.5),
                    device=config.get("device", "cpu")
                )
                
            else:
                raise ValueError(f"Unknown detector type: {detector_type}")
                
        except Exception as e:
            logger.error(f"Error creating detector of type {detector_type}: {str(e)}")
            raise
            
    @staticmethod
    def create_video_stream(detector_type: str = "person", config: Optional[Dict[str, Any]] = None) -> Any:
        """Create a video stream for testing detectors.
        
        Args:
            detector_type: Type of detector (not used currently)
            config: Video stream configuration
                
        Returns:
            Video stream object with read() method
        """
        import cv2
        
        if config is None:
            config = {}
            
        # Get configuration parameters
        camera_index = config.get("camera_index", 0)
        resolution = config.get("resolution", (640, 480))
        framerate = config.get("framerate", 30)
        
        # Create a simple wrapper for OpenCV's VideoCapture
        class VideoStream:
            def __init__(self, camera_index, resolution, framerate):
                self.camera_index = camera_index
                self.resolution = resolution
                self.framerate = framerate
                self.cap = None
                
            def start(self):
                self.cap = cv2.VideoCapture(self.camera_index)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
                return self
                
            def read(self):
                if self.cap is None:
                    self.start()
                ret, frame = self.cap.read()
                return ret, frame
                
            def release(self):
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
        
        return VideoStream(camera_index, resolution, framerate) 