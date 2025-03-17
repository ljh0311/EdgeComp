"""
Detector Factory Module
----------------------
Provides a factory for creating different types of detectors.
This makes it easy to switch between different detection implementations.
"""

import os
import logging
import subprocess
import sys
from typing import Dict, Any, Optional, Union, Type
from enum import Enum

# Import detectors
from .base_detector import BaseDetector
from .lightweight_detector import LightweightDetector, VideoStream as LightweightVideoStream
from .person_detector import PersonDetector
from .person_tracker import PersonTracker
from .motion_detector import MotionDetector
from .emotion_detector import EmotionDetector

# Configure logging
logger = logging.getLogger(__name__)

class DetectorType(Enum):
    """Enum for detector types."""
    LIGHTWEIGHT = "lightweight"
    YOLOV8 = "yolov8"
    TRACKER = "tracker"
    MOTION = "motion"
    EMOTION = "emotion"

class DetectorFactory:
    """Factory for creating different types of detectors."""
    
    @staticmethod
    def create_detector(detector_type: str, config: Optional[Dict[str, Any]] = None) -> BaseDetector:
        """Create a detector of the specified type.
        
        Args:
            detector_type: Type of detector to create (use DetectorType enum)
            config: Configuration for the detector
            
        Returns:
            Detector instance inheriting from BaseDetector
        """
        if config is None:
            config = {}
            
        detector_type = detector_type.lower()
        
        if detector_type == DetectorType.LIGHTWEIGHT.value:
            return DetectorFactory._create_lightweight_detector(config)
        elif detector_type == DetectorType.YOLOV8.value:
            return DetectorFactory._create_yolov8_detector(config)
        elif detector_type == DetectorType.TRACKER.value:
            return DetectorFactory._create_tracker(config)
        elif detector_type == DetectorType.MOTION.value:
            return DetectorFactory._create_motion_detector(config)
        elif detector_type == DetectorType.EMOTION.value:
            return DetectorFactory._create_emotion_detector(config)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    @staticmethod
    def _resolve_model_path(model_path: str, fallback_filename: str = None) -> str:
        """Resolve the model path, checking various locations.
        
        Args:
            model_path: Original model path
            fallback_filename: Fallback filename if original not found
            
        Returns:
            Resolved model path
        """
        # Check if model file exists as is
        if os.path.exists(model_path):
            return model_path
            
        # Try to find the model in the models directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        model_dir = os.path.join(project_root, "models")
        
        # Try with original filename
        model_path_alt = os.path.join(model_dir, os.path.basename(model_path))
        if os.path.exists(model_path_alt):
            logger.info(f"Using model from models directory: {model_path_alt}")
            return model_path_alt
        
        # Try with fallback filename
        if fallback_filename:
            fallback_path = os.path.join(model_dir, fallback_filename)
            if os.path.exists(fallback_path):
                logger.info(f"Using fallback model: {fallback_path}")
                return fallback_path
        
        # Return original path if nothing found (let the detector handle it)
        logger.warning(f"Model not found: {model_path}")
        return model_path
            
    @staticmethod
    def _create_lightweight_detector(config: Dict[str, Any]) -> LightweightDetector:
        """Create a lightweight detector.
        
        Args:
            config: Configuration for the detector
            
        Returns:
            LightweightDetector instance
        """
        model_path = config.get("model_path", "models/person_detection_model.tflite")
        label_path = config.get("label_path", "models/person_labels.txt")
        threshold = config.get("threshold", 0.5)
        resolution = config.get("resolution", (320, 320))
        num_threads = config.get("num_threads", 4)
        
        # Resolve model path
        model_path = DetectorFactory._resolve_model_path(
            model_path, 
            fallback_filename="person_detection_model.tflite"
        )
        
        # Resolve label path
        if not os.path.exists(label_path):
            label_dir = os.path.dirname(model_path)
            label_path = os.path.join(label_dir, os.path.basename(label_path))
        
        # Try to create the detector
        try:
            logger.info(f"Creating lightweight detector with model: {model_path}")
            return LightweightDetector(
                model_path=model_path,
                label_path=label_path,
                threshold=threshold,
                resolution=resolution,
                num_threads=num_threads
            )
        except RuntimeError as e:
            # Check if the error is related to EdgeTPU
            if "edgetpu-custom-op" in str(e):
                logger.error("The model contains EdgeTPU operations but EdgeTPU is not available.")
                logger.info("Attempting to download a CPU-compatible model...")
                
                # Try to download a CPU-compatible model
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                scripts_dir = os.path.join(project_root, "scripts")
                download_script = os.path.join(scripts_dir, "download_tflite_model.py")
                
                if os.path.exists(download_script):
                    try:
                        # Run the download script
                        subprocess.check_call([sys.executable, download_script])
                        
                        # Try again with the downloaded model
                        model_dir = os.path.join(project_root, "models")
                        logger.info("Downloaded CPU-compatible model. Trying again...")
                        return LightweightDetector(
                            model_path=os.path.join(model_dir, "person_detection_model.tflite"),
                            label_path=os.path.join(model_dir, "person_labels.txt"),
                            threshold=threshold,
                            resolution=resolution,
                            num_threads=num_threads
                        )
                    except Exception as download_error:
                        logger.error(f"Failed to download CPU-compatible model: {download_error}")
                        raise RuntimeError("Cannot create lightweight detector. Please run 'python scripts/download_tflite_model.py' manually to download a CPU-compatible model.") from e
                else:
                    raise RuntimeError("Cannot create lightweight detector. Please run 'python scripts/download_tflite_model.py' manually to download a CPU-compatible model.") from e
            else:
                # Re-raise the original error
                raise
        
    @staticmethod
    def _create_yolov8_detector(config: Dict[str, Any]) -> PersonDetector:
        """Create a YOLOv8 detector.
        
        Args:
            config: Configuration for the detector
            
        Returns:
            PersonDetector instance
        """
        model_path = config.get("model_path", "yolov8n.pt")
        threshold = config.get("threshold", 0.5)
        force_cpu = config.get("force_cpu", False)
        
        # Resolve model path
        model_path = DetectorFactory._resolve_model_path(model_path, "yolov8n.pt")
        
        logger.info(f"Creating YOLOv8 detector with model: {model_path}")
        return PersonDetector(
            model_path=model_path,
            threshold=threshold,
            force_cpu=force_cpu
        )
        
    @staticmethod
    def _create_tracker(config: Dict[str, Any]) -> PersonTracker:
        """Create a person tracker.
        
        Args:
            config: Configuration for the tracker
            
        Returns:
            PersonTracker instance
        """
        model_path = config.get("model_path", "yolov8n.pt")
        threshold = config.get("threshold", 0.5)
        force_cpu = config.get("force_cpu", False)
        
        # Resolve model path
        model_path = DetectorFactory._resolve_model_path(model_path, "yolov8n.pt")
        
        logger.info(f"Creating person tracker with model: {model_path}")
        return PersonTracker(
            model_path=model_path,
            threshold=threshold,
            force_cpu=force_cpu
        )
    
    @staticmethod
    def _create_motion_detector(config: Dict[str, Any]) -> MotionDetector:
        """Create a motion detector.
        
        Args:
            config: Configuration for the motion detector
            
        Returns:
            MotionDetector instance
        """
        motion_threshold = config.get("motion_threshold", 0.02)
        fall_threshold = config.get("fall_threshold", 0.15)
        history = config.get("history", 500)
        var_threshold = config.get("var_threshold", 16)
        detect_shadows = config.get("detect_shadows", False)
        
        logger.info(f"Creating motion detector with threshold: {motion_threshold}")
        return MotionDetector(
            motion_threshold=motion_threshold,
            fall_threshold=fall_threshold,
            history=history,
            var_threshold=var_threshold,
            detect_shadows=detect_shadows
        )
    
    @staticmethod
    def _create_emotion_detector(config: Dict[str, Any]) -> EmotionDetector:
        """Create an emotion detector.
        
        Args:
            config: Configuration for the emotion detector
            
        Returns:
            EmotionDetector instance
        """
        model_path = config.get("model_path", "models/emotion_model.pth")
        confidence_threshold = config.get("confidence_threshold", 0.6)
        warning_threshold = config.get("warning_threshold", 0.7)
        critical_threshold = config.get("critical_threshold", 0.8)
        sample_rate = config.get("sample_rate", 16000)
        
        # Resolve model path
        model_path = DetectorFactory._resolve_model_path(model_path, "emotion_model.pth")
        
        logger.info(f"Creating emotion detector with model: {model_path}")
        return EmotionDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            sample_rate=sample_rate
        )
        
    @staticmethod
    def create_video_stream(detector_type: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create a video stream appropriate for the detector type.
        
        Args:
            detector_type: Type of detector to create a video stream for
            config: Configuration for the video stream
            
        Returns:
            VideoStream instance
        """
        if config is None:
            config = {}
            
        detector_type = detector_type.lower()
        
        if detector_type == DetectorType.LIGHTWEIGHT.value:
            from .lightweight_detector import VideoStream
            
            resolution = config.get("resolution", (640, 480))
            framerate = config.get("framerate", 30)
            camera_index = config.get("camera_index", 0)
            buffer_size = config.get("buffer_size", 1)  # Smaller buffer for lower latency
            
            # Create and configure the video stream
            video_stream = VideoStream(
                resolution=resolution,
                framerate=framerate,
                camera_index=camera_index,
                buffer_size=buffer_size
            )
            
            # Set frame skipping based on device capabilities
            # Skip frames if running on a resource-constrained device
            import platform
            if platform.machine() in ('armv7l', 'armv6l'):
                # On Raspberry Pi, skip every other frame
                video_stream.set_skip_frames(1)
            
            return video_stream
        else:
            # For other detectors, use OpenCV's VideoCapture directly
            import cv2
            camera_index = config.get("camera_index", 0)
            resolution = config.get("resolution", (640, 480))
            
            cap = cv2.VideoCapture(camera_index)
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            
            # Set MJPG format for better performance
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Set framerate
            cap.set(cv2.CAP_PROP_FPS, config.get("framerate", 30))
            
            # Set buffer size
            cap.set(cv2.CAP_PROP_BUFFERSIZE, config.get("buffer_size", 1))
            
            return cap 