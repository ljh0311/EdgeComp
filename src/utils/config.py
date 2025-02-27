"""
Configuration Module
==================
Handles system-wide configuration settings and parameters.
"""

import os
import platform
import logging
import torch


class Config:
    """System-wide configuration settings."""
    
    # Platform and hardware detection
    IS_RASPBERRY_PI = platform.machine() in ("armv7l", "aarch64")
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cpu"  # Force CPU for Raspberry Pi

    # Path configurations
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # System-wide settings
    ENABLE_VISUALIZATION = False  # Disable visualization for Raspberry Pi

    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 15  # Optimized for Pi

    # Detection settings
    PERSON_DETECTION = {
        "model_path": os.path.join(MODELS_DIR, "yolov8n.pt"),  # Use YOLOv8 nano model
        "conf": 0.5,  # Confidence threshold
        "iou": 0.45,  # NMS IOU threshold
        "device": DEVICE,
        "enable_visualization": ENABLE_VISUALIZATION
    }

    POSE_DETECTION = {
        "path": os.path.join(MODELS_DIR, "yolov8n-pose.pt"),
        "conf": 0.25,
        "iou": 0.45,
        "device": DEVICE,
        "enable_visualization": ENABLE_VISUALIZATION
    }

    MOTION_DETECTION = {
        "motion_threshold": 100,
        "fall_threshold": 1.5,
        "frame_history": 3,  # Reduced for Pi
        "device": DEVICE,
        "enable_visualization": ENABLE_VISUALIZATION
    }

    # Audio and Emotion detection settings
    EMOTION_DETECTION = {
        "model_path": os.path.join(MODELS_DIR, "emotion_model.pt"),
        "confidence_threshold": 0.5,
        "critical_threshold": 0.7,
        "warning_threshold": 0.6,
        "sampling_rate": 16000,
        "chunk_size": 8000,  # Reduced for Pi
        "device": DEVICE,
        "enable_visualization": ENABLE_VISUALIZATION
    }

    AUDIO = {
        "format": "paFloat32",
        "channels": 1,
        "rate": 44100,
        "chunk": 512,  # Reduced for Pi
        "critical_frequency": 1000,
        "warning_frequency": 500,
        "critical_duration": 500,
        "warning_duration": 200,
    }

    AUDIO_PROCESSING = {
        "sample_rate": 44100,
        "channels": 1,
        "chunk_size": 512,  # Reduced for Pi
        "format": "paFloat32",
        "noise_threshold": 0.1,
        "cry_threshold": 0.6,
        "device": DEVICE,
        "model_path": os.path.join(MODELS_DIR, "audio_model.pt"),
        "enable_cry_detection": True,
        "enable_noise_detection": True,
        "alert_cooldown": 5.0,
        "analysis_window": 1.0,  # Reduced for Pi
        "enable_visualization": ENABLE_VISUALIZATION,
        "emotion_detection": EMOTION_DETECTION
    }

    # Performance settings
    PERFORMANCE = {
        "frame_skip": 1,  # Enable frame skipping
        "detection_interval": 2,  # 2 seconds between detections
        "max_fps": 15,
        "use_gpu": False,  # Force CPU
        "memory_limit": 512,  # Memory limit in MB
        "thread_pool_size": 2  # Limit concurrent threads
    }

    # Logging configuration
    LOGGING = {
        "level": logging.INFO,
        "format": "%(asctime)s [%(levelname)s] %(message)s",
        "datefmt": "%H:%M:%S",
        "handlers": [logging.StreamHandler()],
    }

    @classmethod
    def log_system_info(cls):
        """Log system information."""
        logging.info(f"Platform: {platform.platform()}")
        logging.info(f"Python version: {platform.python_version()}")
        logging.info("Running on Raspberry Pi - CPU only mode")
