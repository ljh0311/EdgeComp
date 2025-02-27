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
    DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"

    # Path configurations
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # System-wide settings
    ENABLE_VISUALIZATION = not IS_RASPBERRY_PI  # Enable visualization except on Raspberry Pi

    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30 if not IS_RASPBERRY_PI else 15  # Lower FPS for Pi

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
        "motion_threshold": 100,  # Threshold for motion detection
        "fall_threshold": 1.5,  # Aspect ratio change threshold for fall detection
        "frame_history": 5,  # Number of frames to keep for motion analysis
        "device": DEVICE,
        "enable_visualization": ENABLE_VISUALIZATION
    }

    # Audio and Emotion detection settings
    EMOTION_DETECTION = {
        "model_path": os.path.join(MODELS_DIR, "emotion_model.pt"),
        "confidence_threshold": 0.5,  # Minimum confidence to consider
        "critical_threshold": 0.7,  # Threshold for critical emotions
        "warning_threshold": 0.6,  # Threshold for warning emotions
        "sampling_rate": 16000,  # Required sampling rate for HuBERT
        "chunk_size": 16000,  # Audio chunk size (1 second at 16kHz)
        "device": DEVICE,
        "enable_visualization": ENABLE_VISUALIZATION
    }

    AUDIO = {
        "format": "paFloat32",
        "channels": 1,
        "rate": 44100,
        "chunk": 1024,
        "critical_frequency": 1000,  # Hz for critical alerts
        "warning_frequency": 500,  # Hz for warning alerts
        "critical_duration": 500,  # ms for critical alerts
        "warning_duration": 200,  # ms for warning alerts
    }

    AUDIO_PROCESSING = {
        "sample_rate": 44100,
        "channels": 1,
        "chunk_size": 1024,
        "format": "paFloat32",
        "noise_threshold": 0.1,  # Threshold for noise detection
        "cry_threshold": 0.6,  # Threshold for cry detection
        "device": DEVICE,
        "model_path": os.path.join(MODELS_DIR, "audio_model.pt"),
        "enable_cry_detection": True,
        "enable_noise_detection": True,
        "alert_cooldown": 5.0,  # Minimum time between alerts in seconds
        "analysis_window": 2.0,  # Time window for audio analysis in seconds
        "enable_visualization": ENABLE_VISUALIZATION,
        "emotion_detection": EMOTION_DETECTION  # Include emotion detection settings
    }

    # Alert settings
    ALERT = {
        "max_history": 10,  # Maximum number of alerts to show in history
        "duration": 8000,  # Time in ms before alert fades
        "critical_color": "#FF3D00",  # Material Design Red A400
        "warning_color": "#FF9100",  # Material Design Orange A400
        "info_color": "#00B0FF",  # Material Design Light Blue A400
        "flash_interval": 500,  # Flash interval for critical alerts
        "critical_frequency": 1000,  # Hz for critical alert sound
        "warning_frequency": 500,  # Hz for warning alert sound
        "critical_duration": 500,  # ms for critical alert sound
        "warning_duration": 200,  # ms for warning alert sound
        "dark_theme": {
            "background": "#1E1E1E",
            "text": "#E0E0E0",
            "accent": "#2D2D2D",
            "accent_text": "#4CAF50",
            "foreground": "#FFFFFF",
        },
    }

    # UI settings
    UI = {
        "min_width": 800,
        "min_height": 600,
        "window_scale": 0.75,  # Initial window size as fraction of screen
        "dark_theme": {
            "background": "#1E1E1E",
            "text": "#E0E0E0",
            "accent": "#2D2D2D",
            "accent_text": "#4CAF50",
            "alert": "#FF5252",
        },
        "enable_visualization": ENABLE_VISUALIZATION
    }

    # Performance settings
    PERFORMANCE = {
        "frame_skip": 0 if not IS_RASPBERRY_PI else 1,  # Skip frames on Pi
        "detection_interval": 1 if not IS_RASPBERRY_PI else 2,  # Seconds between detections
        "max_fps": 30 if not IS_RASPBERRY_PI else 15,
        "use_gpu": CUDA_AVAILABLE,
        "gpu_memory_fraction": 0.6,  # Maximum fraction of GPU memory to use
    }

    # Logging configuration
    LOGGING = {
        "level": logging.INFO,
        "format": "%(asctime)s [%(levelname)s] %(message)s",
        "datefmt": "%H:%M:%S",
        "handlers": [logging.StreamHandler()],
    }

    # Custom filter for YOLO output
    class YOLOFilter(logging.Filter):
        def filter(self, record):
            return (
                "Speed:" not in record.getMessage()
                and not record.getMessage().startswith("0:")
            )

    @classmethod
    def log_system_info(cls):
        """Log system information and GPU availability."""
        logging.info(f"Platform: {platform.platform()}")
        logging.info(f"Python version: {platform.python_version()}")
        if cls.CUDA_AVAILABLE:
            logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logging.info(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
        else:
            logging.info("No GPU available, using CPU")
