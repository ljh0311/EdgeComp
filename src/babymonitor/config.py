"""
Configuration Module
==================
Contains configuration settings for the Baby Monitor System.
"""

import os
import logging
import platform
from pathlib import Path

# Visualization settings
ENABLE_VISUALIZATION = False  # Set to False for Raspberry Pi headless operation


class Config:
    """Configuration settings."""

    # Base paths
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"

    # Logging configuration
    LOGGING = {
        "level": logging.INFO,
        "format": "%(asctime)s - %(levelname)s - %(message)s",
        "handlers": [logging.FileHandler("baby_monitor.log"), logging.StreamHandler()],
    }

    # Web interface settings
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 5000

    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    # Person detection settings
    PERSON_DETECTION = {
        "model_path": str(MODELS_DIR / "yolov8n.pt"),
        "confidence_threshold": 0.5,
        "device": "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu",
    }
    
    # Lightweight detection settings
    LIGHTWEIGHT_DETECTION = {
        "model_path": "models/person_detection_model.tflite",
        "label_path": "models/person_labels.txt",
        "threshold": 0.5,
        "resolution": (640, 480),
        "num_threads": 4,
        "camera_index": 0,
    }
    
    # Detector selection
    # Options: "yolov8", "lightweight"
    DETECTOR_TYPE = "lightweight"
    # Auto-select lightweight detector on resource-constrained devices
    if platform.machine() in ('armv7l', 'armv6l') or os.environ.get("USE_LIGHTWEIGHT", "0") == "1":
        DETECTOR_TYPE = "lightweight"

    # Motion detection settings
    MOTION_DETECTION = {
        "history": 20,
        "dist2threshold": 400,
        "detect_shadows": True,
        "motion_threshold": 0.02,
        "fall_threshold": 0.1,
    } 