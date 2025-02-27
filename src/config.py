"""
Configuration Module
==================
Contains configuration settings for the Baby Monitor System.
"""

import os
import logging
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

    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    # Person detection settings
    PERSON_DETECTION = {
        "model_path": str(MODELS_DIR / "yolov8n.pt"),
        "confidence_threshold": 0.5,
        "device": "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu",
    }

    # Motion detection settings
    MOTION_DETECTION = {
        "history": 20,
        "dist2threshold": 400,
        "detect_shadows": True,
        "motion_threshold": 0.02,
        "fall_threshold": 0.1,
    }

    # Audio processing settings
    AUDIO_PROCESSING = {
        "sample_rate": 16000,
        "channels": 1,
        "format": "paFloat32",
        "chunk_size": 1024,
        "analysis_window": 2.0,  # seconds
        "alert_cooldown": 5.0,  # seconds between alerts
        "model_path": str(MODELS_DIR / "audio_model.pt"),
        "device": "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu",
        # Emotion detection settings
        "emotion_detection": {
            "model_path": str(MODELS_DIR / "hubert-base-ls960_emotion.pt"),
            "confidence_threshold": 0.7,
            "critical_threshold": 0.8,  # Threshold for critical alerts
            "warning_threshold": 0.6,  # Threshold for warning alerts
            "device": "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu",
            # Emotion alert mappings
            "critical_emotions": ["Anger", "Fear", "Sadness"],
            "warning_emotions": ["Worried"],
        },
    }

    # Alert settings
    ALERT_SETTINGS = {
        "duration": 5000,  # Alert display duration in ms
        "max_history": 50,  # Maximum number of alerts to keep in history
        "flash_interval": 500,  # Flash interval for critical alerts in ms
        # Sound settings
        "critical_frequency": 2000,
        "critical_duration": 1000,
        "warning_frequency": 1000,
        "warning_duration": 500,
        # Theme colors
        "critical_color": "#e74c3c",
        "warning_color": "#f1c40f",
        "dark_theme": {
            "background": "#1a1a1a",
            "text": "#ffffff",
            "accent_text": "#3498db",
        },
    }

    # Feature extraction settings
    FEATURE_EXTRACTION = {
        "frame_length": 2048,
        "hop_length": 512,
        "n_mels": 128,
        "n_mfcc": 20,
        "fmin": 20,
        "fmax": 8000,
    }
