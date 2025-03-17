"""
Baby Monitor System Configuration
--------------------------------
Configuration settings for the Baby Monitor System.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Person detection configuration
PERSON_DETECTION = {
    "threshold": 0.7,
    "device": "cpu",
    "model_path": os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml"),
    "upper_body_model_path": os.path.join(MODELS_DIR, "haarcascade_upperbody.xml"),
    "full_body_model_path": os.path.join(MODELS_DIR, "haarcascade_fullbody.xml"),
    "lower_body_model_path": os.path.join(MODELS_DIR, "haarcascade_lowerbody.xml"),
}

# Emotion detection configuration
EMOTION_DETECTION = {
    "threshold": 0.7,
    "device": "cpu",
    "model_path": os.path.join(MODELS_DIR, "emotion_model.pth"),
    "sample_rate": 16000,
    "chunk_size": 8000,
}

# Camera configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Web interface configuration
WEB_INTERFACE = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
}

# Logging configuration
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(BASE_DIR, "logs", "babymonitor.log"),
}

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# System configuration
SYSTEM = {
    "alert_threshold": 0.7,
    "alert_cooldown": 10,  # seconds
    "detection_interval": 0.1,  # seconds
}

# Default paths for Haar Cascade models
DEFAULT_HAAR_CASCADES = {
    "face": "haarcascade_frontalface_default.xml",
    "upper_body": "haarcascade_upperbody.xml",
    "full_body": "haarcascade_fullbody.xml",
    "lower_body": "haarcascade_lowerbody.xml",
}

# Class for accessing configuration
class Config:
    """Configuration class for the Baby Monitor System."""
    
    # Person detection
    PERSON_DETECTION = PERSON_DETECTION
    
    # Emotion detection
    EMOTION_DETECTION = EMOTION_DETECTION
    
    # Camera
    CAMERA_WIDTH = CAMERA_WIDTH
    CAMERA_HEIGHT = CAMERA_HEIGHT
    CAMERA_FPS = CAMERA_FPS
    
    # Web interface
    WEB_INTERFACE = WEB_INTERFACE
    
    # Logging
    LOGGING = LOGGING
    
    # System
    SYSTEM = SYSTEM
    
    # Paths
    BASE_DIR = BASE_DIR
    MODELS_DIR = MODELS_DIR
    
    @classmethod
    def get_haar_cascade_path(cls, cascade_name):
        """Get the path to a Haar Cascade model."""
        if cascade_name not in DEFAULT_HAAR_CASCADES:
            raise ValueError(f"Unknown Haar Cascade: {cascade_name}")
        
        # Check if the model exists in the models directory
        model_path = os.path.join(cls.MODELS_DIR, DEFAULT_HAAR_CASCADES[cascade_name])
        if os.path.exists(model_path):
            return model_path
        
        # If not, check if it's available in OpenCV's data directory
        import cv2
        opencv_path = os.path.join(cv2.data.haarcascades, DEFAULT_HAAR_CASCADES[cascade_name])
        if os.path.exists(opencv_path):
            return opencv_path
        
        raise FileNotFoundError(f"Could not find Haar Cascade model: {cascade_name}") 