"""
Configuration Module
==================
Contains configuration settings for the Baby Monitor System.
"""

import os
import platform
import logging
import torch
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

class Config:
    """Configuration settings."""
    
    # Platform and hardware detection
    IS_RASPBERRY_PI = platform.machine() in ("armv7l", "aarch64")
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cpu"  # Force CPU for Raspberry Pi
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # System-wide settings
    ENABLE_VISUALIZATION = False  # Disable visualization for Raspberry Pi
    
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Model paths
    YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
    HUBERT_MODEL_PATH = MODELS_DIR / "hubert"
    
    # Person detection settings
    PERSON_DETECTION = {
        'model_path': str(MODELS_DIR / "yolov8n.pt"),
        'confidence_threshold': 0.5,
        'device': 'cuda' if os.environ.get('USE_CUDA', '0') == '1' else 'cpu'
    }
    
    # Motion detection settings
    MOTION_DETECTION = {
        'history': 20,
        'dist2threshold': 400,
        'detect_shadows': True,
        'motion_threshold': 0.02,
        'fall_threshold': 0.1
    }
    
    # Audio processing settings
    AUDIO_PROCESSING = {
        'sample_rate': 16000,
        'channels': 1,
        'block_duration': 2,  # seconds
        'device': None,  # Use default audio device
        'format': 'paFloat32',
        'chunk_size': 1024,
        'gain': 5.0,
        'rms_threshold': 0.1,
        'alert_cooldown': 5.0,
        'analysis_window': 1.0,
        'model_path': str(MODELS_DIR / "hubert-base-ls960_emotion.pt"),  # Use HuBERT model for all audio processing
        'emotion_detection': {
            'confidence_threshold': 0.7,
            'critical_threshold': 0.8,
            'warning_threshold': 0.6,
            'device': 'cuda' if os.environ.get('USE_CUDA', '0') == '1' else 'cpu'
        }
    }
    
    # Emotion detection settings
    EMOTION_DETECTION = {
        'model_path': str(MODELS_DIR),  # Directory containing emotion models
        'confidence_threshold': 0.5,
        'critical_threshold': 0.7,
        'warning_threshold': 0.6,
        'sampling_rate': 16000,
        'chunk_size': 8000,
        'device': 'cuda' if os.environ.get('USE_CUDA', '0') == '1' else 'cpu',
        'enable_visualization': True
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
        'level': logging.INFO,
        'format': '%(asctime)s - %(levelname)s - %(message)s',
        'handlers': [
            logging.FileHandler('baby_monitor.log'),
            logging.StreamHandler()
        ]
    }

    @classmethod
    def log_system_info(cls):
        """Log system information."""
        logging.info(f"Platform: {platform.platform()}")
        logging.info(f"Python version: {platform.python_version()}")
        logging.info("Running on Raspberry Pi - CPU only mode")
