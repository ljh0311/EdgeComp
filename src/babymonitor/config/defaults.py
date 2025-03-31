"""
Default Configurations
--------------------
Default configuration values for the Baby Monitor System.
"""

import os
from pathlib import Path

# Get base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

# Default camera configuration
DEFAULT_CAMERA_CONFIG = {
    'width': 640,
    'height': 480,
    'fps': 30,
    'default_device': 0,
    'flip_horizontal': False,
    'flip_vertical': False,
    'rotation': 0
}

# Default audio configuration
DEFAULT_AUDIO_CONFIG = {
    'sample_rate': 16000,
    'chunk_size': 8000,
    'channels': 1,
    'format': 'float32',
    'device': None,  # Use system default
    'gain': 1.0
}

# Default detection configuration
DEFAULT_DETECTION_CONFIG = {
    'person': {
        'threshold': 0.7,
        'device': 'cpu',
        'model_path': str(MODELS_DIR / 'haarcascade_frontalface_default.xml'),
        'upper_body_model_path': str(MODELS_DIR / 'haarcascade_upperbody.xml'),
        'full_body_model_path': str(MODELS_DIR / 'haarcascade_fullbody.xml'),
        'lower_body_model_path': str(MODELS_DIR / 'haarcascade_lowerbody.xml')
    },
    'emotion': {
        'threshold': 0.7,
        'device': 'cpu',
        'model_path': str(MODELS_DIR / 'emotion' / 'emotion_model.pth'),
        'default_model': 'speechbrain'
    }
}

# Default web interface configuration
DEFAULT_WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'cors_origins': '*',
    'static_folder': 'static',
    'template_folder': 'templates'
}

# Default logging configuration
DEFAULT_LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'babymonitor.log',
    'max_size': 10485760,  # 10MB
    'backup_count': 5,
    'console_output': True
}

# Default system configuration
DEFAULT_SYSTEM_CONFIG = {
    'alert_threshold': 0.7,
    'alert_cooldown': 10,  # seconds
    'detection_interval': 0.1,  # seconds
    'save_detections': True,
    'save_emotions': True,
    'history_length': 1000,  # number of events to keep in memory
    'backup_interval': 3600,  # seconds
    'cleanup_interval': 86400  # seconds
} 