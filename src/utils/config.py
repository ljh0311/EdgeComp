"""
Configuration Module
==================
Handles system-wide configuration settings and parameters.
"""

import os
import platform
import logging

class Config:
    # Platform detection
    IS_RASPBERRY_PI = platform.machine() in ('armv7l', 'aarch64')
    
    # Path configurations
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # Emotion detection settings
    EMOTION_DETECTION = {
        'model_path': os.path.join(MODELS_DIR, 'hubert-base-ls960_emotion.pt'),
        'confidence_threshold': 0.5,  # Minimum confidence to consider
        'critical_threshold': 0.7,    # Threshold for critical emotions
        'warning_threshold': 0.6,     # Threshold for warning emotions
        'sampling_rate': 16000,       # Required sampling rate for HuBERT
        'chunk_size': 16000,          # Audio chunk size (1 second at 16kHz)
        'device': 'cpu'               # Use CPU by default
    }
    
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30 if not IS_RASPBERRY_PI else 15  # Lower FPS for Pi
    
    # Detection settings
    PERSON_DETECTION = {
        'path': 'yolov8n.pt',  # Use YOLOv8 nano model
        'conf': 0.5,  # Confidence threshold
        'iou': 0.45,  # NMS IOU threshold
        'device': 'cpu'  # Use CPU for inference
    }
    
    POSE_DETECTION = {
        'path': os.path.join(MODELS_DIR, 'yolov8n-pose.pt'),
        'conf': 0.25,
        'iou': 0.45,
        'device': 'cpu'
    }
    
    # Motion detection settings
    MOTION_DETECTION = {
        'motion_threshold': 100,  # Threshold for motion detection
        'fall_threshold': 1.5,  # Aspect ratio change threshold for fall detection
        'frame_history': 5  # Number of frames to keep for motion analysis
    }
    
    # Alert settings
    ALERT = {
        'max_history': 10,          # Maximum number of alerts to show in history
        'duration': 8000,           # Time in ms before alert fades (increased from 5000)
        'critical_color': '#FF3D00', # Material Design Red A400
        'warning_color': '#FF9100',  # Material Design Orange A400
        'info_color': '#00B0FF',    # Material Design Light Blue A400
        'flash_interval': 500,      # Flash interval for critical alerts
        'critical_frequency': 1000,  # Hz for critical alert sound
        'warning_frequency': 500,    # Hz for warning alert sound
        'critical_duration': 500,    # ms for critical alert sound
        'warning_duration': 200,     # ms for warning alert sound
        'dark_theme': {
            'background': '#1E1E1E',    # Dark background
            'text': '#E0E0E0',          # Light text
            'accent': '#2D2D2D',        # Slightly lighter than background
            'accent_text': '#4CAF50',   # Material Design Green 500
            'foreground': '#FFFFFF'      # White text
        }
    }
    
    # UI settings
    UI = {
        'min_width': 800,
        'min_height': 600,
        'window_scale': 0.75,  # Initial window size as fraction of screen
        'dark_theme': {
            'background': '#1E1E1E',  # Darker background for better contrast
            'text': '#E0E0E0',        # Light gray text for better readability
            'accent': '#2D2D2D',      # Slightly lighter than background for depth
            'accent_text': '#4CAF50', # Material green for highlights
            'alert': '#FF5252'        # Material red for alerts
        }
    }
    
    # Performance settings
    PERFORMANCE = {
        'frame_skip': 0 if not IS_RASPBERRY_PI else 1,  # Skip frames on Pi for better performance
        'detection_interval': 1 if not IS_RASPBERRY_PI else 2,  # Seconds between detections
        'max_fps': 30 if not IS_RASPBERRY_PI else 15
    }
    
    # Logging configuration
    LOGGING = {
        'level': logging.INFO,
        'format': '%(asctime)s [%(levelname)s] %(message)s',
        'datefmt': '%H:%M:%S',
        'handlers': [
            logging.StreamHandler()
        ]
    }
    
    # Custom filter for YOLO output
    class YOLOFilter(logging.Filter):
        def filter(self, record):
            return ('Speed:' not in record.getMessage() and
                   not record.getMessage().startswith('0:'))
    
    # Audio settings
    AUDIO = {
        'format': 'paFloat32',
        'channels': 1,
        'rate': 44100,
        'chunk': 1024,
        'critical_frequency': 1000, # Hz for critical alerts
        'warning_frequency': 500,   # Hz for warning alerts
        'critical_duration': 500,   # ms for critical alerts
        'warning_duration': 200,    # ms for warning alerts
    } 