"""
Configuration Module
==================
Handles system-wide configuration settings and parameters.
"""

class Config:
    # Camera settings
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    
    # Detection settings
    PERSON_DETECTION = {
        'path': 'yolov8n.pt',      # Default small model
        'conf': 0.25,              # Confidence threshold
        'iou': 0.45,               # IOU threshold
        'device': 'cpu'            # Device to run on ('cpu', 'cuda', etc)
    }
    
    POSE_DETECTION = {
        'path': 'yolov8n-pose.pt', # Default pose model
        'conf': 0.25,              # Confidence threshold
        'iou': 0.45,               # IOU threshold
        'device': 'cpu'            # Device to run on
    }
    
    # Motion detection settings
    MOTION_DETECTION = {
        'motion_threshold': 100,    # Pixel distance for rapid movement (lower = more sensitive)
        'fall_threshold': 1.5,      # Aspect ratio change for fall detection (lower = more sensitive)
    }
    
    # Alert settings
    ALERT = {
        'duration': 10000,          # Alert duration in milliseconds
        'max_history': 5,           # Maximum number of alerts to show in history
        'critical_color': '#ff3333',# Color for critical alerts
        'warning_color': '#ff9900', # Color for warning alerts
        'flash_interval': 500,      # Flash interval for critical alerts in milliseconds
        'dark_theme': {            # Theme settings for alerts
            'background': '#1a1a1a',
            'foreground': 'white',
            'accent': '#333333',
            'font_family': 'Arial',
        }
    }
    
    # UI settings
    UI = {
        'min_width': 800,
        'min_height': 600,
        'window_scale': 0.8,        # Initial window size as fraction of screen size
        'dark_theme': {
            'background': '#1a1a1a',
            'foreground': 'white',
            'accent': '#333333',
            'font_family': 'Arial',
        }
    }
    
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
    
    # Logging settings
    LOGGING = {
        'level': 'INFO',
        'format': '%(asctime)s - %(levelname)s - %(message)s',
        'filename': 'baby_monitor.log'  # Changed from 'file' to 'filename'
    } 