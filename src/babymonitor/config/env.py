"""
Environment Configuration
-----------------------
Handles environment-specific configuration settings for the Baby Monitor System.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def load_env_config(env_file=None):
    """
    Load environment-specific configuration from .env file
    
    Args:
        env_file (str, optional): Path to .env file. If None, searches in default locations.
        
    Returns:
        dict: Environment-specific configuration settings
    """
    # Load environment variables from .env file
    if env_file and os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        # Search for .env file in common locations
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        env_locations = [
            base_dir / '.env',
            base_dir / 'config' / '.env',
            Path.home() / '.babymonitor' / '.env'
        ]
        for env_path in env_locations:
            if env_path.exists():
                load_dotenv(env_path)
                break
    
    # Initialize environment config dictionary
    env_config = {}
    
    # Camera configuration from environment
    if os.getenv('CAMERA_DEVICE'):
        env_config.setdefault('camera', {})['default_device'] = int(os.getenv('CAMERA_DEVICE'))
    if os.getenv('CAMERA_WIDTH'):
        env_config.setdefault('camera', {})['width'] = int(os.getenv('CAMERA_WIDTH'))
    if os.getenv('CAMERA_HEIGHT'):
        env_config.setdefault('camera', {})['height'] = int(os.getenv('CAMERA_HEIGHT'))
    
    # Audio configuration from environment
    if os.getenv('AUDIO_DEVICE'):
        env_config.setdefault('audio', {})['device'] = os.getenv('AUDIO_DEVICE')
    if os.getenv('AUDIO_GAIN'):
        env_config.setdefault('audio', {})['gain'] = float(os.getenv('AUDIO_GAIN'))
    
    # Detection configuration from environment
    if os.getenv('DETECTION_DEVICE'):
        env_config.setdefault('detection', {}).setdefault('person', {})['device'] = os.getenv('DETECTION_DEVICE')
        env_config.setdefault('detection', {}).setdefault('emotion', {})['device'] = os.getenv('DETECTION_DEVICE')
    if os.getenv('DETECTION_THRESHOLD'):
        threshold = float(os.getenv('DETECTION_THRESHOLD'))
        env_config.setdefault('detection', {}).setdefault('person', {})['threshold'] = threshold
        env_config.setdefault('detection', {}).setdefault('emotion', {})['threshold'] = threshold
    
    # Web interface configuration from environment
    if os.getenv('WEB_HOST'):
        env_config.setdefault('web', {})['host'] = os.getenv('WEB_HOST')
    if os.getenv('WEB_PORT'):
        env_config.setdefault('web', {})['port'] = int(os.getenv('WEB_PORT'))
    if os.getenv('WEB_DEBUG'):
        env_config.setdefault('web', {})['debug'] = os.getenv('WEB_DEBUG').lower() == 'true'
    
    # Logging configuration from environment
    if os.getenv('LOG_LEVEL'):
        env_config.setdefault('logging', {})['level'] = os.getenv('LOG_LEVEL')
    if os.getenv('LOG_FILE'):
        env_config.setdefault('logging', {})['file'] = os.getenv('LOG_FILE')
    
    # System configuration from environment
    if os.getenv('ALERT_THRESHOLD'):
        env_config.setdefault('system', {})['alert_threshold'] = float(os.getenv('ALERT_THRESHOLD'))
    if os.getenv('ALERT_COOLDOWN'):
        env_config.setdefault('system', {})['alert_cooldown'] = int(os.getenv('ALERT_COOLDOWN'))
    if os.getenv('SAVE_DETECTIONS'):
        env_config.setdefault('system', {})['save_detections'] = os.getenv('SAVE_DETECTIONS').lower() == 'true'
    if os.getenv('SAVE_EMOTIONS'):
        env_config.setdefault('system', {})['save_emotions'] = os.getenv('SAVE_EMOTIONS').lower() == 'true'
    
    return env_config 