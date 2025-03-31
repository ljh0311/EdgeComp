"""
Configuration Package
-------------------
Central configuration management for the Baby Monitor System.
"""

import os
from pathlib import Path
import logging
import yaml
import json

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
for directory in [MODELS_DIR, LOGS_DIR, CONFIG_DIR]:
    directory.mkdir(exist_ok=True)

# Load environment-specific settings if they exist
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)

# Default configurations
from .defaults import (
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_DETECTION_CONFIG,
    DEFAULT_WEB_CONFIG,
    DEFAULT_LOGGING_CONFIG,
    DEFAULT_SYSTEM_CONFIG
)

class Config:
    """Central configuration management."""
    
    def __init__(self):
        self.reload()
    
    def reload(self):
        """Reload all configuration settings."""
        # Load configurations from files if they exist, otherwise use defaults
        self.camera = self._load_config("camera.yaml", DEFAULT_CAMERA_CONFIG)
        self.audio = self._load_config("audio.yaml", DEFAULT_AUDIO_CONFIG)
        self.detection = self._load_config("detection.yaml", DEFAULT_DETECTION_CONFIG)
        self.web = self._load_config("web.yaml", DEFAULT_WEB_CONFIG)
        self.logging = self._load_config("logging.yaml", DEFAULT_LOGGING_CONFIG)
        self.system = self._load_config("system.yaml", DEFAULT_SYSTEM_CONFIG)
        
        # Set up logging
        self._setup_logging()
    
    def _load_config(self, filename: str, defaults: dict) -> dict:
        """Load configuration from file or use defaults."""
        config_file = CONFIG_DIR / filename
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return {**defaults, **yaml.safe_load(f)}
            except Exception as e:
                logging.warning(f"Error loading {filename}: {e}. Using defaults.")
                return defaults
        return defaults
    
    def save(self):
        """Save current configuration to files."""
        configs = {
            "camera.yaml": self.camera,
            "audio.yaml": self.audio,
            "detection.yaml": self.detection,
            "web.yaml": self.web,
            "logging.yaml": self.logging,
            "system.yaml": self.system
        }
        
        for filename, config in configs.items():
            config_file = CONFIG_DIR / filename
            try:
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            except Exception as e:
                logging.error(f"Error saving {filename}: {e}")
    
    def _setup_logging(self):
        """Set up logging based on configuration."""
        log_config = self.logging
        log_file = LOGS_DIR / log_config.get('file', 'babymonitor.log')
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

# Create global configuration instance
config = Config()

# Export commonly used paths and configurations
CAMERA_WIDTH = config.camera.get('width', 640)
CAMERA_HEIGHT = config.camera.get('height', 480)
CAMERA_FPS = config.camera.get('fps', 30)

PERSON_DETECTION = config.detection.get('person', {})
EMOTION_DETECTION = config.detection.get('emotion', {})

WEB_INTERFACE = config.web
SYSTEM = config.system 