"""
Configuration settings for the Baby Monitor System.
"""

import os
from pathlib import Path
import json
import logging
from typing import Dict, Any

class Config:
    """Configuration class for the Baby Monitor System."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "data"
        
        # Create necessary directories
        for dir_path in [self.models_dir, self.logs_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Default configuration
        self.config = {
            "camera": {
                "index": int(os.getenv("CAMERA_INDEX", "0")),
                "resolution": (640, 480),
                "fps": 30
            },
            "audio": {
                "device_index": int(os.getenv("AUDIO_DEVICE_INDEX", "0")),
                "sample_rate": 16000,
                "chunk_size": 1024
            },
            "detector": {
                "type": os.getenv("DETECTOR_TYPE", "yolov8"),
                "confidence_threshold": 0.5,
                "use_gpu": os.getenv("GPU_ENABLED", "true").lower() == "true"
            },
            "web": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False
            },
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "file": self.logs_dir / "babymonitor.log"
            }
        }
        
        # Load custom configuration if exists
        self._load_custom_config()
        
    def _load_custom_config(self):
        """Load custom configuration from config file if it exists."""
        config_file = self.base_dir / "config" / "system_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                    self._update_config(self.config, custom_config)
            except Exception as e:
                logging.error(f"Error loading custom config: {e}")
                
    def _update_config(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Recursively update configuration dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._update_config(base[key], value)
            else:
                base[key] = value
                
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            config = config.setdefault(k, {})
            
        config[keys[-1]] = value
        
    def save(self):
        """Save current configuration to file."""
        config_file = self.base_dir / "config" / "system_config.json"
        config_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving config: {e}")

# Create global configuration instance
config = Config()
