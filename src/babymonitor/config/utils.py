"""
Configuration Utilities
---------------------
Helper functions for configuration management.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Union

def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a dictionary with another dictionary
    
    Args:
        base_dict (dict): Base dictionary to update
        update_dict (dict): Dictionary with updates to apply
        
    Returns:
        dict: Updated dictionary
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_json_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON file
    
    Args:
        file_path (str or Path): Path to JSON configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        json.JSONDecodeError: If the configuration file is invalid JSON
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_config(config: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save configuration to a JSON file
    
    Args:
        config (dict): Configuration dictionary to save
        file_path (str or Path): Path to save the configuration file
        
    Raises:
        OSError: If there's an error creating the directory or writing the file
    """
    file_path = Path(file_path)
    os.makedirs(file_path.parent, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

def validate_config(config: Dict[str, Any], required_keys: Dict[str, type]) -> bool:
    """
    Validate configuration dictionary against required keys and types
    
    Args:
        config (dict): Configuration dictionary to validate
        required_keys (dict): Dictionary of required keys and their expected types
        
    Returns:
        bool: True if configuration is valid, False otherwise
        
    Example:
        required_keys = {
            'host': str,
            'port': int,
            'debug': bool
        }
    """
    for key, expected_type in required_keys.items():
        if key not in config:
            return False
        if not isinstance(config[key], expected_type):
            return False
    return True

def get_config_path(filename: str) -> Path:
    """
    Get the full path for a configuration file
    
    Args:
        filename (str): Name of the configuration file
        
    Returns:
        Path: Full path to the configuration file
    """
    # Check common configuration locations
    config_locations = [
        Path.cwd() / 'config',
        Path.home() / '.babymonitor',
        Path(__file__).parent.parent / 'config'
    ]
    
    for location in config_locations:
        file_path = location / filename
        if file_path.exists():
            return file_path
    
    # Return default location if file doesn't exist
    return config_locations[-1] / filename 