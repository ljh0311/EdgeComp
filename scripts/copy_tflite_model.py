#!/usr/bin/env python3
"""
Script to copy the TFLite model from the BirdRepeller project to our models directory.
This script ensures that the necessary model files are available for the lightweight detector.
"""

import os
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to copy the TFLite model files."""
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define source and destination paths
    bird_repeller_models_dir = os.path.join(project_root, 'BirdRepeller', 'models')
    destination_models_dir = os.path.join(project_root, 'models')
    
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_models_dir, exist_ok=True)
    
    # Define the files to copy
    files_to_copy = [
        ('pet_detection_model.tflite', 'person_detection_model.tflite'),
        ('pet_labels.txt', 'person_labels.txt')
    ]
    
    # Copy the files
    for src_file, dst_file in files_to_copy:
        src_path = os.path.join(bird_repeller_models_dir, src_file)
        dst_path = os.path.join(destination_models_dir, dst_file)
        
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied {src_path} to {dst_path}")
            except Exception as e:
                logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
        else:
            logger.warning(f"Source file not found: {src_path}")
    
    # Create a simple README file in the models directory
    readme_path = os.path.join(destination_models_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("""# TensorFlow Lite Models

This directory contains TensorFlow Lite models for lightweight object detection.

## Models

- `person_detection_model.tflite`: A lightweight model for person detection
- `person_labels.txt`: Labels file for the person detection model

These models are optimized for resource-constrained devices like Raspberry Pi.
""")
    logger.info(f"Created README file at {readme_path}")
    
    logger.info("Model copying completed")

if __name__ == '__main__':
    main() 