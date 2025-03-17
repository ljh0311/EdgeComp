#!/usr/bin/env python3
"""
Script to download a pre-trained TFLite model for person detection.
This script downloads a CPU-compatible TFLite model without EdgeTPU dependencies.
"""

import os
import sys
import logging
import requests
import zipfile
import io
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL for a pre-trained MobileNet SSD v2 model from TensorFlow Hub
MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"

def download_file(url, local_filename):
    """Download a file from a URL to a local file."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def main():
    """Main function to download and prepare the model."""
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define output paths
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'person_detection_model.tflite')
    
    # Download the model
    logger.info(f"Downloading pre-trained model from {MODEL_URL}...")
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        
        # Extract the model from the zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Find the .tflite file in the zip
            tflite_files = [f for f in zip_ref.namelist() if f.endswith('.tflite')]
            if not tflite_files:
                logger.error("No .tflite file found in the downloaded zip.")
                return
                
            # Extract the first .tflite file
            tflite_file = tflite_files[0]
            logger.info(f"Extracting {tflite_file} to {model_path}...")
            with zip_ref.open(tflite_file) as source, open(model_path, 'wb') as target:
                target.write(source.read())
                
        logger.info(f"Model saved to {model_path}")
        
        # Create a simple labels file
        labels_path = os.path.join(models_dir, 'person_labels.txt')
        
        # Create labels file with only person and background
        with open(labels_path, 'w') as f:
            f.write("person\nbackground\n")
        
        logger.info(f"Labels file saved to {labels_path}")
        logger.info("Model download completed successfully!")
        logger.info("You can now use this model with the lightweight detector.")
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return

if __name__ == '__main__':
    main() 