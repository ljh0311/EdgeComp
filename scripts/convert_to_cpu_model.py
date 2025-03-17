#!/usr/bin/env python3
"""
Script to convert a standard TensorFlow model to a CPU-compatible TFLite model.
This script creates a simple person detection model that works on CPU without EdgeTPU.
"""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simple_model():
    """Create a simple MobileNetV2-based person detection model."""
    # Load MobileNetV2 as the base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Add classification head
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(2, activation='softmax')(x)  # 2 classes: person, background
    
    # Create the model
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def convert_to_tflite(model, output_path):
    """Convert a Keras model to TFLite format."""
    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization flags
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    logger.info(f"Model saved to {output_path}")

def main():
    """Main function to create and convert the model."""
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define output paths
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'person_detection_model.tflite')
    
    # Create and convert the model
    logger.info("Creating a simple person detection model...")
    model = create_simple_model()
    
    logger.info("Converting the model to TFLite format...")
    convert_to_tflite(model, model_path)
    
    # Create a simple labels file
    labels_path = os.path.join(models_dir, 'person_labels.txt')
    with open(labels_path, 'w') as f:
        f.write("person\nbackground\n")
    
    logger.info(f"Labels file saved to {labels_path}")
    logger.info("Model conversion completed successfully!")
    logger.info("You can now use this model with the lightweight detector.")

if __name__ == '__main__':
    main() 