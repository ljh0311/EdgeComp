#!/usr/bin/env python
"""
Initialize the basic emotion model.
This script creates a new model and saves it to disk.
"""

import os
import torch
import json
import sys
from pathlib import Path

def main():
    """Create and save a basic emotion model."""
    # Add the model directory to the path
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            'models', 'emotion', 'basic_emotion')
    sys.path.append(model_dir)
    
    # Import the model
    try:
        from model import BasicEmotionModel
        print(f"Successfully imported BasicEmotionModel from {model_dir}")
    except ImportError as e:
        print(f"Error importing BasicEmotionModel: {e}")
        return
    
    # Load configuration to get the emotions count
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        emotions = config.get('emotions', ['crying', 'laughing', 'babbling', 'silence'])
        num_emotions = len(emotions)
        print(f"Creating model with {num_emotions} emotions: {emotions}")
    else:
        num_emotions = 4
        print(f"Config not found, using default number of emotions: {num_emotions}")
    
    # Create model
    model = BasicEmotionModel(num_emotions=num_emotions)
    
    # Initialize with random weights
    model.eval()
    
    # Save model
    output_path = os.path.join(model_dir, 'model.pt')
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")
    print(f"Model state_dict has {len(model.state_dict())} layers")
    
    # Print model structure
    print("\nModel architecture:")
    print(model)
    
    # Check if model.pt exists after saving
    if os.path.exists(output_path):
        print(f"Model file exists at {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
    else:
        print(f"ERROR: Model file not created at {output_path}")

if __name__ == "__main__":
    main() 