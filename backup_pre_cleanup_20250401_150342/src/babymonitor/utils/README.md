# Utilities Module

This directory contains utility modules for the Baby Monitor System.

## Overview

The utilities module provides various helper functions and classes for the Baby Monitor System, including:

- Model management and loading
- Configuration management
- System monitoring and platform checking
- Camera utilities

## Modules

### Model Management

- **model_manager.py**: Manages model paths and provides functions to get model paths.
- **model_loader.py**: Loads models asynchronously in the background to prevent memory issues and performance bottlenecks.
- **setup_models.py**: Sets up models by downloading them if they don't exist.

### Configuration

- **config.py**: Contains configuration settings for the Baby Monitor System.

### System Utilities

- **system_monitor.py**: Monitors system resources such as CPU and memory usage.
- **platform_checker.py**: Checks the platform and provides information about the system.

### Camera Utilities

- **camera.py**: Provides camera utilities for capturing frames.

## Test Scripts

- **test_model_loader.py**: Demonstrates the functionality of the ModelLoader class.
- **test_detectors.py**: Tests all detectors with a camera feed.

## Usage

### Model Loader

The ModelLoader class provides a way to load models asynchronously in the background. This is useful for loading large models without blocking the main thread.

```python
from babymonitor.utils.model_loader import ModelLoader

# Initialize model loader
model_loader = ModelLoader(max_workers=2)
model_loader.start()

# Define callback function
def model_loaded_callback(model_id, model, error):
    if error:
        print(f"Error loading model {model_id}: {error}")
    else:
        print(f"Model {model_id} loaded successfully")

# Load model asynchronously
model_loader.load_model(
    model_id="yolov8_yolov8n.pt",
    model_type="yolov8",
    model_path="models/yolov8n.pt",
    config={"threshold": 0.5, "force_cpu": False},
    callback=model_loaded_callback
)

# Wait for model to load
while not model_loader.is_model_loaded("yolov8_yolov8n.pt"):
    status = model_loader.get_loading_status("yolov8_yolov8n.pt")
    print(f"Loading status: {status['status']}, progress: {status['progress']}%")
    time.sleep(1)

# Get the loaded model
model = model_loader.get_model("yolov8_yolov8n.pt")

# Clean up
model_loader.cleanup()
```

### Configuration

The Config class provides a centralized place for all configuration settings. It can be loaded from a file or environment variables.

```python
from babymonitor.utils.config import Config

# Initialize configuration
config = Config()

# Load configuration from file
config.load_from_file("config.json")

# Get configuration value
camera_width = config.get("CAMERA_WIDTH", 640)

# Set configuration value
config.set("CAMERA_WIDTH", 1280)

# Save configuration to file
config.save_to_file("config.json")
```

### Testing Detectors

The test_detectors.py script can be used to test all detectors with a camera feed.

```bash
# Test tracker detector
python -m src.babymonitor.utils.test_detectors --detector tracker --camera 0

# Test YOLOv8 detector
python -m src.babymonitor.utils.test_detectors --detector yolov8 --model yolov8n.pt --force-cpu

# Test lightweight detector
python -m src.babymonitor.utils.test_detectors --detector lightweight --model person_detection_model.tflite

# Test motion detector
python -m src.babymonitor.utils.test_detectors --detector motion
```
