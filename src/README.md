# Baby Monitor System Source Directory

This directory contains the source code for the Baby Monitor system, organized into a clean structure.

## Directory Structure

### Main Package: `babymonitor/`

The main application package with the following subdirectories:

- **`detectors/`**: Contains detection modules
  - `emotion_detector.py`: Sound-based emotion detection
  - `person_detector.py`: Person detection using computer vision
  - `motion_detector.py`: Motion detection
  - `detector_factory.py`: Factory for detector instantiation

- **`camera/`**: Camera modules
  - `camera_manager.py`: Manages multiple camera instances
  - `camera.py`: Basic camera operations

- **`utils/`**: Utility functions and helpers
  - `camera.py`: Basic camera operations (will be consolidated with `camera/camera.py`)
  - `model_loader.py`: Helpers for loading ML models
  - `platform_checker.py`: Checks platform compatibility
  - `system_monitor.py`: Monitors system resources

- **`web/`**: Web interface
  - `app.py`: Flask application
  - `templates/`: HTML templates
  - `static/`: CSS, JavaScript, images

- **`emotion/`**: Emotion detection models
  - `models/`: Pre-trained emotion models
  - `train_emotion_model.py`: Training script for emotion models

### Other Directories

- **`models/`**: Pre-trained ML models
- **`services/`**: Background service implementations
- **`examples/`**: Example scripts for API usage
- **`core/`**: Core functionality

## Code Organization

- **Main entry points:**
  - `babymonitor/web/app.py`: Web interface
  - `babymonitor/__init__.py`: Package initialization

- **Configuration:**
  - `config.py`: Global configuration

## Cleanup Recommendations

1. Consolidate duplicate camera implementations:
   - Move `utils/camera.py` functionality to `camera/camera.py`

2. Merge similar emotion model implementations:
   - Unify models in `emotion/models/` 

3. Remove unnecessary duplicate files in:
   - `src/src/` directory 
   - `babymonitor.egg-info/` (auto-generated)

4. Create proper setup.py for installation 