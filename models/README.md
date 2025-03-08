# Models Directory

This directory contains all the machine learning models used by the Baby Monitor System.

## Required Models

1. **YOLOv8n** (Person Detection)
   - File: `yolov8n.pt`
   - Downloaded automatically when running the application
   - Used for person detection in video feed

2. **HuBERT** (Emotion Recognition)
   - Files: Downloaded automatically to this directory when using HuBERT emotion detector
   - Used for audio emotion recognition
   - Models are managed by the SpeechBrain library

## Model Management

- Models are automatically downloaded when running the application
- Large model files are excluded from git via `.gitignore`
- If you need to manually download models:
  1. Run the main application: `python src/main.py`
  2. The required models will be downloaded automatically
  3. Or use the emotion recognition test script: `python src/run_emotion_recognition.py`

## Model Versions

- YOLOv8n: Latest version from Ultralytics
- HuBERT: Latest version from SpeechBrain

## Notes

- Models are stored in this directory to keep the repository organized
- The `.gitignore` file excludes `.pt`, `.pth`, and `.onnx` files to prevent large files in the repository
- If you encounter any issues with model downloads, check your internet connection and try running the application again
