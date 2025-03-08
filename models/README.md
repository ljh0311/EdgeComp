# Models Directory

This directory contains all the machine learning models used by the Baby Monitor System.

## Required Models

1. **YOLOv8n** (Person Detection)
   - File: `yolov8n.pt`
   - Purpose: Real-time person detection and tracking
   - Features:
     - Person detection and tracking
     - Fall detection support
     - Real-time processing optimized
   - Size: ~6MB
   - Downloaded automatically when running the application

2. **HuBERT** (Emotion Recognition)
   - Directory: `hubert/`
   - Purpose: Audio-based emotion recognition
   - Features:
     - Real-time emotion classification
     - Multiple emotion detection
     - Optimized for baby sounds
   - Models managed by SpeechBrain library
   - Downloaded automatically when using HuBERT emotion detector

## Model Management

### Automatic Download
Models are automatically downloaded when:
1. Running the main application: `python src/main.py`
2. Using the emotion recognition test: `python src/run_emotion_recognition.py`

### Manual Download
If needed, you can manually download the models:

1. YOLOv8n:
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

2. HuBERT:
   - Models are managed by SpeechBrain
   - Will be downloaded automatically on first use
   - Stored in this directory under `hubert/`

## Model Versions

- YOLOv8n: Latest version from Ultralytics
- HuBERT: Latest version from SpeechBrain

## Performance Notes

- YOLOv8n is optimized for real-time processing
- Both CPU and CUDA acceleration supported
- Memory usage:
  - YOLOv8n: ~500MB RAM
  - HuBERT: ~1GB RAM

## Notes

- Models are stored in this directory to keep the repository organized
- Large model files (`.pt`, `.pth`, `.onnx`) are excluded from git
- Model paths are configured in `src/utils/config.py`
- If download fails:
  1. Check your internet connection
  2. Verify you have sufficient disk space
  3. Try running the application again
  4. Check the logs for specific error messages
