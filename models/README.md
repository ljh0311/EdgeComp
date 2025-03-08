# Model Files

This directory contains the trained models used by the Baby Monitor System.

## Required Models

1. `hubert-base-ls960_emotion.pt` - Emotion detection model (HuBERT-based)
   - Used for detecting emotions in baby sounds
   - Trained on 6 emotion classes: Natural, Anger, Worried, Happy, Fear, Sadness
   - Model accuracy: 77.47%
   - Primary model for all audio analysis

2. `yolov8n.pt` - Person detection model (YOLOv8 nano)
   - Used for detecting people and their positions
   - Pre-trained on COCO dataset
   - Optimized for real-time detection

3. `yolov8n-pose.pt` - Pose estimation model (YOLOv8 nano)
   - Used for detecting body poses and tracking movement
   - Pre-trained on COCO keypoints
   - Used for fall detection and motion analysis

## Model Sources

- Emotion detection model: HuBERT base model fine-tuned on custom emotion dataset
- YOLOv8 models: Downloaded from Ultralytics (<https://github.com/ultralytics/yolov8>)

## Model Updates

The system now uses a unified approach for audio processing:

- All audio analysis (emotion detection, cry detection, etc.) is handled by the HuBERT model
- The model provides high accuracy emotion classification with real-time performance
- No separate models needed for cry detection as it's part of the emotion classification
