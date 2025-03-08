"""
Detectors Module
=============
Provides a unified interface for different types of detectors.
"""

from .sound_hubert import EmotionDetector as HuBERTDetector
from .sound_wav2vec2 import Wav2Vec2Detector
from .sound_basic import BasicDetector
from .vision_yolo import PersonDetector
from .motion_mog2 import MotionDetector

__all__ = [
    'HuBERTDetector',
    'Wav2Vec2Detector',
    'BasicDetector',
    'PersonDetector',
    'MotionDetector',
    'AVAILABLE_SOUND_DETECTORS'
]

# Available sound emotion detectors
AVAILABLE_SOUND_DETECTORS = {
    'hubert': {
        'name': 'HuBERT (High Accuracy)',
        'class': HuBERTDetector,
        'description': 'Best accuracy but most resource intensive'
    },
    'wav2vec2': {
        'name': 'Wav2Vec2 (Balanced)',
        'class': Wav2Vec2Detector,
        'description': 'Good balance of accuracy and resource usage'
    },
    'basic': {
        'name': 'Basic NN (Fast)',
        'class': BasicDetector,
        'description': 'Fastest and most lightweight'
    }
}

# Detector capabilities summary:
# 1. HuBERTDetector (sound_hubert.py):
#    - Uses HuBERT model for sound emotion detection
#    - 6 emotions: Natural, Anger, Worried, Happy, Fear, Sadness
#    - Best suited for real-time emotion analysis from audio
#
# 2. Wav2Vec2Detector (sound_wav2vec2.py):
#    - Uses Wav2Vec2 model for sound emotion detection
#    - 8 emotions: angry, calm, disgust, fearful, happy, neutral, sad, surprised
#    - Good balance between accuracy and speed
#
# 3. BasicDetector (sound_basic.py):
#    - Uses simple neural network for sound emotion detection
#    - 4 emotions: angry, happy, sad, neutral
#    - Fastest but less accurate
#
# 4. PersonDetector (vision_yolo.py):
#    - Uses YOLOv5 model for person detection
#    - Detects and tracks people in video frames
#    - Provides bounding boxes and confidence scores
#
# 5. MotionDetector (motion_mog2.py):
#    - Uses MOG2 algorithm for motion detection
#    - Detects rapid motion and potential falls
#    - Based on frame analysis and aspect ratios
