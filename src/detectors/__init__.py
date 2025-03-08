"""
Detection Modules
===============
Collection of detection modules for video and audio analysis.
"""

from .vision_yolo import PersonDetector
from .motion_mog2 import MotionDetector
from ..emotion.models.unified_sound_detector import BaseSoundDetector
from ..emotion.models.unified_basic import UnifiedBasicDetector
from ..emotion.models.unified_wav2vec2 import UnifiedWav2Vec2Detector
from ..emotion.models.unified_hubert import UnifiedHuBERTDetector

__all__ = [
    'PersonDetector',
    'MotionDetector',
    'BaseSoundDetector',
    'UnifiedBasicDetector',
    'UnifiedWav2Vec2Detector',
    'UnifiedHuBERTDetector'
]

# Available emotion detection models
EMOTION_MODELS = {
    'basic': UnifiedBasicDetector,
    'wav2vec2': UnifiedWav2Vec2Detector,
    'hubert': UnifiedHuBERTDetector
}

def get_emotion_detector(model_name, config, web_app=None):
    """Get emotion detector instance by name.
    
    Args:
        model_name (str): Name of the model to use
        config (dict): Configuration dictionary
        web_app: Optional web application instance
        
    Returns:
        BaseSoundDetector: Initialized emotion detector
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in EMOTION_MODELS:
        raise ValueError(f"Unknown emotion model: {model_name}. Available models: {list(EMOTION_MODELS.keys())}")
        
    detector_class = EMOTION_MODELS[model_name]
    return detector_class(config, web_app)

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
