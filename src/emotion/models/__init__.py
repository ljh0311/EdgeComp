"""
Emotion Recognition Models
======================
Provides a unified interface for different emotion recognition models.
"""

from .hubert_model import HuBERTEmotionDetector
from .wav2vec2_model import Wav2Vec2EmotionRecognizer
from .basic_model import BasicEmotionRecognizer

__all__ = [
    'HuBERTEmotionDetector',
    'Wav2Vec2EmotionRecognizer',
    'BasicEmotionRecognizer'
]

# Model capabilities summary:
# 1. HuBERTEmotionDetector:
#    - Uses HuBERT transformer model
#    - 6 emotions: Natural, Anger, Worried, Happy, Fear, Sadness
#    - Best accuracy but most resource intensive
#
# 2. Wav2Vec2EmotionRecognizer:
#    - Uses Wav2Vec2 transformer model
#    - 8 emotions: angry, calm, disgust, fearful, happy, neutral, sad, surprised
#    - Good balance of accuracy and resource usage
#
# 3. BasicEmotionRecognizer:
#    - Uses simple feedforward neural network
#    - 4 emotions: angry, happy, sad, neutral
#    - Fastest and most lightweight, but less accurate 