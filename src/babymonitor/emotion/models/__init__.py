"""
Emotion Recognition Models
======================
Provides a unified interface for different emotion recognition models.
"""

from .unified_sound_detector import BaseSoundDetector
from .unified_hubert import UnifiedHuBERTDetector
from .unified_wav2vec2 import UnifiedWav2Vec2Detector
from .unified_basic import UnifiedBasicDetector

__all__ = [
    'BaseSoundDetector',
    'UnifiedHuBERTDetector',
    'UnifiedWav2Vec2Detector',
    'UnifiedBasicDetector',
    'AVAILABLE_DETECTORS'
]

# Available emotion detectors
AVAILABLE_DETECTORS = {
    'hubert': {
        'name': 'HuBERT (High Accuracy)',
        'class': UnifiedHuBERTDetector,
        'description': 'Best accuracy but most resource intensive',
        'edge_device': False
    },
    'wav2vec2': {
        'name': 'Wav2Vec2 (Edge Optimized)',
        'class': UnifiedWav2Vec2Detector,
        'description': 'Balanced performance, optimized for edge devices',
        'edge_device': True
    },
    'basic': {
        'name': 'Basic NN (Edge Optimized)',
        'class': UnifiedBasicDetector,
        'description': 'Ultra-lightweight model for resource-constrained devices',
        'edge_device': True
    }
}

# Model capabilities and requirements:
# 1. UnifiedHuBERTDetector:
#    - 6 emotions: Natural, Anger, Worried, Happy, Fear, Sadness
#    - Memory: ~1GB RAM
#    - Best for desktop/server deployment
#
# 2. UnifiedWav2Vec2Detector (Edge Optimized):
#    - 8 emotions: angry, calm, disgust, fearful, happy, neutral, sad, surprised
#    - Memory: ~250MB RAM
#    - INT8 quantization
#    - Chunked processing
#    - Suitable for Raspberry Pi 4
#
# 3. UnifiedBasicDetector (Edge Optimized):
#    - 4 emotions: angry, happy, sad, neutral
#    - Memory: ~50MB RAM
#    - INT8 quantization
#    - Feature caching
#    - Minimal processing
#    - Ideal for Raspberry Pi 3/400 