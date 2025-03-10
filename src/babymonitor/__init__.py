"""
Baby Monitor Package
==================
A comprehensive baby monitoring system with emotion recognition, 
camera monitoring, and other features.
"""

from .core.main import BabyMonitorSystem
from .camera.camera import Camera
from .audio.audio_processor import AudioProcessor
from .emotion.emotion import EmotionRecognizer
from .detectors.person_detector import PersonDetector
from .detectors.motion_detector import MotionDetector
from .web.web_app import BabyMonitorWeb

__version__ = "1.0.0"
__all__ = [
    'BabyMonitorSystem',
    'Camera',
    'AudioProcessor',
    'EmotionRecognizer',
    'PersonDetector',
    'MotionDetector',
    'BabyMonitorWeb'
]
