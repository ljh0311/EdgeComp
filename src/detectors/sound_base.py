"""
Base Sound Emotion Detector
========================
Base class for all sound emotion detectors to ensure compatibility.
"""

from abc import ABC, abstractmethod
import logging

class BaseSoundDetector(ABC):
    """Base class for all sound emotion detectors"""
    
    def __init__(self, config, web_app=None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.web_app = web_app
        
    @property
    @abstractmethod
    def model_name(self):
        """Return the name of the model for display"""
        pass
    
    @property
    @abstractmethod
    def supported_emotions(self):
        """Return list of emotions this model can detect"""
        pass
        
    @abstractmethod
    def detect(self, audio_data):
        """
        Detect emotions in audio data.
        
        Args:
            audio_data: Audio data array (sampling rate should be 16kHz)
            
        Returns:
            tuple: (emotion_label, confidence)
        """
        pass
    
    def get_emotion_level(self, emotion, confidence):
        """
        Determine alert level based on emotion and confidence.
        
        Args:
            emotion (str): Detected emotion
            confidence (float): Confidence score
            
        Returns:
            str: Alert level ('critical', 'warning', or None)
        """
        if confidence < self.config.get('confidence_threshold', 0.5):
            return None
            
        if emotion in ['Anger', 'Fear', 'Sadness'] and confidence > self.config.get('critical_threshold', 0.7):
            return 'critical'
        elif emotion in ['Worried'] or confidence > self.config.get('warning_threshold', 0.6):
            return 'warning'
        
        return None 