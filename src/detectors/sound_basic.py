"""
Basic Neural Network Emotion Detection Module
=======================================
Handles emotion detection using basic neural network model.
"""

import logging
from .sound_base import BaseSoundDetector
from ..emotion.models.sound_basic import BasicEmotionRecognizer

class BasicDetector(BaseSoundDetector):
    """Real-time emotion detector using basic neural network"""
    
    def __init__(self, config, web_app=None):
        """
        Initialize the emotion detector.
        
        Args:
            config (dict): Configuration for emotion detection
            web_app: Optional web application instance for real-time updates
        """
        super().__init__(config, web_app)
        self.model = BasicEmotionRecognizer(config['model_path'])
        self.logger.info(f"Initialized {self.model_name}")
    
    @property
    def model_name(self):
        """Return the name of the model"""
        return "Basic NN (Fast)"
    
    @property
    def supported_emotions(self):
        """Return list of supported emotions"""
        return ["angry", "happy", "sad", "neutral"]
    
    def detect(self, audio_data):
        """
        Detect emotions in audio data.
        
        Args:
            audio_data: Audio data array (sampling rate should be 16kHz)
            
        Returns:
            tuple: (emotion_label, confidence)
        """
        try:
            emotion, confidence = self.model.detect(audio_data)
            if self.web_app:
                self.web_app.emit_emotion(emotion, confidence)
            return emotion, confidence
        except Exception as e:
            self.logger.error(f"Error in emotion detection: {str(e)}")
            return None, 0.0 