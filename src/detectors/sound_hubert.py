"""
Emotion Detection Module
=====================
Handles emotion detection using HuBERT model.
"""

import logging
from .sound_base import BaseSoundDetector
from ..emotion.models.sound_hubert import HuBERTEmotionDetector

class EmotionDetector(BaseSoundDetector):
    """Real-time emotion detector using HuBERT model"""
    
    def __init__(self, config, web_app=None):
        """
        Initialize the emotion detector.
        
        Args:
            config (dict): Configuration for emotion detection
            web_app: Optional web application instance for real-time updates
        """
        super().__init__(config, web_app)
        self.model = HuBERTEmotionDetector(config['model_path'])
        self.logger.info(f"Initialized {self.model_name}")
    
    @property
    def model_name(self):
        """Return the name of the model"""
        return "HuBERT (High Accuracy)"
    
    @property
    def supported_emotions(self):
        """Return list of supported emotions"""
        return ["Natural", "Anger", "Worried", "Happy", "Fear", "Sadness"]
    
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

    def get_emotion_level(self, emotion, confidence):
        """
        Determine alert level based on emotion and confidence.
        
        Args:
            emotion (str): Detected emotion
            confidence (float): Confidence score
            
        Returns:
            str: Alert level ('critical', 'warning', or None)
        """
        if confidence < self.config['confidence_threshold']:
            return None
            
        if emotion in ['Anger', 'Fear', 'Sadness'] and confidence > self.config['critical_threshold']:
            return 'critical'
        elif emotion in ['Worried'] or confidence > self.config['warning_threshold']:
            return 'warning'
        
        return None 