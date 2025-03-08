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
        try:
            self.model = HuBERTEmotionDetector(config, web_app)
            self.logger.info(f"Initialized {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.model_name}: {str(e)}")
            raise
    
    @property
    def model_name(self):
        """Return the name of the model"""
        return "HuBERT (High Accuracy)"
    
    @property
    def supported_emotions(self):
        """Return list of supported emotions"""
        return self.model.supported_emotions if self.model else []
    
    def start(self):
        """Start the emotion detector."""
        if self.model:
            self.model.start()
            self.is_running = True
            self.logger.info(f"{self.model_name} started")
    
    def stop(self):
        """Stop the emotion detector."""
        if self.model:
            self.model.stop()
            self.is_running = False
            self.logger.info(f"{self.model_name} stopped")
    
    def detect(self, audio_data):
        """
        Detect emotions in audio data.
        
        Args:
            audio_data: Audio data array (sampling rate should be 16kHz)
            
        Returns:
            tuple: (emotion_label, confidence)
        """
        try:
            if not self.is_running or not self.model:
                return None, 0.0
                
            emotion, confidence = self.model.detect(audio_data)
            if emotion and confidence > 0:
                level = self.get_emotion_level(emotion, confidence)
                if level and self.web_app:
                    self.web_app.emit_emotion(emotion, confidence, level)
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
        if confidence < self.config.get('confidence_threshold', 0.5):
            return None
            
        if emotion in ['Anger', 'Fear', 'Sadness'] and confidence > self.config.get('critical_threshold', 0.7):
            return 'critical'
        elif emotion in ['Worried'] or confidence > self.config.get('warning_threshold', 0.6):
            return 'warning'
        
        return None 