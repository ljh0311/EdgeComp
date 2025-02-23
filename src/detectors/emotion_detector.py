"""
Emotion Detection Module
=====================
Handles emotion detection using HuBERT model.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import logging
import numpy as np

class EmotionDetector:
    def __init__(self, config):
        """
        Initialize the emotion detector.
        
        Args:
            config (dict): Configuration for emotion detection
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Initialize feature extractor
            self.embedding = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')
            
            # Initialize HuBERT model
            self.hubert = AutoModel.from_pretrained(
                'facebook/hubert-base-ls960',
                output_hidden_states=True
            ).to(self.device)
            
            # Initialize classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(768, 128),
                nn.ReLU(),
                nn.Linear(128, 6)
            ).to(self.device)
            
            # Load trained weights
            self.classifier.load_state_dict(torch.load(
                config['model_path'],
                map_location=self.device
            ))
            
            self.emotion_labels = ["Natural", "Anger", "Worried", "Happy", "Fear", "Sadness"]
            self.logger.info("Emotion detection model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize emotion detection model: {str(e)}")
            raise
    
    def detect(self, audio_data):
        """
        Detect emotions in audio data.
        
        Args:
            audio_data: Audio data array (sampling rate should be 16kHz)
            
        Returns:
            tuple: (emotion_label, confidence)
        """
        try:
            with torch.no_grad():
                # Preprocess audio
                inputs = self.embedding(
                    audio_data,
                    sampling_rate=16000,
                    return_tensors='pt'
                ).input_values.to(self.device)
                
                # Get HuBERT features
                outputs = self.hubert(inputs.squeeze(0))
                features = outputs.last_hidden_state.mean(dim=1)
                
                # Get emotion predictions
                logits = self.classifier(features)
                probs = torch.softmax(logits, dim=1)
                
                # Get predicted emotion
                pred_idx = torch.argmax(probs, dim=1)[0]
                confidence = probs[0][pred_idx].item()
                emotion = self.emotion_labels[pred_idx]
                
                self.logger.debug(f"Detected emotion: {emotion} with confidence {confidence:.2f}")
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