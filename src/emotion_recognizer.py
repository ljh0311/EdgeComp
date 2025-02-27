"""
Emotion Recognition Module
======================
Handles real-time emotion recognition from audio input.
"""

import os
import queue
import threading
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
import librosa


class EmotionModel(nn.Module):
    """Emotion recognition model architecture"""
    def __init__(self, input_size=768, hidden_size=256, num_emotions=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_emotions)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class EmotionRecognizer:
    """Real-time emotion recognition system"""

    EMOTIONS = ["angry", "happy", "sad", "neutral"]

    def __init__(self, model_path=None):
        """Initialize the emotion recognition system"""
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.block_size = 4000  # 250ms at 16kHz
        
        # Initialize state
        self.running = False
        self.audio_queue = queue.Queue()
        self.stream = None
        
        # Initialize model
        if model_path is None:
            model_path = str(Path(__file__).parent / "models" / "emotion_model.pt")
        self._load_model(model_path)

    def _load_model(self, model_path):
        """Load the emotion recognition model"""
        try:
            # Create model instance
            self.model = EmotionModel()
            
            # Load state dict
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                self.model.load_state_dict(state_dict)
                self.logger.info(f"Loaded emotion model from {model_path}")
            else:
                self.logger.warning(f"No model found at {model_path}, using base model")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Error loading emotion model: {str(e)}")
            raise

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream processing"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        # Normalize and convert to float32
        audio_data = indata.copy()
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        self.audio_queue.put(audio_data)

    def process_audio(self):
        """Process audio data and detect emotions"""
        while self.running:
            try:
                # Get audio data
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Extract features
                features = self._extract_features(audio_data)
                
                # Get predictions
                with torch.no_grad():
                    feature_tensor = torch.from_numpy(features).float().to(self.device)
                    if feature_tensor.ndim == 1:
                        feature_tensor = feature_tensor.unsqueeze(0)
                    
                    outputs = self.model(feature_tensor)
                    probs = F.softmax(outputs, dim=1)[0]
                    
                    # Get most likely emotion
                    emotion_idx = torch.argmax(probs).item()
                    emotion = self.EMOTIONS[emotion_idx]
                    confidence = probs[emotion_idx].item()
                    
                    # Log result
                    self.logger.debug(f"Detected emotion: {emotion} ({confidence:.2f})")
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing audio: {str(e)}")
                continue

    def _extract_features(self, audio_data):
        """Extract audio features for emotion recognition"""
        try:
            # Flatten and ensure correct shape
            audio_data = audio_data.flatten()
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=20
            ).flatten()
            
            # Extract additional features
            spectral = librosa.feature.spectral_centroid(
                y=audio_data,
                sr=self.sample_rate
            ).flatten()
            
            # Combine features
            features = np.concatenate([mfcc, spectral])
            
            # Pad or truncate to match model input size
            target_size = 768  # Model input size
            if len(features) < target_size:
                features = np.pad(features, (0, target_size - len(features)))
            else:
                features = features[:target_size]
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.zeros(768)  # Return zero features on error

    def start(self):
        """Start emotion recognition"""
        if self.running:
            return

        try:
            self.running = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.block_size,
                callback=self.audio_callback
            )
            self.stream.start()
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self.process_audio)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            self.logger.info("Emotion recognition started")
            
        except Exception as e:
            self.logger.error(f"Error starting emotion recognition: {str(e)}")
            self.running = False
            raise

    def stop(self):
        """Stop emotion recognition"""
        if not self.running:
            return

        self.running = False
        
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
            
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        self.logger.info("Emotion recognition stopped") 