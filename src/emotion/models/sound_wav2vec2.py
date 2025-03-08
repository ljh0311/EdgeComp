"""
Wav2Vec2-based Emotion Recognition Model
====================================
Uses Facebook's Wav2Vec2 model for emotion recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config
import logging
import numpy as np
import queue
import threading
import sounddevice as sd

class Wav2Vec2EmotionRecognizer:
    """Real-time emotion recognition using Wav2Vec2"""
    
    EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    def __init__(self, web_app=None):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.running = False
        self.audio_queue = queue.Queue()
        self.process_thread = None
        self.stream = None
        self.web_app = web_app
        
        # Audio settings
        self.sample_rate = 16000  # Required by wav2vec2
        self.channels = 1
        self.block_size = 16000  # 1 second of audio
        
        try:
            # Initialize base wav2vec2 model
            config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", config=config)
            
            # Add custom classification head
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),  # 768 is wav2vec2-base hidden size
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, len(self.EMOTIONS))
            )
            
            self.model.to(self.device)
            self.classifier.to(self.device)
            self.model.eval()
            self.classifier.eval()
            
            self.logger.info("Wav2Vec2 emotion recognizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Wav2Vec2 emotion recognizer: {str(e)}")
            raise

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream processing"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        try:
            # Convert to mono if necessary
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata.flatten()
            
            # Normalize audio
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            
            self.audio_queue.put(audio_data)
            
        except Exception as e:
            self.logger.error(f"Error in audio callback: {str(e)}")

    def process_audio(self):
        """Process audio data and detect emotions"""
        audio_buffer = np.array([], dtype=np.float32)
        
        while self.running:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Add to buffer
                audio_buffer = np.concatenate([audio_buffer, audio_data])
                
                # Process when we have enough data
                if len(audio_buffer) >= self.block_size:
                    # Take the most recent block_size samples
                    if len(audio_buffer) > self.block_size:
                        audio_buffer = audio_buffer[-self.block_size:]
                    
                    # Process audio through model
                    with torch.no_grad():
                        # Convert to tensor
                        inputs = torch.FloatTensor(audio_buffer).unsqueeze(0).to(self.device)
                        
                        # Get wav2vec2 features
                        outputs = self.model(inputs)
                        hidden_states = outputs.last_hidden_state
                        
                        # Pool features (mean pooling)
                        pooled = torch.mean(hidden_states, dim=1)
                        
                        # Classify emotion
                        logits = self.classifier(pooled)
                        probs = F.softmax(logits, dim=1)
                        emotion_idx = torch.argmax(probs, dim=1)[0]
                        confidence = probs[0][emotion_idx].item()
                        emotion = self.EMOTIONS[emotion_idx]
                        
                        if confidence > 0.3:  # Confidence threshold
                            self.logger.info(f"Detected emotion: {emotion} ({confidence:.2f})")
                            if self.web_app:
                                self.web_app.emit_emotion(emotion, confidence)
                    
                    # Reset buffer but keep a small overlap
                    overlap = 1600  # 0.1 seconds
                    audio_buffer = audio_buffer[-overlap:] if len(audio_buffer) > overlap else np.array([], dtype=np.float32)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing audio: {str(e)}")
                continue

    def start(self):
        """Start the emotion recognizer."""
        if self.running:
            return
        
        try:
            self.running = True
            
            # Start audio stream
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
            
            self.logger.info("Wav2Vec2 emotion recognition started")
            
        except Exception as e:
            self.logger.error(f"Error starting emotion recognition: {str(e)}")
            self.running = False
            raise

    def stop(self):
        """Stop the emotion recognizer."""
        if not self.running:
            return
        
        self.running = False
        
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        self.logger.info("Wav2Vec2 emotion recognition stopped") 