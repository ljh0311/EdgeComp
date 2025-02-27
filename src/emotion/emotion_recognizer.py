"""
Emotion recognition module using wav2vec2 model.
"""

import torch
import logging
import threading
import queue
import numpy as np
import sounddevice as sd
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import os

class EmotionRecognizer:
    def __init__(self, model_path=None, web_app=None):
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
            # Use a more stable pre-trained model
            model_name = "facebook/wav2vec2-base"
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=4,  # angry, happy, neutral, sad
                ignore_mismatched_sizes=True
            )
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            
            # Load custom weights if available
            if model_path and os.path.exists(model_path):
                self.logger.info(f"Loading custom model weights from {model_path}")
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            self.model.to(self.device)
            self.model.eval()
            
            self.emotions = ['angry', 'happy', 'neutral', 'sad']
            self.logger.info("Emotion recognizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing emotion recognizer: {str(e)}")
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
            
            # Resample to 16kHz if needed
            if self.stream.samplerate != self.sample_rate:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=self.stream.samplerate,
                    target_sr=self.sample_rate
                )
            
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
                        inputs = self.processor(
                            audio_buffer,
                            sampling_rate=self.sample_rate,
                            return_tensors="pt",
                            padding=True
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                        emotion_idx = torch.argmax(probs, dim=1)[0]
                        confidence = probs[0][emotion_idx].item()
                        emotion = self.emotions[emotion_idx]
                        
                        if confidence > 0.5:  # Only emit confident predictions
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
            
            self.logger.info("Emotion recognition started")
            
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
        
        self.logger.info("Emotion recognition stopped") 