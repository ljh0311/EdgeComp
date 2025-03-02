"""
Audio Processor Module
==================
Handles real-time audio processing and sound classification for baby monitoring.
Uses PyAudio for audio capture and a pre-trained model for sound classification.
"""

import pyaudio
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import threading
import queue
import time
import logging
from pathlib import Path
import librosa
from scipy import signal
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=4):  # cry, scream, happy, background
        super().__init__()
        # Load pre-trained wav2vec model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.base_model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=num_classes
        )
        
    def forward(self, x):
        # Process audio through wav2vec and get predictions
        features = self.feature_extractor(x, sampling_rate=16000, return_tensors="pt")
        outputs = self.base_model(**features)
        return outputs.logits

class AudioProcessor:
    def __init__(self, config, alert_callback=None):
        """Initialize the audio processor with configuration and callback."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.alert_callback = alert_callback
        self.is_running = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.processing_thread = None
        self.audio_queue = queue.Queue(maxsize=100)
        self.visualization_callback = None
        self.stream_lock = threading.Lock()
        
        # Initialize state
        self.last_alert_time = 0
        
        # Initialize audio processing components
        self.device = torch.device(config.get('device', 'cpu'))
        self.setup_model()
        self._initialize_stream()
        
        # Sound classification thresholds
        self.class_names = ['cry', 'scream', 'happy', 'background']
        self.thresholds = {
            'cry': 0.6,
            'scream': 0.7,
            'happy': 0.5
        }
        
        self.logger.info("Audio processor initialized")

    def setup_model(self):
        """Setup the audio classification model."""
        try:
            # Initialize model on CPU first
            self.model = AudioClassifier()
            model_path = Path(self.config.get('model_path'))
            
            if model_path.exists():
                # Load model weights with map_location to ensure CPU loading
                state_dict = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                self.logger.info("Loaded audio model from %s", model_path)
            else:
                self.logger.warning("No pre-trained model found at %s, using base model", model_path)
            
            # Move model to specified device after loading
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            self.logger.error("Error setting up audio model: %s", str(e))
            raise

    def _initialize_stream(self):
        """Initialize the audio stream."""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            format_map = {
                'paFloat32': pyaudio.paFloat32,
                'paInt16': pyaudio.paInt16,
                'paInt32': pyaudio.paInt32
            }
            
            audio_format = format_map.get(self.config.get('format', 'paFloat32'), pyaudio.paFloat32)
            
            self.stream = self.audio.open(
                format=audio_format,
                channels=self.config.get('channels', 1),
                rate=self.config.get('sample_rate', 16000),
                input=True,
                frames_per_buffer=self.config.get('chunk_size', 1024),
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            self.logger.info("Audio stream initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing audio stream: {str(e)}")
            raise

    def set_visualization_callback(self, callback):
        """Set callback for visualization updates."""
        self.visualization_callback = callback

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream to process incoming audio data."""
        try:
            if self.is_running:
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                if not self.audio_queue.full():
                    self.audio_queue.put_nowait(audio_data)
                
                # Send data for visualization if callback is set
                if self.visualization_callback:
                    self.visualization_callback(audio_data)
                    
            return (in_data, pyaudio.paContinue)
        except Exception as e:
            self.logger.error("Error in audio callback: %s", str(e))
            return (None, pyaudio.paAbort)

    def process_audio(self):
        """Process audio data from the queue and perform sound classification."""
        window_size = int(self.config['analysis_window'] * self.config['sample_rate'])
        audio_buffer = np.array([], dtype=np.float32)
        
        while self.is_running:
            try:
                # Get audio data from queue
                if self.audio_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                data = self.audio_queue.get()
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                # Add to buffer
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                
                # Process when buffer is full
                if len(audio_buffer) >= window_size:
                    # Normalize audio
                    audio_buffer = audio_buffer[:window_size]
                    audio_normalized = audio_buffer / np.max(np.abs(audio_buffer))
                    
                    # Resample to 16kHz for wav2vec
                    audio_resampled = librosa.resample(
                        audio_normalized,
                        orig_sr=self.config['sample_rate'],
                        target_sr=16000
                    )
                    
                    # Classify sound
                    with torch.no_grad():
                        # Move tensor to same device as model
                        tensor_data = torch.FloatTensor(audio_resampled).to(self.device)
                        predictions = self.model(tensor_data)
                        probabilities = torch.softmax(predictions, dim=1)[0]
                        
                        # Get highest probability class
                        max_prob, pred_class = torch.max(probabilities, dim=0)
                        class_name = self.class_names[pred_class]
                        
                        # Check against thresholds and alert if necessary
                        current_time = time.time()
                        if (current_time - self.last_alert_time) > self.config['alert_cooldown']:
                            if class_name in self.thresholds and max_prob > self.thresholds[class_name]:
                                if class_name in ['cry', 'scream']:
                                    self.alert_callback(
                                        f"Baby {class_name} detected! (Confidence: {max_prob:.1%})",
                                        level="critical"
                                    )
                                elif class_name == 'happy':
                                    self.alert_callback(
                                        f"Happy baby sounds detected! (Confidence: {max_prob:.1%})",
                                        level="info"
                                    )
                                self.last_alert_time = current_time
                    
                    # Reset buffer
                    audio_buffer = np.array([], dtype=np.float32)
                
            except Exception as e:
                self.logger.error("Error processing audio: %s", str(e))
                time.sleep(0.1)

    def start(self):
        """Start audio processing."""
        try:
            with self.stream_lock:
                if not self.is_running:
                    self.is_running = True
                    if not self.stream or not self.stream.is_active():
                        self._initialize_stream()
                    self.processing_thread = threading.Thread(target=self._process_audio)
                    self.processing_thread.daemon = True
                    self.processing_thread.start()
                    self.logger.info("Audio processing started")
        except Exception as e:
            self.logger.error(f"Error starting audio processing: {str(e)}")
            self.is_running = False
            raise

    def stop(self):
        """Stop audio processing."""
        try:
            with self.stream_lock:
                if self.is_running:
                    self.is_running = False
                    if self.stream:
                        self.stream.stop_stream()
                        self.stream.close()
                        self.stream = None
                    if self.processing_thread:
                        self.processing_thread.join(timeout=2)
                    self.logger.info("Audio processing stopped")
        except Exception as e:
            self.logger.error(f"Error stopping audio processing: {str(e)}")

    def _process_audio(self):
        """Process audio data from the queue."""
        while self.is_running:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get_nowait()
                    
                    # Update visualization if callback is set
                    if self.visualization_callback:
                        self.visualization_callback(audio_data)
                    
                    # Process audio for alerts (implement your logic here)
                    if self.alert_callback and self._should_alert(audio_data):
                        self.alert_callback("High audio level detected", "warning")
                
                time.sleep(0.001)  # Small sleep to prevent CPU overload
                
            except Exception as e:
                self.logger.error(f"Error processing audio data: {str(e)}")
                time.sleep(0.1)  # Longer sleep on error

    def _should_alert(self, audio_data):
        """Check if audio level should trigger an alert."""
        try:
            # Simple threshold-based detection
            audio_level = np.abs(audio_data).mean()
            return audio_level > self.config.get('alert_threshold', 0.1)
        except Exception as e:
            self.logger.error(f"Error checking audio level: {str(e)}")
            return False

    def is_active(self):
        """Check if audio processing is active."""
        return self.is_running and self.stream and self.stream.is_active() 

    def __del__(self):
        """Cleanup when object is deleted."""
        try:
            self.stop()
            self.audio.terminate()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 