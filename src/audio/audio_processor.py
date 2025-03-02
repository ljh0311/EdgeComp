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
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(4)
        self.fc = nn.Linear(64 * 256, num_classes)  # Adjust size based on input
        
    def forward(self, x):
        # Reshape input: (batch_size, samples) -> (batch_size, 1, samples)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
            
            # Find the SteelSeries Sonar microphone
            device_index = None
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if "SteelSeries Sonar - Microphone" in device_info["name"] and device_info["maxInputChannels"] > 0:
                    device_index = i
                    self.logger.info(f"Found SteelSeries Sonar microphone at index {i}")
                    break
            
            if device_index is None:
                self.logger.warning("SteelSeries Sonar microphone not found, using default input device")
            
            self.stream = self.audio.open(
                format=audio_format,
                channels=self.config.get('channels', 1),
                rate=self.config.get('sample_rate', 16000),
                input=True,
                input_device_index=device_index,  # Use the found device or None for default
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
                # Convert to float32 and normalize
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                
                # Calculate RMS volume for better sensitivity
                rms = np.sqrt(np.mean(np.square(audio_data)))
                gain = self.config.get('gain', 5.0)  # Adjustable gain factor
                
                # Apply dynamic range compression for better visualization of loud sounds
                audio_data = np.sign(audio_data) * np.log1p(np.abs(audio_data) * gain)
                
                # Normalize to [-1, 1]
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                if not self.audio_queue.full():
                    self.audio_queue.put_nowait((audio_data, rms))
                
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
        rms_buffer = []
        
        while self.is_running:
            try:
                # Get audio data from queue
                if self.audio_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                audio_data, rms = self.audio_queue.get()
                audio_chunk = audio_data
                rms_buffer.append(rms)
                
                # Add to buffer
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                
                # Process when buffer is full
                if len(audio_buffer) >= window_size:
                    # Calculate average RMS for the window
                    avg_rms = np.mean(rms_buffer)
                    rms_threshold = self.config.get('rms_threshold', 0.1)
                    
                    # Only process if the sound is loud enough
                    if avg_rms > rms_threshold:
                        # Normalize audio
                        audio_buffer = audio_buffer[:window_size]
                        audio_normalized = audio_buffer / np.max(np.abs(audio_buffer))
                        
                        # Convert to tensor and process
                        with torch.no_grad():
                            tensor_data = torch.FloatTensor(audio_normalized).to(self.device)
                            predictions = self.model(tensor_data.unsqueeze(0))
                            probabilities = torch.softmax(predictions, dim=1)[0]
                            
                            # Get highest probability class
                            max_prob, pred_class = torch.max(probabilities, dim=0)
                            class_name = self.class_names[pred_class]
                            
                            # Send visualization data
                            if self.visualization_callback:
                                self.visualization_callback(audio_normalized)
                            
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
                    
                    # Reset buffers
                    audio_buffer = np.array([], dtype=np.float32)
                    rms_buffer = []
                
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
                    self.processing_thread = threading.Thread(target=self.process_audio)
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