"""
Audio Feature Extraction Module
-----------------------------
Implements audio processing techniques inspired by image edge detection algorithms,
adapted for real-time audio analysis and visualization.
"""

import numpy as np
from scipy import signal
import librosa
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import threading
import queue
from dataclasses import dataclass
import time

@dataclass
class AudioFeatureConfig:
    """Configuration for audio feature extraction"""
    frame_length: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 20
    fmin: int = 20
    fmax: int = 8000
    
class AudioFeatureExtractor:
    """Extracts and visualizes audio features using edge detection inspired techniques"""
    
    def __init__(self, config: Optional[AudioFeatureConfig] = None):
        self.config = config or AudioFeatureConfig()
        self.plot_queue = queue.Queue()
        self.visualization_thread = None
        self.running = False
        
    def _sobel_temporal_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply Sobel-inspired temporal filtering to detect rapid changes in audio
        Similar to how Sobel detects edges in images, this detects 'edges' in audio signal
        """
        # Create Sobel-like kernel for temporal changes
        kernel = np.array([-1, 0, 1]) * np.array([1, 2, 1]).reshape(-1, 1) / 8
        return signal.convolve(audio, kernel, mode='valid')
    
    def _scharr_spectral_filter(self, spectrogram: np.ndarray) -> np.ndarray:
        """Apply Scharr-inspired filtering to detect spectral transitions
        Helps identify sudden changes in frequency content
        """
        # Create Scharr-like kernel for spectral changes
        kernel = np.array([3, 10, 3]) / 16
        return signal.convolve(spectrogram, kernel.reshape(-1, 1), mode='valid')
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract multiple audio features using edge detection inspired techniques"""
        features = {}
        
        # 1. Temporal edge detection (amplitude changes)
        features['temporal_edges'] = self._sobel_temporal_filter(audio)
        
        # 2. Spectral edge detection
        D = librosa.stft(audio, 
                        n_fft=self.config.frame_length,
                        hop_length=self.config.hop_length)
        mag_spec = np.abs(D)
        features['spectral_edges'] = self._scharr_spectral_filter(mag_spec)
        
        # 3. Mel-spectrogram with edge enhancement
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax
        )
        features['mel_edges'] = self._scharr_spectral_filter(mel_spec)
        
        # 4. MFCC with temporal derivatives (similar to edge detection)
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.frame_length,
            hop_length=self.config.hop_length
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        features['mfcc'] = mfcc
        features['mfcc_edges'] = mfcc_delta
        
        return features
    
    def start_visualization(self):
        """Start real-time visualization thread"""
        if self.running:
            return
            
        self.running = True
        self.visualization_thread = threading.Thread(target=self._visualization_loop)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
        
    def stop_visualization(self):
        """Stop visualization thread"""
        self.running = False
        if self.visualization_thread:
            self.visualization_thread.join()
            self.visualization_thread = None
            
    def _visualization_loop(self):
        """Main visualization loop"""
        plt.ion()  # Enable interactive plotting
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        fig.tight_layout(pad=3.0)
        
        while self.running:
            try:
                features = self.plot_queue.get_nowait()
                
                # Clear previous plots
                for ax in axes:
                    ax.clear()
                    
                # 1. Temporal edges
                axes[0].plot(features['temporal_edges'])
                axes[0].set_title('Temporal Edge Detection')
                axes[0].set_xlabel('Time')
                axes[0].set_ylabel('Amplitude Change')
                
                # 2. Spectral edges
                librosa.display.specshow(
                    librosa.amplitude_to_db(features['spectral_edges'], ref=np.max),
                    y_axis='log',
                    x_axis='time',
                    ax=axes[1]
                )
                axes[1].set_title('Spectral Edge Detection')
                
                # 3. Mel-spectrogram edges
                librosa.display.specshow(
                    librosa.amplitude_to_db(features['mel_edges'], ref=np.max),
                    y_axis='mel',
                    x_axis='time',
                    ax=axes[2]
                )
                axes[2].set_title('Mel-Spectrogram Edge Detection')
                
                # 4. MFCC edges
                librosa.display.specshow(
                    features['mfcc_edges'],
                    x_axis='time',
                    ax=axes[3]
                )
                axes[3].set_title('MFCC Edge Detection')
                
                plt.draw()
                plt.pause(0.01)
                
            except queue.Empty:
                plt.pause(0.1)
                continue
                
        plt.ioff()
        plt.close()
        
    def process_audio(self, audio: np.ndarray, sr: int):
        """Process audio and update visualization"""
        features = self.extract_features(audio, sr)
        if self.running:
            self.plot_queue.put(features)
        return features
        
class EmotionFeatureExtractor(AudioFeatureExtractor):
    """Specialized feature extractor for emotion recognition"""
    
    def __init__(self, config: Optional[AudioFeatureConfig] = None):
        super().__init__(config)
        
    def extract_emotion_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract features specifically useful for emotion recognition"""
        # Get base features
        features = self.extract_features(audio, sr)
        
        # Additional emotion-specific features
        
        # 1. Intensity changes (can indicate emotional intensity)
        intensity = librosa.feature.rms(
            y=audio,
            frame_length=self.config.frame_length,
            hop_length=self.config.hop_length
        )
        intensity_edges = self._sobel_temporal_filter(intensity[0])
        features['intensity_edges'] = intensity_edges
        
        # 2. Pitch changes (important for emotion)
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        pitch_edges = self._sobel_temporal_filter(np.nan_to_num(f0))
        features['pitch_edges'] = pitch_edges
        
        # 3. Spectral contrast (helps distinguish speech characteristics)
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=sr,
            n_fft=self.config.frame_length,
            hop_length=self.config.hop_length
        )
        features['spectral_contrast'] = contrast
        
        return features

def create_emotion_extractor(config: Optional[AudioFeatureConfig] = None) -> EmotionFeatureExtractor:
    """Factory function to create an emotion feature extractor"""
    if config is None:
        config = AudioFeatureConfig(
            frame_length=2048,    # Good balance for emotion analysis
            hop_length=512,       # 75% overlap
            n_mels=128,          # Detailed mel spectrogram
            n_mfcc=20,           # Standard for speech analysis
            fmin=20,             # Capture low frequency emotions
            fmax=8000            # Cover main speech frequencies
        )
    return EmotionFeatureExtractor(config) 