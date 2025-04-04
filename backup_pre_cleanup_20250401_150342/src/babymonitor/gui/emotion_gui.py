"""
Emotion Detection GUI Module
===========================
A PyQt5-based GUI for real-time emotion detection testing.
"""

import sys
import numpy as np
import sounddevice as sd
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QPushButton, QComboBox, QMessageBox, QProgressBar,
                           QGroupBox, QFormLayout, QSlider, QCheckBox, QFileDialog, QTabWidget)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QPainter, QColor, QIcon
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

class EmotionRecordingThread(QThread):
    """Thread for recording audio for emotion detection."""
    recording_complete = pyqtSignal(np.ndarray)
    
    def __init__(self, duration=3, sample_rate=16000, device=None):
        super().__init__()
        self.duration = duration
        self.sample_rate = sample_rate
        self.device = device
        
    def run(self):
        try:
            # Record audio
            audio_data = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                device=self.device
            )
            sd.wait()  # Wait for recording to complete
            
            # Emit recorded audio
            self.recording_complete.emit(audio_data)
        except Exception as e:
            logger.error(f"Error recording audio: {e}")

class EmotionDetectorGUI(QMainWindow):
    def __init__(self, standalone=True):
        super().__init__()
        self.setWindowTitle("Real-time Emotion Detection")
        self.setGeometry(100, 100, 1000, 600)  # Made window larger
        
        # Track if this is standalone or embedded
        self.standalone = standalone

        # Initialize emotion detection model
        try:
            model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(model_name)
            self.model.eval()  # Set to evaluation mode
            self.model_loaded = True
            
            logger.info("Emotion detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading emotion detection model: {e}")
            self.model_loaded = False
        
        # Define emotions
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Define emotion colors with friendly names for UI
        self.emotion_colors = {
            'angry': {'color': '#FF0000', 'name': 'Angry', 'icon': 'emoji-angry'},     # Red
            'disgust': {'color': '#804000', 'name': 'Disgust', 'icon': 'emoji-frown'},   # Brown
            'fear': {'color': '#800080', 'name': 'Fear', 'icon': 'emoji-dizzy'},      # Purple
            'happy': {'color': '#00FF00', 'name': 'Happy', 'icon': 'emoji-smile'},     # Green
            'neutral': {'color': '#0000FF', 'name': 'Neutral', 'icon': 'emoji-neutral'},   # Blue
            'sad': {'color': '#808080', 'name': 'Sad', 'icon': 'emoji-frown-fill'},       # Gray
            'surprise': {'color': '#FFA500', 'name': 'Surprise', 'icon': 'emoji-surprise'}   # Orange
        }
        
        # Create additional emotion mappings for baby monitoring
        self.baby_emotions = {
            'crying': {'color': '#FF0000', 'name': 'Crying', 'icon': 'emoji-frown'},
            'laughing': {'color': '#00FF00', 'name': 'Laughing', 'icon': 'emoji-smile'},
            'babbling': {'color': '#0000FF', 'name': 'Babbling', 'icon': 'chat-dots'},
            'silence': {'color': '#808080', 'name': 'Silence', 'icon': 'volume-mute'}
        }
        
        # Audio parameters
        self.sample_rate = 44100  # Standard sample rate
        self.block_duration = 2  # seconds
        self.buffer = np.zeros(int(self.sample_rate * self.block_duration))
        self.is_recording = False
        
        # Create a temporary directory for recordings
        self.temp_dir = tempfile.mkdtemp(prefix="emotion_detector_")
        self.current_recording = None
        self.recording_thread = None

        # Initialize confidence values
        self.confidences = {emotion: 0.0 for emotion in self.emotions}
        self.current_emotion = "neutral"

        # Setup GUI
        self.setup_ui()

        # Setup audio stream
        self.stream = None
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_emotion)
        self.timer.start(100)  # Update every 100ms
        
        # Connect signals
        if self.standalone:
            # Start recording by default in standalone mode
            self.start_recording()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        
        # Create a main content area with tabs
        self.tabs = QTabWidget()
        
        # Live detection tab
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)

        # Title
        title_label = QLabel("Real-time Emotion Detection")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        live_layout.addWidget(title_label)

        # Device and Controls row
        controls_layout = QHBoxLayout()
        
        # Device selection panel
        device_group = QGroupBox("Audio Input")
        device_layout = QVBoxLayout(device_group)
        
        device_selection = QHBoxLayout()
        
        # Device selection
        self.device_combo = QComboBox()
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                self.device_combo.addItem(f"{dev['name']} (Input)", i)
        device_selection.addWidget(QLabel("Microphone:"))
        device_selection.addWidget(self.device_combo)
        
        device_layout.addLayout(device_selection)
        
        # Sample rate selection
        sample_rate_layout = QHBoxLayout()
        sample_rate_layout.addWidget(QLabel("Sample Rate:"))
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["16000 Hz", "22050 Hz", "44100 Hz"])
        sample_rate_layout.addWidget(self.sample_rate_combo)
        
        device_layout.addLayout(sample_rate_layout)
        
        controls_layout.addWidget(device_group)
        
        # Recording controls
        recording_group = QGroupBox("Controls")
        recording_layout = QVBoxLayout(recording_group)
        
        # Start/Stop button
        buttons_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Recording")
        self.start_button.clicked.connect(self.toggle_recording)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        buttons_layout.addWidget(self.start_button)
        
        self.save_button = QPushButton("Save Recording")
        self.save_button.clicked.connect(self.save_recording)
        self.save_button.setEnabled(False)
        buttons_layout.addWidget(self.save_button)
        
        recording_layout.addLayout(buttons_layout)
        
        # Recording status
        self.status_label = QLabel("Ready to record")
        recording_layout.addWidget(self.status_label)
        
        controls_layout.addWidget(recording_group)
        
        live_layout.addLayout(controls_layout)

        # Current emotion display
        self.emotion_label = QLabel("Current Emotion: None")
        self.emotion_label.setAlignment(Qt.AlignCenter)
        self.emotion_label.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        live_layout.addWidget(self.emotion_label)

        # Create progress bars for each emotion
        emotion_group = QGroupBox("Emotion Analysis")
        emotion_layout = QVBoxLayout(emotion_group)
        
        self.progress_bars = {}
        for emotion in self.emotions:
            emotion_row = QHBoxLayout()
            
            # Emotion label
            label = QLabel(self.emotion_colors[emotion]['name'])
            label.setMinimumWidth(100)
            label.setStyleSheet(f"font-weight: bold; color: {self.emotion_colors[emotion]['color']}")
            emotion_row.addWidget(label)
            
            # Progress bar
            progress = QProgressBar()
            progress.setMinimum(0)
            progress.setMaximum(100)
            progress.setTextVisible(True)
            progress.setFormat("%v%")
            progress.setStyleSheet(f"""
                QProgressBar {{
                    border: 2px solid grey;
                    border-radius: 5px;
                    text-align: center;
                    height: 20px;
                }}
                QProgressBar::chunk {{
                    background-color: {self.emotion_colors[emotion]['color']};
                }}
            """)
            emotion_row.addWidget(progress)
            
            # Percentage label
            value_label = QLabel("0%")
            value_label.setMinimumWidth(50)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            emotion_row.addWidget(value_label)
            
            emotion_layout.addLayout(emotion_row)
            self.progress_bars[emotion] = (progress, value_label)
        
        live_layout.addWidget(emotion_group)
        
        # Add the live tab
        self.tabs.addTab(live_tab, "Live Detection")
        
        # Add file analysis tab
        file_tab = QWidget()
        file_layout = QVBoxLayout(file_tab)
        
        # File selection section
        file_group = QGroupBox("Audio File Selection")
        file_group_layout = QVBoxLayout(file_group)
        
        file_selection = QHBoxLayout()
        self.file_path = QLabel("No file selected")
        file_selection.addWidget(self.file_path)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_audio_file)
        file_selection.addWidget(browse_btn)
        
        file_group_layout.addLayout(file_selection)
        
        # Add controls for file processing
        file_controls = QHBoxLayout()
        
        analyze_btn = QPushButton("Analyze File")
        analyze_btn.clicked.connect(self.analyze_audio_file)
        file_controls.addWidget(analyze_btn)
        
        play_file_btn = QPushButton("Play File")
        play_file_btn.clicked.connect(self.play_audio_file)
        file_controls.addWidget(play_file_btn)
        
        file_group_layout.addLayout(file_controls)
        
        file_layout.addWidget(file_group)
        
        # Results display for file analysis
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.file_results_label = QLabel("No results available")
        results_layout.addWidget(self.file_results_label)
        
        file_layout.addWidget(results_group)
        
        # Add the file tab
        self.tabs.addTab(file_tab, "File Analysis")
        
        # Add advanced settings tab if in standalone mode
        if self.standalone:
            settings_tab = QWidget()
            settings_layout = QVBoxLayout(settings_tab)
            
            # Model settings
            model_group = QGroupBox("Model Settings")
            model_layout = QFormLayout(model_group)
            
            self.model_variants = QComboBox()
            self.model_variants.addItems([
                "wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                "wav2vec2-base-emotion",
                "emotion-wav2vec-large"
            ])
            model_layout.addRow("Model Variant:", self.model_variants)
            
            self.confidence_threshold = QSlider(Qt.Horizontal)
            self.confidence_threshold.setRange(1, 10)
            self.confidence_threshold.setValue(5)
            threshold_layout = QHBoxLayout()
            threshold_layout.addWidget(QLabel("Low"))
            threshold_layout.addWidget(self.confidence_threshold)
            threshold_layout.addWidget(QLabel("High"))
            model_layout.addRow("Confidence Threshold:", threshold_layout)
            
            settings_layout.addWidget(model_group)
            
            # Advanced options
            advanced_group = QGroupBox("Advanced Settings")
            advanced_layout = QFormLayout(advanced_group)
            
            self.use_gpu = QCheckBox()
            if torch.cuda.is_available():
                self.use_gpu.setChecked(True)
            else:
                self.use_gpu.setChecked(False)
                self.use_gpu.setEnabled(False)
            advanced_layout.addRow("Use GPU (if available):", self.use_gpu)
            
            settings_layout.addWidget(advanced_group)
            
            self.tabs.addTab(settings_tab, "Settings")
        
        layout.addWidget(self.tabs)
        central_widget.setLayout(layout)

    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        # Resample the input data to 16kHz for the model
        audio_data = librosa.resample(indata[:, 0], orig_sr=self.sample_rate, target_sr=16000)
        self.buffer = np.roll(self.buffer, -len(audio_data))
        self.buffer[-len(audio_data):] = audio_data

    def update_emotion(self):
        if not self.is_recording or not self.model_loaded:
            return

        try:
            # Process audio buffer
            audio_data = self.buffer.copy()
            
            # Skip processing if audio is too quiet
            if np.abs(audio_data).mean() < 0.01:
                return

            # Prepare input for the model
            inputs = self.feature_extractor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            # Get emotion predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                
                # Update confidences for all emotions
                for i, emotion in enumerate(self.emotions):
                    confidence = predictions[i].item()
                    self.confidences[emotion] = confidence
                    
                    # Update progress bar and label
                    if emotion in self.progress_bars:
                        progress_bar, value_label = self.progress_bars[emotion]
                        percentage = int(confidence * 100)
                        progress_bar.setValue(percentage)
                        value_label.setText(f"{percentage}%")

                # Update current emotion (highest confidence)
                max_confidence_idx = torch.argmax(predictions).item()
                self.current_emotion = self.emotions[max_confidence_idx]
                
                # Update main emotion label
                emotion_info = self.emotion_colors[self.current_emotion]
                self.emotion_label.setText(f"Current Emotion: {emotion_info['name']}")
                self.emotion_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {emotion_info['color']}")

        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
            self.start_button.setText("Stop Recording")
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
            """)
        else:
            self.stop_recording()
            self.start_button.setText("Start Recording")
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)

    def start_recording(self):
        try:
            device_idx = self.device_combo.currentData()
            
            # Get sample rate from the combobox
            sample_rate_text = self.sample_rate_combo.currentText()
            self.sample_rate = int(sample_rate_text.split()[0])
            
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                device=device_idx,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
            )
            self.stream.start()
            self.is_recording = True
            self.status_label.setText("Recording active")
            
            # Also start a recording for potential saving
            self.start_recording_for_save()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not start audio stream: {str(e)}")
            self.is_recording = False
            self.start_button.setText("Start Recording")
    
    def start_recording_for_save(self):
        """Start a separate recording that can be saved to file."""
        try:
            device_idx = self.device_combo.currentData()
            sample_rate_text = self.sample_rate_combo.currentText()
            sample_rate = int(sample_rate_text.split()[0])
            
            # Start recording in a separate thread
            self.recording_thread = EmotionRecordingThread(
                duration=10,  # Record 10 seconds by default
                sample_rate=sample_rate,
                device=device_idx
            )
            self.recording_thread.recording_complete.connect(self.on_recording_complete)
            self.recording_thread.start()
            
            # Disable save button until recording is complete
            self.save_button.setEnabled(False)
        except Exception as e:
            logger.error(f"Error starting recording for save: {e}")

    def on_recording_complete(self, audio_data):
        """Handle completion of recording for saving."""
        self.current_recording = audio_data
        self.save_button.setEnabled(True)
        logger.info("Recording for saving completed")

    def stop_recording(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_recording = False
            self.status_label.setText("Recording stopped")
            
            # If recording thread is still running, let it complete
            if self.recording_thread and self.recording_thread.isRunning():
                self.recording_thread.wait()

    def save_recording(self):
        """Save the current recording to a file."""
        if self.current_recording is None:
            QMessageBox.warning(self, "No Recording", "No recording available to save.")
            return
            
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Audio Recording", "", "WAV Files (*.wav);;All Files (*)"
            )
            
            if file_path:
                # Normalize the path if needed
                if not file_path.lower().endswith(".wav"):
                    file_path += ".wav"
                
                # Get sample rate
                sample_rate_text = self.sample_rate_combo.currentText()
                sample_rate = int(sample_rate_text.split()[0])
                
                # Save the audio file using scipy.io.wavfile or similar
                import scipy.io.wavfile as wavfile
                wavfile.write(file_path, sample_rate, self.current_recording)
                
                QMessageBox.information(self, "Success", f"Recording saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            QMessageBox.critical(self, "Save Error", f"Failed to save recording: {str(e)}")
    
    def browse_audio_file(self):
        """Open file browser to select an audio file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.ogg);;All Files (*)"
        )
        
        if file_path:
            self.file_path.setText(file_path)
    
    def analyze_audio_file(self):
        """Analyze the selected audio file for emotions."""
        file_path = self.file_path.text()
        
        if file_path == "No file selected" or not os.path.exists(file_path):
            QMessageBox.warning(self, "No File", "Please select a valid audio file first.")
            return
            
        try:
            # Load and analyze the audio file (simplified implementation)
            QMessageBox.information(self, "Analysis", "File analysis not fully implemented.")
            self.file_results_label.setText(f"Analysis of {os.path.basename(file_path)} would be shown here.")
        except Exception as e:
            logger.error(f"Error analyzing audio file: {e}")
            QMessageBox.critical(self, "Analysis Error", f"Failed to analyze file: {str(e)}")
    
    def play_audio_file(self):
        """Play the selected audio file."""
        file_path = self.file_path.text()
        
        if file_path == "No file selected" or not os.path.exists(file_path):
            QMessageBox.warning(self, "No File", "Please select a valid audio file first.")
            return
            
        try:
            # Simple alternative for playing (doesn't work in all environments)
            QMessageBox.information(self, "Playback", "File playback not fully implemented.")
        except Exception as e:
            logger.error(f"Error playing audio file: {e}")
            QMessageBox.critical(self, "Playback Error", f"Failed to play file: {str(e)}")

    def closeEvent(self, event):
        self.stop_recording()
        
        # Clean up temporary directory
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
            
        event.accept()

def launch_emotion_gui():
    """Launch the emotion detection GUI."""
    app = QApplication(sys.argv)
    window = EmotionDetectorGUI(standalone=True)
    window.show()
    return app.exec_() 