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
                           QLabel, QPushButton, QComboBox, QMessageBox, QProgressBar)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QColor
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import logging

logger = logging.getLogger(__name__)

class EmotionDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Emotion Detection")
        self.setGeometry(100, 100, 1000, 600)  # Made window larger

        # Initialize emotion detection model
        model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Define emotions
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.emotion_colors = {
            'angry': '#FF0000',     # Red
            'disgust': '#804000',   # Brown
            'fear': '#800080',      # Purple
            'happy': '#00FF00',     # Green
            'neutral': '#0000FF',   # Blue
            'sad': '#808080',       # Gray
            'surprise': '#FFA500'   # Orange
        }
        
        # Audio parameters
        self.sample_rate = 44100  # Standard sample rate
        self.block_duration = 2  # seconds
        self.buffer = np.zeros(int(self.sample_rate * self.block_duration))
        self.is_recording = False

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

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Real-time Emotion Detection")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Current emotion display
        self.emotion_label = QLabel("Current Emotion: None")
        self.emotion_label.setAlignment(Qt.AlignCenter)
        self.emotion_label.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        layout.addWidget(self.emotion_label)

        # Create progress bars for each emotion
        self.progress_bars = {}
        for emotion in self.emotions:
            emotion_layout = QHBoxLayout()
            
            # Emotion label
            label = QLabel(emotion.capitalize())
            label.setMinimumWidth(100)
            label.setStyleSheet(f"font-weight: bold; color: {self.emotion_colors[emotion]}")
            emotion_layout.addWidget(label)
            
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
                    background-color: {self.emotion_colors[emotion]};
                }}
            """)
            emotion_layout.addWidget(progress)
            
            # Percentage label
            value_label = QLabel("0%")
            value_label.setMinimumWidth(50)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            emotion_layout.addWidget(value_label)
            
            layout.addLayout(emotion_layout)
            self.progress_bars[emotion] = (progress, value_label)

        # Controls section
        controls_layout = QHBoxLayout()
        
        # Device selection
        self.device_combo = QComboBox()
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                self.device_combo.addItem(f"{dev['name']} (Input)", i)
        controls_layout.addWidget(self.device_combo)

        # Start/Stop button
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
        controls_layout.addWidget(self.start_button)

        layout.addLayout(controls_layout)
        central_widget.setLayout(layout)

    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        # Resample the input data to 16kHz for the model
        audio_data = librosa.resample(indata[:, 0], orig_sr=self.sample_rate, target_sr=16000)
        self.buffer = np.roll(self.buffer, -len(audio_data))
        self.buffer[-len(audio_data):] = audio_data

    def update_emotion(self):
        if not self.is_recording:
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
                    progress_bar, value_label = self.progress_bars[emotion]
                    percentage = int(confidence * 100)
                    progress_bar.setValue(percentage)
                    value_label.setText(f"{percentage}%")

                # Update current emotion (highest confidence)
                max_confidence_idx = torch.argmax(predictions).item()
                self.current_emotion = self.emotions[max_confidence_idx]
                
                # Update main emotion label
                self.emotion_label.setText(f"Current Emotion: {self.current_emotion.capitalize()}")
                self.emotion_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {self.emotion_colors[self.current_emotion]}")

        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")

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
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                device=device_idx,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
            )
            self.stream.start()
            self.is_recording = True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not start audio stream: {str(e)}")
            self.is_recording = False
            self.start_button.setText("Start Recording")

    def stop_recording(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_recording = False

    def closeEvent(self, event):
        self.stop_recording()
        event.accept()

def launch_emotion_gui():
    """Launch the emotion detection GUI."""
    app = QApplication(sys.argv)
    window = EmotionDetectorGUI()
    window.show()
    return app.exec_() 