"""
Developer Tools Module
=====================
A PyQt5-based panel for development and testing tools for the Baby Monitor System.
Includes model testing, microphone testing, and other development utilities.
"""

import os
import time
import tempfile
import numpy as np
import cv2
import sounddevice as sd
import librosa
import torch
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                           QFileDialog, QComboBox, QTabWidget, QGroupBox, QProgressBar,
                           QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton,
                           QSlider, QFrame, QFormLayout, QSplitter, QMessageBox, QLineEdit)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QUrl, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon
import logging

logger = logging.getLogger(__name__)

class MicrophoneTestThread(QThread):
    """Thread for testing microphone by recording audio samples."""
    recording_completed = pyqtSignal(np.ndarray)
    
    def __init__(self, duration=3, sample_rate=16000, device_idx=None):
        super().__init__()
        self.duration = duration
        self.sample_rate = sample_rate
        self.device_idx = device_idx
        
    def run(self):
        try:
            # Record audio
            recording = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                device=self.device_idx
            )
            sd.wait()  # Wait for recording to complete
            
            # Emit the recorded audio
            self.recording_completed.emit(recording)
        except Exception as e:
            logger.error(f"Error during microphone test: {e}")

class ModelTestThread(QThread):
    """Thread for testing AI models on selected data."""
    test_completed = pyqtSignal(dict)
    progress_update = pyqtSignal(int)
    
    def __init__(self, model_type, model, test_data):
        super().__init__()
        self.model_type = model_type  # "emotion" or "person"
        self.model = model
        self.test_data = test_data
        
    def run(self):
        try:
            results = {}
            
            if self.model_type == "emotion":
                results = self.test_emotion_model()
            elif self.model_type == "person":
                results = self.test_person_model()
            
            self.test_completed.emit(results)
        except Exception as e:
            logger.error(f"Error during model testing: {e}")
            self.test_completed.emit({"error": str(e)})
    
    def test_emotion_model(self):
        """Test emotion detection model on audio data."""
        # Simulated emotion detection results
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Simulate processing time
        time.sleep(2)
        
        # Generate random probabilities that sum to 1
        probs = np.random.random(len(emotions))
        probs = probs / probs.sum()
        
        # Create emotion data dictionary
        results = {
            'dominant_emotion': emotions[np.argmax(probs)],
            'probabilities': {emotion: float(prob) for emotion, prob in zip(emotions, probs)},
            'processing_time_ms': np.random.randint(50, 200)
        }
        
        return results
    
    def test_person_model(self):
        """Test person detection model on image data."""
        # Simulated person detection results
        time.sleep(2)
        
        # Random detection results
        results = {
            'detections': [],
            'processing_time_ms': np.random.randint(30, 150)
        }
        
        # Generate 0-3 random detections
        num_detections = np.random.randint(0, 4)
        for i in range(num_detections):
            # Generate random bounding box
            x1 = np.random.randint(0, 400)
            y1 = np.random.randint(0, 300)
            width = np.random.randint(50, 200)
            height = np.random.randint(50, 200)
            
            detection = {
                'bbox': [x1, y1, x1 + width, y1 + height],
                'confidence': np.random.uniform(0.5, 0.99)
            }
            
            results['detections'].append(detection)
            
        return results

class DevToolsPanel(QWidget):
    """A panel containing developer tools for the Baby Monitor System."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()
        
        # Initialize temporary directory for test recordings
        self.temp_dir = tempfile.mkdtemp(prefix="babymonitor_")
        
        # Initialize variables
        self.current_audio = None
        self.current_image = None
        self.microphone_thread = None
        self.model_test_thread = None
        
    def setup_ui(self):
        """Set up the developer tools UI."""
        layout = QVBoxLayout(self)
        
        # Create tabs for different tools
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create tab for model testing
        self.setup_model_testing_tab()
        
        # Create tab for microphone testing
        self.setup_microphone_testing_tab()
        
        # Create tab for camera testing
        self.setup_camera_testing_tab()
        
        # Create tab for system diagnostics
        self.setup_diagnostics_tab()
        
        # Bottom status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Developer Tools Ready")
        status_layout.addWidget(self.status_label)
        layout.addLayout(status_layout)
    
    def setup_model_testing_tab(self):
        """Set up the model testing tab."""
        model_tab = QWidget()
        layout = QVBoxLayout(model_tab)
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout(model_group)
        
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Emotion Detection", "Person Detection"])
        model_layout.addRow("Model Type:", self.model_type_combo)
        
        self.model_variant_combo = QComboBox()
        self.model_variant_combo.addItems(["Default Model", "Lightweight Model", "High Accuracy Model"])
        model_layout.addRow("Model Variant:", self.model_variant_combo)
        
        # Connect model type changes to update available variants
        self.model_type_combo.currentIndexChanged.connect(self.update_model_variants)
        
        layout.addWidget(model_group)
        
        # Test data selection
        test_data_group = QGroupBox("Test Data")
        test_data_layout = QVBoxLayout(test_data_group)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        file_layout.addWidget(self.file_path_edit)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_test_file)
        file_layout.addWidget(self.browse_btn)
        
        test_data_layout.addLayout(file_layout)
        
        # Microphone input option
        self.use_mic_checkbox = QCheckBox("Use Microphone Input (for emotion detection)")
        test_data_layout.addWidget(self.use_mic_checkbox)
        
        # Webcam input option
        self.use_webcam_checkbox = QCheckBox("Use Webcam Input (for person detection)")
        test_data_layout.addWidget(self.use_webcam_checkbox)
        
        layout.addWidget(test_data_group)
        
        # Model parameters
        params_group = QGroupBox("Model Parameters")
        params_layout = QFormLayout(params_group)
        
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.1, 1.0)
        self.confidence_threshold.setValue(0.5)
        self.confidence_threshold.setSingleStep(0.05)
        params_layout.addRow("Confidence Threshold:", self.confidence_threshold)
        
        layout.addWidget(params_group)
        
        # Test controls
        controls_layout = QHBoxLayout()
        
        self.run_test_btn = QPushButton("Run Test")
        self.run_test_btn.clicked.connect(self.run_model_test)
        controls_layout.addWidget(self.run_test_btn)
        
        self.stop_test_btn = QPushButton("Stop")
        self.stop_test_btn.setEnabled(False)
        self.stop_test_btn.clicked.connect(self.stop_model_test)
        controls_layout.addWidget(self.stop_test_btn)
        
        layout.addLayout(controls_layout)
        
        # Test results
        results_group = QGroupBox("Test Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        self.tabs.addTab(model_tab, "Model Testing")
    
    def setup_microphone_testing_tab(self):
        """Set up the microphone testing tab."""
        mic_tab = QWidget()
        layout = QVBoxLayout(mic_tab)
        
        # Microphone selection
        mic_group = QGroupBox("Microphone Selection")
        mic_layout = QFormLayout(mic_group)
        
        self.mic_combo = QComboBox()
        
        # Get available input devices
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                self.mic_combo.addItem(f"{dev['name']} (Input)", i)
        
        mic_layout.addRow("Microphone Device:", self.mic_combo)
        
        # Recording duration
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 10)
        self.duration_spin.setValue(3)
        self.duration_spin.setSuffix(" seconds")
        mic_layout.addRow("Recording Duration:", self.duration_spin)
        
        # Sample rate
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["16000 Hz", "22050 Hz", "44100 Hz"])
        mic_layout.addRow("Sample Rate:", self.sample_rate_combo)
        
        layout.addWidget(mic_group)
        
        # Recording controls
        controls_layout = QHBoxLayout()
        
        self.record_btn = QPushButton("Record")
        self.record_btn.clicked.connect(self.start_microphone_test)
        controls_layout.addWidget(self.record_btn)
        
        self.stop_recording_btn = QPushButton("Stop Recording")
        self.stop_recording_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_recording_btn)
        
        self.play_btn = QPushButton("Play Recording")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_recording)
        controls_layout.addWidget(self.play_btn)
        
        layout.addLayout(controls_layout)
        
        # Recording status
        self.recording_status = QLabel("Ready to record")
        layout.addWidget(self.recording_status)
        
        # Waveform display (placeholder)
        waveform_group = QGroupBox("Audio Waveform")
        waveform_layout = QVBoxLayout(waveform_group)
        
        self.waveform_label = QLabel("No recording available")
        self.waveform_label.setAlignment(Qt.AlignCenter)
        self.waveform_label.setMinimumHeight(150)
        self.waveform_label.setStyleSheet("background-color: #252525; border-radius: 5px;")
        waveform_layout.addWidget(self.waveform_label)
        
        layout.addWidget(waveform_group)
        
        # Audio metrics
        metrics_group = QGroupBox("Audio Metrics")
        metrics_layout = QFormLayout(metrics_group)
        
        self.audio_level_bar = QProgressBar()
        self.audio_level_bar.setRange(0, 100)
        self.audio_level_bar.setValue(0)
        metrics_layout.addRow("Audio Level:", self.audio_level_bar)
        
        self.noise_level = QLabel("N/A")
        metrics_layout.addRow("Background Noise:", self.noise_level)
        
        self.frequency_range = QLabel("N/A")
        metrics_layout.addRow("Frequency Range:", self.frequency_range)
        
        layout.addWidget(metrics_group)
        
        self.tabs.addTab(mic_tab, "Microphone Testing")
    
    def setup_camera_testing_tab(self):
        """Set up the camera testing tab."""
        camera_tab = QWidget()
        layout = QVBoxLayout(camera_tab)
        
        # Camera selection
        camera_group = QGroupBox("Camera Selection")
        camera_layout = QFormLayout(camera_group)
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Default Camera", 0)
        
        # Add available cameras
        for i in range(1, 5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.camera_combo.addItem(f"Camera {i}", i)
                    cap.release()
            except:
                pass
        
        camera_layout.addRow("Camera Device:", self.camera_combo)
        
        # Resolution selection
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "800x600", "1280x720"])
        camera_layout.addRow("Resolution:", self.resolution_combo)
        
        # FPS selection
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["15 FPS", "30 FPS", "60 FPS"])
        camera_layout.addRow("Frame Rate:", self.fps_combo)
        
        layout.addWidget(camera_group)
        
        # Camera preview
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.camera_preview = QLabel("Camera preview not available")
        self.camera_preview.setAlignment(Qt.AlignCenter)
        self.camera_preview.setMinimumSize(640, 480)
        self.camera_preview.setStyleSheet("background-color: black;")
        preview_layout.addWidget(self.camera_preview)
        
        # Camera controls
        controls_layout = QHBoxLayout()
        
        self.start_preview_btn = QPushButton("Start Preview")
        self.start_preview_btn.clicked.connect(self.start_camera_preview)
        controls_layout.addWidget(self.start_preview_btn)
        
        self.stop_preview_btn = QPushButton("Stop Preview")
        self.stop_preview_btn.setEnabled(False)
        self.stop_preview_btn.clicked.connect(self.stop_camera_preview)
        controls_layout.addWidget(self.stop_preview_btn)
        
        self.take_snapshot_btn = QPushButton("Take Snapshot")
        self.take_snapshot_btn.clicked.connect(self.take_camera_snapshot)
        controls_layout.addWidget(self.take_snapshot_btn)
        
        preview_layout.addLayout(controls_layout)
        
        layout.addWidget(preview_group)
        
        # Camera metrics
        metrics_group = QGroupBox("Camera Metrics")
        metrics_layout = QFormLayout(metrics_group)
        
        self.actual_fps = QLabel("0 FPS")
        metrics_layout.addRow("Actual Frame Rate:", self.actual_fps)
        
        self.camera_latency = QLabel("0 ms")
        metrics_layout.addRow("Frame Latency:", self.camera_latency)
        
        layout.addWidget(metrics_group)
        
        self.tabs.addTab(camera_tab, "Camera Testing")
    
    def setup_diagnostics_tab(self):
        """Set up the system diagnostics tab."""
        diag_tab = QWidget()
        layout = QVBoxLayout(diag_tab)
        
        # System information
        sys_info_group = QGroupBox("System Information")
        sys_info_layout = QFormLayout(sys_info_group)
        
        self.cpu_info = QLabel("CPU Info: Not available")
        sys_info_layout.addRow(self.cpu_info)
        
        self.gpu_info = QLabel("GPU Info: Not available")
        sys_info_layout.addRow(self.gpu_info)
        
        self.memory_info = QLabel("Memory: Not available")
        sys_info_layout.addRow(self.memory_info)
        
        self.python_version = QLabel("Python Version: Not available")
        sys_info_layout.addRow(self.python_version)
        
        self.opencv_version = QLabel("OpenCV Version: Not available")
        sys_info_layout.addRow(self.opencv_version)
        
        self.pytorch_version = QLabel("PyTorch Version: Not available")
        sys_info_layout.addRow(self.pytorch_version)
        
        layout.addWidget(sys_info_group)
        
        # Performance tests
        perf_group = QGroupBox("Performance Tests")
        perf_layout = QVBoxLayout(perf_group)
        
        # Performance test buttons
        test_buttons_layout = QHBoxLayout()
        
        self.cpu_test_btn = QPushButton("CPU Performance Test")
        self.cpu_test_btn.clicked.connect(self.run_cpu_test)
        test_buttons_layout.addWidget(self.cpu_test_btn)
        
        self.gpu_test_btn = QPushButton("GPU Inference Test")
        self.gpu_test_btn.clicked.connect(self.run_gpu_test)
        test_buttons_layout.addWidget(self.gpu_test_btn)
        
        self.io_test_btn = QPushButton("I/O Performance Test")
        self.io_test_btn.clicked.connect(self.run_io_test)
        test_buttons_layout.addWidget(self.io_test_btn)
        
        perf_layout.addLayout(test_buttons_layout)
        
        # Performance results
        self.perf_results = QTextEdit()
        self.perf_results.setReadOnly(True)
        perf_layout.addWidget(self.perf_results)
        
        layout.addWidget(perf_group)
        
        # Logs and debug
        debug_group = QGroupBox("Debug Information")
        debug_layout = QVBoxLayout(debug_group)
        
        # Debug controls
        debug_controls = QHBoxLayout()
        
        self.debug_level_combo = QComboBox()
        self.debug_level_combo.addItems(["INFO", "DEBUG", "WARNING", "ERROR"])
        debug_controls.addWidget(QLabel("Log Level:"))
        debug_controls.addWidget(self.debug_level_combo)
        
        self.clear_logs_btn = QPushButton("Clear Logs")
        self.clear_logs_btn.clicked.connect(self.clear_debug_logs)
        debug_controls.addWidget(self.clear_logs_btn)
        
        debug_layout.addLayout(debug_controls)
        
        # Debug log display
        self.debug_log = QTextEdit()
        self.debug_log.setReadOnly(True)
        debug_layout.addWidget(self.debug_log)
        
        layout.addWidget(debug_group)
        
        self.tabs.addTab(diag_tab, "Diagnostics")
    
    def update_model_variants(self, index):
        """Update available model variants based on selected model type."""
        self.model_variant_combo.clear()
        
        if index == 0:  # Emotion Detection
            self.model_variant_combo.addItems([
                "wav2vec2-base-emotion",
                "emotion-wav2vec-large",
                "hubert-emotion-recognition"
            ])
        else:  # Person Detection
            self.model_variant_combo.addItems([
                "YOLOv8n",
                "YOLOv8s",
                "YOLOv8m",
                "FasterRCNN-ResNet50"
            ])
    
    def browse_test_file(self):
        """Open file browser to select test data."""
        model_type = self.model_type_combo.currentIndex()
        
        if model_type == 0:  # Emotion Detection
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Audio File", "", 
                "Audio Files (*.wav *.mp3 *.ogg);;All Files (*)"
            )
        else:  # Person Detection
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image File", "",
                "Image Files (*.jpg *.jpeg *.png);;All Files (*)"
            )
        
        if file_path:
            self.file_path_edit.setText(file_path)
            
            # Disable other input methods
            self.use_mic_checkbox.setChecked(False)
            self.use_webcam_checkbox.setChecked(False)
    
    def run_model_test(self):
        """Run a test on the selected model with the provided data."""
        # Validate inputs
        if not self.file_path_edit.text() and not self.use_mic_checkbox.isChecked() and not self.use_webcam_checkbox.isChecked():
            QMessageBox.warning(self, "Missing Input", "Please select a file or enable microphone/webcam input.")
            return
        
        self.run_test_btn.setEnabled(False)
        self.stop_test_btn.setEnabled(True)
        
        # Clear previous results
        self.results_text.clear()
        self.results_text.append("Starting model test...\n")
        
        # Get model type
        model_type = "emotion" if self.model_type_combo.currentIndex() == 0 else "person"
        
        # Create a mock model for testing purposes
        model = {"name": self.model_variant_combo.currentText(), "type": model_type}
        
        # Get test data (either file path or indicate to use live input)
        test_data = {
            "file_path": self.file_path_edit.text() if self.file_path_edit.text() else None,
            "use_mic": self.use_mic_checkbox.isChecked(),
            "use_webcam": self.use_webcam_checkbox.isChecked(),
            "confidence_threshold": self.confidence_threshold.value()
        }
        
        # Start the test thread
        self.model_test_thread = ModelTestThread(model_type, model, test_data)
        self.model_test_thread.test_completed.connect(self.display_test_results)
        self.model_test_thread.start()
    
    def stop_model_test(self):
        """Stop the current model test."""
        if self.model_test_thread and self.model_test_thread.isRunning():
            self.model_test_thread.terminate()
            self.model_test_thread.wait()
            
            self.results_text.append("Test stopped.\n")
        
        self.run_test_btn.setEnabled(True)
        self.stop_test_btn.setEnabled(False)
    
    def display_test_results(self, results):
        """Display the results of the model test."""
        self.run_test_btn.setEnabled(True)
        self.stop_test_btn.setEnabled(False)
        
        if "error" in results:
            self.results_text.append(f"Error: {results['error']}")
            return
        
        self.results_text.append(f"Test completed successfully.\n")
        self.results_text.append(f"Processing time: {results.get('processing_time_ms', 0)} ms\n")
        
        if "dominant_emotion" in results:
            # Emotion detection results
            self.results_text.append(f"Dominant emotion: {results['dominant_emotion']}\n")
            self.results_text.append("Emotion probabilities:")
            
            for emotion, prob in results['probabilities'].items():
                self.results_text.append(f"  - {emotion}: {prob:.4f}")
        
        elif "detections" in results:
            # Person detection results
            detections = results["detections"]
            self.results_text.append(f"People detected: {len(detections)}\n")
            
            for i, detection in enumerate(detections):
                bbox = detection.get("bbox", [0, 0, 0, 0])
                confidence = detection.get("confidence", 0)
                
                self.results_text.append(f"Person {i+1}:")
                self.results_text.append(f"  - Bounding box: {bbox}")
                self.results_text.append(f"  - Confidence: {confidence:.4f}")
    
    def start_microphone_test(self):
        """Start recording audio for microphone test."""
        try:
            self.record_btn.setEnabled(False)
            self.stop_recording_btn.setEnabled(True)
            self.play_btn.setEnabled(False)
            
            # Get recording parameters
            device_idx = self.mic_combo.currentData()
            duration = self.duration_spin.value()
            
            # Get sample rate from combo box text
            sample_rate_text = self.sample_rate_combo.currentText()
            sample_rate = int(sample_rate_text.split()[0])
            
            self.recording_status.setText(f"Recording... ({duration} seconds)")
            
            # Start recording thread
            self.microphone_thread = MicrophoneTestThread(duration, sample_rate, device_idx)
            self.microphone_thread.recording_completed.connect(self.recording_finished)
            self.microphone_thread.start()
            
            # Start a timer to update UI during recording
            self.recording_timer = QTimer()
            self.recording_timer.timeout.connect(self.update_recording_progress)
            self.recording_timer.start(100)  # Update every 100ms
            
            # Store start time
            self.recording_start_time = time.time()
        except Exception as e:
            logger.error(f"Error starting microphone test: {e}")
            QMessageBox.critical(self, "Recording Error", f"Failed to start recording: {str(e)}")
            self.record_btn.setEnabled(True)
            self.stop_recording_btn.setEnabled(False)
    
    def update_recording_progress(self):
        """Update the UI with recording progress."""
        if not hasattr(self, 'recording_start_time'):
            return
            
        elapsed = time.time() - self.recording_start_time
        duration = self.duration_spin.value()
        
        if elapsed < duration:
            # Update progress
            progress = int((elapsed / duration) * 100)
            self.audio_level_bar.setValue(progress)
            
            # Simulate audio level with random values
            random_level = np.random.randint(30, 90)
            self.audio_level_bar.setValue(random_level)
        else:
            # Recording should be complete
            self.recording_timer.stop()
    
    def recording_finished(self, audio_data):
        """Handle the completion of audio recording."""
        self.current_audio = audio_data
        
        self.record_btn.setEnabled(True)
        self.stop_recording_btn.setEnabled(False)
        self.play_btn.setEnabled(True)
        
        self.recording_status.setText("Recording complete")
        
        # Analyze the audio data
        self.analyze_audio(audio_data)
    
    def analyze_audio(self, audio_data):
        """Analyze recorded audio and display metrics."""
        try:
            # Calculate audio level
            audio_level = np.abs(audio_data).mean() * 100
            self.audio_level_bar.setValue(int(audio_level))
            
            # Calculate noise level (simulated)
            noise_level = np.abs(audio_data).std() * 100
            self.noise_level.setText(f"{noise_level:.2f}%")
            
            # Calculate frequency range (simulated)
            min_freq = 100  # Hz
            max_freq = 8000  # Hz
            self.frequency_range.setText(f"{min_freq}-{max_freq} Hz")
            
            # Generate a simple waveform visualization
            # (In a real implementation, you would generate an actual waveform image)
            self.waveform_label.setText("Waveform visualization would appear here")
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
    
    def play_recording(self):
        """Play the recorded audio."""
        if self.current_audio is None:
            return
            
        try:
            # Get sample rate from combo box
            sample_rate_text = self.sample_rate_combo.currentText()
            sample_rate = int(sample_rate_text.split()[0])
            
            # Play the recorded audio
            sd.play(self.current_audio, sample_rate)
            
            self.recording_status.setText("Playing recording...")
            
            # Disable play button temporarily
            self.play_btn.setEnabled(False)
            
            # Re-enable after playback
            duration = len(self.current_audio) / sample_rate
            QTimer.singleShot(int(duration * 1000), self.playback_finished)
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            QMessageBox.critical(self, "Playback Error", f"Failed to play recording: {str(e)}")
    
    def playback_finished(self):
        """Handle completion of audio playback."""
        self.play_btn.setEnabled(True)
        self.recording_status.setText("Ready")
    
    def start_camera_preview(self):
        """Start the camera preview."""
        # This would connect to the camera and show a live feed
        self.start_preview_btn.setEnabled(False)
        self.stop_preview_btn.setEnabled(True)
        
        # In a real implementation, you would start a camera thread here
        self.camera_preview.setText("Camera preview would appear here")
        
        # Update metrics (simulated)
        self.actual_fps.setText("30 FPS")
        self.camera_latency.setText("33 ms")
    
    def stop_camera_preview(self):
        """Stop the camera preview."""
        self.start_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        
        # In a real implementation, you would stop the camera thread here
        self.camera_preview.setText("Camera preview stopped")
        
        # Reset metrics
        self.actual_fps.setText("0 FPS")
        self.camera_latency.setText("0 ms")
    
    def take_camera_snapshot(self):
        """Take a snapshot from the camera."""
        # This would capture a frame from the camera
        # In a real implementation, you would capture from the active camera
        QMessageBox.information(self, "Snapshot", "Snapshot functionality not implemented")
    
    def run_cpu_test(self):
        """Run a CPU performance test."""
        self.perf_results.clear()
        self.perf_results.append("Running CPU performance test...\n")
        
        # Simulate a CPU-intensive task
        start_time = time.time()
        
        # Matrix multiplication as a simple CPU test
        size = 1000
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        c = np.dot(a, b)
        
        elapsed = time.time() - start_time
        
        self.perf_results.append(f"CPU Test completed in {elapsed:.4f} seconds\n")
        self.perf_results.append(f"Matrix multiplication ({size}x{size})")
        self.perf_results.append(f"FLOPS: {2 * size**3 / elapsed:.2f}")
    
    def run_gpu_test(self):
        """Run a GPU inference test."""
        self.perf_results.clear()
        self.perf_results.append("Running GPU inference test...\n")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.perf_results.append(f"CUDA available: {torch.cuda.get_device_name(0)}\n")
            
            # Create random tensor on GPU
            size = 1000
            a = torch.rand(size, size, device=device)
            b = torch.rand(size, size, device=device)
            
            # Warmup
            for _ in range(5):
                c = torch.matmul(a, b)
            
            # Benchmark
            start_time = time.time()
            iterations = 20
            
            for _ in range(iterations):
                c = torch.matmul(a, b)
            
            # Synchronize to wait for completion
            torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            
            self.perf_results.append(f"GPU Test completed in {elapsed:.4f} seconds")
            self.perf_results.append(f"Average time per iteration: {elapsed/iterations*1000:.2f} ms")
            self.perf_results.append(f"Estimated FLOPS: {2 * size**3 * iterations / elapsed:.2f}")
        else:
            self.perf_results.append("CUDA not available. GPU test skipped.")
    
    def run_io_test(self):
        """Run an I/O performance test."""
        self.perf_results.clear()
        self.perf_results.append("Running I/O performance test...\n")
        
        # Create a temporary file for I/O testing
        temp_file = os.path.join(self.temp_dir, "io_test.dat")
        
        # Write test
        data_size_mb = 100
        chunk_size = 1024 * 1024  # 1MB chunks
        data = b'0' * chunk_size
        
        self.perf_results.append(f"Writing {data_size_mb} MB of data...\n")
        
        start_time = time.time()
        
        with open(temp_file, 'wb') as f:
            for _ in range(data_size_mb):
                f.write(data)
        
        write_time = time.time() - start_time
        write_speed = data_size_mb / write_time
        
        self.perf_results.append(f"Write test completed in {write_time:.4f} seconds")
        self.perf_results.append(f"Write speed: {write_speed:.2f} MB/s\n")
        
        # Read test
        self.perf_results.append(f"Reading {data_size_mb} MB of data...\n")
        
        start_time = time.time()
        
        with open(temp_file, 'rb') as f:
            while f.read(chunk_size):
                pass
        
        read_time = time.time() - start_time
        read_speed = data_size_mb / read_time
        
        self.perf_results.append(f"Read test completed in {read_time:.4f} seconds")
        self.perf_results.append(f"Read speed: {read_speed:.2f} MB/s")
        
        # Clean up
        try:
            os.remove(temp_file)
        except:
            pass
    
    def clear_debug_logs(self):
        """Clear the debug log display."""
        self.debug_log.clear()
        
    def update_system_info(self):
        """Update the system information display."""
        try:
            # CPU info
            self.cpu_info.setText(f"CPU: 4 cores @ 2.5 GHz")  # Placeholder
            
            # GPU info
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.gpu_info.setText(f"GPU: {gpu_name}")
            else:
                self.gpu_info.setText("GPU: Not available")
            
            # Memory info (simulated)
            self.memory_info.setText("Memory: 8 GB total, 4 GB available")
            
            # Python version
            import platform
            self.python_version.setText(f"Python Version: {platform.python_version()}")
            
            # OpenCV version
            self.opencv_version.setText(f"OpenCV Version: {cv2.__version__}")
            
            # PyTorch version
            self.pytorch_version.setText(f"PyTorch Version: {torch.__version__}")
        except Exception as e:
            logger.error(f"Error updating system info: {e}") 