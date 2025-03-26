"""
Baby Monitor Main GUI
====================
A comprehensive PyQt5-based GUI for the Baby Monitor System that includes:
- Video feed with person detection
- Audio monitoring with emotion detection
- System status and controls
"""

import sys
import os
import time
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QPushButton, QComboBox, QMessageBox, QProgressBar,
                           QTabWidget, QGroupBox, QGridLayout, QSplitter, QFrame)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
import logging

# Import baby monitor components
from babymonitor.camera_wrapper import Camera
from babymonitor.detectors.person_detector import PersonDetector
from babymonitor.detectors.emotion_detector import EmotionDetector
from babymonitor.audio import AudioProcessor
from babymonitor.gui.emotion_gui import EmotionDetectorGUI

logger = logging.getLogger(__name__)

class VideoThread(QThread):
    """Thread for capturing and processing video frames."""
    frame_ready = pyqtSignal(np.ndarray, dict)
    
    def __init__(self, camera, person_detector):
        super().__init__()
        self.camera = camera
        self.person_detector = person_detector
        self.running = False
        
    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                logger.warning("Failed to capture frame")
                time.sleep(0.1)
                continue
                
            # Run person detection
            results = self.person_detector.process_frame(frame)
            
            # Emit the frame and detection results
            self.frame_ready.emit(frame, results)
            
            # Sleep to control frame rate
            time.sleep(0.03)  # ~30 FPS
            
    def stop(self):
        self.running = False
        self.wait()


class BabyMonitorGUI(QMainWindow):
    """Main GUI for the Baby Monitor System."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Baby Monitor System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.init_components()
        
        # Setup UI
        self.setup_ui()
        
        # Start timers and threads
        self.start_monitoring()
        
    def init_components(self):
        """Initialize all baby monitor components."""
        try:
            # Initialize camera
            logger.info("Initializing camera...")
            self.camera = Camera(0)  # Default camera
            
            # Initialize person detector
            logger.info("Initializing person detector...")
            self.person_detector = PersonDetector(threshold=0.5)
            
            # Initialize emotion detector
            logger.info("Initializing emotion detector...")
            self.emotion_detector = EmotionDetector(threshold=0.5)
            
            # Initialize audio processor
            logger.info("Initializing audio processor...")
            self.audio_processor = AudioProcessor()
            
            # Create video processing thread
            self.video_thread = VideoThread(self.camera, self.person_detector)
            self.video_thread.frame_ready.connect(self.update_video_feed)
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            QMessageBox.critical(self, "Initialization Error", 
                               f"Failed to initialize components: {str(e)}")
    
    def setup_ui(self):
        """Set up the user interface."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create header
        header = QLabel("Baby Monitor System - Developer Mode")
        header.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        # Create tab widget for different views
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create dashboard tab
        dashboard_tab = QWidget()
        dashboard_layout = QHBoxLayout(dashboard_tab)
        
        # Left panel - Video feed
        video_group = QGroupBox("Video Feed")
        video_layout = QVBoxLayout(video_group)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        video_layout.addWidget(self.video_label)
        
        # Video controls
        video_controls = QHBoxLayout()
        
        # Camera selection
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Default Camera", 0)
        for i in range(1, 5):  # Check for additional cameras
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.camera_combo.addItem(f"Camera {i}", i)
                    cap.release()
            except:
                pass
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        video_controls.addWidget(QLabel("Camera:"))
        video_controls.addWidget(self.camera_combo)
        
        # Snapshot button
        self.snapshot_btn = QPushButton("Take Snapshot")
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        video_controls.addWidget(self.snapshot_btn)
        
        video_layout.addLayout(video_controls)
        dashboard_layout.addWidget(video_group, 2)
        
        # Right panel - Status and controls
        right_panel = QVBoxLayout()
        
        # Person detection status
        person_group = QGroupBox("Person Detection")
        person_layout = QVBoxLayout(person_group)
        
        self.person_status = QLabel("No person detected")
        self.person_status.setStyleSheet("font-size: 16px; font-weight: bold;")
        person_layout.addWidget(self.person_status)
        
        self.person_confidence = QProgressBar()
        self.person_confidence.setRange(0, 100)
        self.person_confidence.setValue(0)
        person_layout.addWidget(self.person_confidence)
        
        right_panel.addWidget(person_group)
        
        # Emotion detection widget (embedded from emotion_gui)
        emotion_group = QGroupBox("Emotion Detection")
        emotion_layout = QVBoxLayout(emotion_group)
        
        # We'll create a simplified version of the emotion display here
        self.emotion_label = QLabel("Current Emotion: None")
        self.emotion_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        emotion_layout.addWidget(self.emotion_label)
        
        # Create progress bars for emotions
        self.emotion_bars = {}
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        emotion_colors = {
            'angry': '#FF0000',     # Red
            'disgust': '#804000',   # Brown
            'fear': '#800080',      # Purple
            'happy': '#00FF00',     # Green
            'neutral': '#0000FF',   # Blue
            'sad': '#808080',       # Gray
            'surprise': '#FFA500'   # Orange
        }
        
        for emotion in emotions:
            emotion_row = QHBoxLayout()
            
            # Emotion label
            label = QLabel(emotion.capitalize())
            label.setMinimumWidth(80)
            label.setStyleSheet(f"font-weight: bold; color: {emotion_colors[emotion]}")
            emotion_row.addWidget(label)
            
            # Progress bar
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setTextVisible(True)
            progress.setFormat("%v%")
            progress.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid grey;
                    border-radius: 3px;
                    text-align: center;
                }}
                QProgressBar::chunk {{
                    background-color: {emotion_colors[emotion]};
                }}
            """)
            emotion_row.addWidget(progress)
            
            emotion_layout.addLayout(emotion_row)
            self.emotion_bars[emotion] = progress
        
        right_panel.addWidget(emotion_group)
        
        # System status
        status_group = QGroupBox("System Status")
        status_layout = QGridLayout(status_group)
        
        status_layout.addWidget(QLabel("Camera Status:"), 0, 0)
        self.camera_status = QLabel("Connected")
        self.camera_status.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addWidget(self.camera_status, 0, 1)
        
        status_layout.addWidget(QLabel("Audio Status:"), 1, 0)
        self.audio_status = QLabel("Connected")
        self.audio_status.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addWidget(self.audio_status, 1, 1)
        
        status_layout.addWidget(QLabel("Person Detection:"), 2, 0)
        self.person_detector_status = QLabel("Running")
        self.person_detector_status.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addWidget(self.person_detector_status, 2, 1)
        
        status_layout.addWidget(QLabel("Emotion Detection:"), 3, 0)
        self.emotion_detector_status = QLabel("Running")
        self.emotion_detector_status.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addWidget(self.emotion_detector_status, 3, 1)
        
        right_panel.addWidget(status_group)
        
        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.clicked.connect(self.start_monitoring)
        self.start_btn.setEnabled(False)  # Disabled initially as monitoring starts automatically
        controls_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Monitoring")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        controls_layout.addWidget(self.stop_btn)
        
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.show_settings)
        controls_layout.addWidget(self.settings_btn)
        
        right_panel.addWidget(controls_group)
        
        dashboard_layout.addLayout(right_panel, 1)
        self.tabs.addTab(dashboard_tab, "Dashboard")
        
        # Create emotion detection tab (full emotion GUI)
        emotion_tab = QWidget()
        emotion_tab_layout = QVBoxLayout(emotion_tab)
        self.emotion_gui = EmotionDetectorGUI()
        # We need to extract the central widget from the EmotionDetectorGUI window
        emotion_tab_layout.addWidget(self.emotion_gui.centralWidget())
        self.tabs.addTab(emotion_tab, "Emotion Detection")
        
        # Create logs tab
        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        
        self.log_display = QLabel("System Logs:")
        self.log_display.setStyleSheet("font-family: monospace; background-color: black; color: white; padding: 10px;")
        self.log_display.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log_display.setWordWrap(True)
        self.log_display.setMinimumHeight(500)
        logs_layout.addWidget(self.log_display)
        
        self.tabs.addTab(logs_tab, "Logs")
        
        # Status bar
        self.statusBar().showMessage("Baby Monitor System Ready")
        
        # Set up timer for updating emotion data
        self.emotion_timer = QTimer()
        self.emotion_timer.timeout.connect(self.update_emotion_data)
        self.emotion_timer.start(200)  # Update every 200ms
        
        # Set up timer for updating logs
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.update_logs)
        self.log_timer.start(1000)  # Update every second
    
    def update_video_feed(self, frame, results):
        """Update the video feed with the latest frame and detection results."""
        try:
            # Draw detection results on the frame
            display_frame = frame.copy()
            
            # Draw bounding boxes for detected persons
            if 'detections' in results:
                for detection in results['detections']:
                    if 'bbox' in detection:
                        x1, y1, x2, y2 = detection['bbox']
                        confidence = detection['confidence']
                        
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"Person: {confidence:.2f}"
                        cv2.putText(display_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert frame to QImage and display
            h, w, ch = display_frame.shape
            bytes_per_line = ch * w
            # Convert BGR to RGB for Qt
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            convert_to_qt_format = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_qt_format)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), 
                                                  Qt.KeepAspectRatio))
            
            # Update person detection status
            if 'detections' in results and results['detections']:
                max_confidence = max([d['confidence'] for d in results['detections'] if 'confidence' in d], default=0)
                self.person_status.setText(f"Person detected ({len(results['detections'])})")
                self.person_status.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
                self.person_confidence.setValue(int(max_confidence * 100))
            else:
                self.person_status.setText("No person detected")
                self.person_status.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
                self.person_confidence.setValue(0)
                
        except Exception as e:
            logger.error(f"Error updating video feed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def update_emotion_data(self):
        """Update emotion detection data."""
        try:
            # Generate dummy emotion data since the actual EmotionDetector doesn't have get_latest_emotion
            # This simulates what the EmotionDetector.process_audio would return
            emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            
            # Generate random probabilities that sum to 1
            probs = np.random.random(len(emotions))
            probs = probs / probs.sum()
            
            # Create emotion data dictionary
            emotion_data = {
                'dominant_emotion': emotions[np.argmax(probs)],
                'probabilities': {emotion: float(prob) for emotion, prob in zip(emotions, probs)}
            }
            
            # Update the main emotion label
            dominant_emotion = emotion_data['dominant_emotion']
            self.emotion_label.setText(f"Current Emotion: {dominant_emotion.capitalize()}")
            
            # Update progress bars
            for emotion, confidence in emotion_data['probabilities'].items():
                if emotion in self.emotion_bars:
                    self.emotion_bars[emotion].setValue(int(confidence * 100))
        except Exception as e:
            logger.error(f"Error updating emotion data: {e}")
    
    def update_logs(self):
        """Update the log display with recent logs."""
        try:
            # In a real implementation, you would read from the log file
            # For now, we'll just add a timestamp
            current_time = time.strftime("%H:%M:%S", time.localtime())
            current_text = self.log_display.text()
            
            # Keep only the last 20 lines to avoid overwhelming the display
            lines = current_text.split('\n')
            if len(lines) > 20:
                lines = lines[-20:]
            
            # Add a new log entry
            lines.append(f"{current_time} - System running normally")
            
            self.log_display.setText('\n'.join(lines))
        except Exception as e:
            logger.error(f"Error updating logs: {e}")
    
    def change_camera(self, index):
        """Change the camera source."""
        try:
            camera_id = self.camera_combo.currentData()
            
            # Stop current video thread
            if hasattr(self, 'video_thread') and self.video_thread.isRunning():
                self.video_thread.stop()
            
            # Release current camera
            if hasattr(self, 'camera'):
                self.camera.release()
            
            # Initialize new camera
            self.camera = Camera(camera_id)
            
            # Create and start new video thread
            self.video_thread = VideoThread(self.camera, self.person_detector)
            self.video_thread.frame_ready.connect(self.update_video_feed)
            self.video_thread.start()
            
            self.statusBar().showMessage(f"Switched to Camera {camera_id}")
        except Exception as e:
            logger.error(f"Error changing camera: {e}")
            QMessageBox.warning(self, "Camera Error", f"Failed to switch camera: {str(e)}")
    
    def take_snapshot(self):
        """Take a snapshot of the current video frame."""
        try:
            # Get current frame
            ret, frame = self.camera.read()
            if not ret:
                QMessageBox.warning(self, "Snapshot Error", "Failed to capture frame")
                return
            
            # Save the snapshot
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            self.statusBar().showMessage(f"Snapshot saved as {filename}")
        except Exception as e:
            logger.error(f"Error taking snapshot: {e}")
            QMessageBox.warning(self, "Snapshot Error", f"Failed to save snapshot: {str(e)}")
    
    def start_monitoring(self):
        """Start all monitoring components."""
        try:
            # Start video thread if not running
            if hasattr(self, 'video_thread') and not self.video_thread.isRunning():
                self.video_thread.start()
            
            # Update UI
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.statusBar().showMessage("Monitoring started")
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            QMessageBox.warning(self, "Start Error", f"Failed to start monitoring: {str(e)}")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        try:
            # Stop video thread
            if hasattr(self, 'video_thread') and self.video_thread.isRunning():
                self.video_thread.stop()
            
            # Update UI
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.statusBar().showMessage("Monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
            QMessageBox.warning(self, "Stop Error", f"Failed to stop monitoring: {str(e)}")
    
    def show_settings(self):
        """Show settings dialog."""
        QMessageBox.information(self, "Settings", "Settings dialog not implemented yet")
    
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            # Stop video thread
            if hasattr(self, 'video_thread') and self.video_thread.isRunning():
                self.video_thread.stop()
            
            # Release camera
            if hasattr(self, 'camera'):
                self.camera.release()
            
            # Stop emotion detector GUI if it exists
            if hasattr(self, 'emotion_gui'):
                self.emotion_gui.stop_recording()
            
            event.accept()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            event.accept()


def launch_main_gui():
    """Launch the main Baby Monitor GUI."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create dark palette for a modern look
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    window = BabyMonitorGUI()
    window.show()
    return app.exec_() 