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
import argparse
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QPushButton, QComboBox, QMessageBox, QProgressBar,
                           QTabWidget, QGroupBox, QGridLayout, QSplitter, QFrame,
                           QLineEdit, QFileDialog, QSlider, QCheckBox, QRadioButton)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon
import logging

# Import baby monitor components
from babymonitor.camera_wrapper import Camera
from babymonitor.detectors.person_detector import PersonDetector
from babymonitor.detectors.emotion_detector import EmotionDetector
from babymonitor.audio import AudioProcessor
from babymonitor.gui.emotion_gui import EmotionDetectorGUI
from babymonitor.gui.dev_tools import DevToolsPanel

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
    
    def __init__(self, dev_mode=False):
        super().__init__()
        self.setWindowTitle("Baby Monitor System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Store developer mode flag
        self.dev_mode = dev_mode
        
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
        title = "Baby Monitor System - Developer Mode" if self.dev_mode else "Baby Monitor System"
        header = QLabel(title)
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
        
        # Right panel - Status and controls (enhanced from index.html)
        right_panel = QVBoxLayout()
        
        # Add System Status panel inspired by index.html
        system_status_group = QGroupBox("System Status")
        system_status_layout = QVBoxLayout(system_status_group)
        
        # System status items
        self.status_items = {}
        
        # Create status items inspired by index.html
        status_items_data = [
            {"icon": "clock", "label": "Uptime:", "id": "uptime", "initial": "00:00:00"},
            {"icon": "camera", "label": "Camera:", "id": "cameraStatus", "initial": "Connecting..."},
            {"icon": "person", "label": "Person Detection:", "id": "personDetectorStatus", "initial": "Initializing..."},
            {"icon": "emoji-smile", "label": "Emotion Detection:", "id": "emotionDetectorStatus", "initial": "Initializing..."},
            {"icon": "cpu", "label": "CPU Usage:", "id": "cpuUsage", "initial": "0%", "progress": True},
            {"icon": "memory", "label": "Memory Usage:", "id": "memoryUsage", "initial": "0%", "progress": True}
        ]
        
        for item in status_items_data:
            item_layout = QHBoxLayout()
            
            # Label with icon
            label = QLabel(item["label"])
            label.setStyleSheet("font-weight: bold;")
            item_layout.addWidget(label)
            
            # Value (either text or progress bar)
            if item.get("progress", False):
                progress = QProgressBar()
                progress.setRange(0, 100)
                progress.setValue(0)
                progress.setFormat("%v%")
                progress.setTextVisible(True)
                self.status_items[item["id"]] = progress
                item_layout.addWidget(progress)
            else:
                value = QLabel(item["initial"])
                self.status_items[item["id"]] = value
                item_layout.addWidget(value)
            
            system_status_layout.addLayout(item_layout)
        
        right_panel.addWidget(system_status_group)
        
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
        
        # Add Quick Actions panel from index.html
        if not self.dev_mode:  # Only show in normal mode
            quick_actions_group = QGroupBox("Quick Actions")
            quick_actions_layout = QVBoxLayout(quick_actions_group)
            
            # Add quick action buttons
            actions = [
                {"icon": "camera", "text": "Take Snapshot", "callback": self.take_snapshot},
                {"icon": "bell", "text": "Toggle Notifications", "callback": self.toggle_notifications},
                {"icon": "gear", "text": "Settings", "callback": self.show_settings}
            ]
            
            for action in actions:
                btn = QPushButton(action["text"])
                btn.clicked.connect(action["callback"])
                quick_actions_layout.addWidget(btn)
            
            right_panel.addWidget(quick_actions_group)
        
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
        
        # Create metrics tab (inspired by metrics.html)
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        
        # Add layout for metrics similar to metrics.html
        metrics_header = QLabel("System Metrics")
        metrics_header.setStyleSheet("font-size: 20px; font-weight: bold;")
        metrics_layout.addWidget(metrics_header)
        
        # Time range control
        time_range_layout = QHBoxLayout()
        time_range_label = QLabel("Time Range:")
        time_range_layout.addWidget(time_range_label)
        
        for range_option in ["1h", "3h", "24h"]:
            btn = QPushButton(range_option)
            btn.setCheckable(True)
            if range_option == "1h":
                btn.setChecked(True)
            time_range_layout.addWidget(btn)
        
        time_range_layout.addStretch()
        metrics_layout.addLayout(time_range_layout)
        
        # Stats grid similar to metrics.html
        stats_grid = QGridLayout()
        
        stat_items = [
            {"icon": "speedometer", "title": "FPS", "id": "fps_value", "initial": "0"},
            {"icon": "people", "title": "People Detected", "id": "detections_value", "initial": "0"},
            {"icon": "cpu", "title": "CPU Usage", "id": "cpu_usage_value", "initial": "0%"},
            {"icon": "memory", "title": "Memory Usage", "id": "memory_usage_value", "initial": "0%"}
        ]
        
        self.metric_values = {}
        
        for i, item in enumerate(stat_items):
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setStyleSheet("background-color: #252525; border-radius: 8px; padding: 10px;")
            
            layout = QVBoxLayout(frame)
            
            title = QLabel(item["title"])
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("color: #aaaaaa; font-size: 0.8rem;")
            
            value = QLabel(item["initial"])
            value.setAlignment(Qt.AlignCenter)
            value.setStyleSheet("font-size: 1.5rem; font-weight: bold; color: #ffffff;")
            self.metric_values[item["id"]] = value
            
            layout.addWidget(title)
            layout.addWidget(value)
            
            # Calculate grid position (2x2 grid)
            row, col = i // 2, i % 2
            stats_grid.addWidget(frame, row, col)
        
        metrics_layout.addLayout(stats_grid)
        
        # Add emotion distribution
        emotion_grid = QGridLayout()
        emotions = ['crying', 'laughing', 'babbling', 'silence']
        emotion_icons = {
            'crying': 'emoji-frown',
            'laughing': 'emoji-laughing',
            'babbling': 'chat-dots',
            'silence': 'volume-mute'
        }
        
        self.emotion_percentage = {}
        
        for i, emotion in enumerate(emotions):
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setStyleSheet(f"background-color: #252525; border-radius: 8px; padding: 10px;")
            
            layout = QVBoxLayout(frame)
            
            title = QLabel(emotion.capitalize())
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("color: #ffffff; font-size: 1rem;")
            
            value = QLabel("0%")
            value.setAlignment(Qt.AlignCenter)
            value.setStyleSheet("font-size: 1.2rem; font-weight: bold; color: #ffffff;")
            self.emotion_percentage[emotion] = value
            
            layout.addWidget(title)
            layout.addWidget(value)
            
            # Calculate grid position (2x2 grid)
            row, col = i // 2, i % 2
            emotion_grid.addWidget(frame, row, col)
        
        metrics_layout.addLayout(emotion_grid)
        
        # Add system health indicators
        health_group = QGroupBox("System Health")
        health_layout = QVBoxLayout(health_group)
        
        health_items = [
            {"label": "CPU Usage", "id": "cpu_health"},
            {"label": "Memory Usage", "id": "memory_health"},
            {"label": "Camera Status", "id": "camera_health"},
            {"label": "AI Processing", "id": "ai_health"}
        ]
        
        self.health_bars = {}
        
        for item in health_items:
            item_layout = QHBoxLayout()
            
            label = QLabel(item["label"])
            label.setStyleSheet("color: #aaaaaa;")
            item_layout.addWidget(label)
            
            progress = QProgressBar()
            progress.setRange(0, 100)
            if item["id"] == "camera_health" or item["id"] == "ai_health":
                progress.setValue(100)
            else:
                progress.setValue(0)
            progress.setTextVisible(False)
            
            color = "#198754"  # Green
            progress.setStyleSheet(f"""
                QProgressBar {{
                    border: none;
                    border-radius: 4px;
                    background-color: #333;
                    height: 8px;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 4px;
                }}
            """)
            
            self.health_bars[item["id"]] = progress
            item_layout.addWidget(progress)
            
            if item["id"] == "camera_health" or item["id"] == "ai_health":
                status_text = "Active"
            else:
                status_text = "0%"
            value = QLabel(status_text)
            value.setStyleSheet("font-weight: bold; color: #ffffff;")
            item_layout.addWidget(value)
            
            health_layout.addLayout(item_layout)
        
        metrics_layout.addWidget(health_group)
        
        self.tabs.addTab(metrics_tab, "Metrics")
        
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
        
        # Add developer tools tab if in developer mode
        if self.dev_mode:
            dev_tools_tab = DevToolsPanel(self)
            self.tabs.addTab(dev_tools_tab, "Developer Tools")
        
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
        
        # Set up timer for updating system metrics
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_metrics)
        self.metrics_timer.start(1000)  # Update every second
    
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
    
    def update_metrics(self):
        """Update system metrics display."""
        try:
            # Update FPS (simulated)
            fps = np.random.randint(25, 35)
            self.metric_values["fps_value"].setText(str(fps))
            
            # Update CPU usage (simulated)
            cpu_usage = np.random.randint(10, 30)
            self.metric_values["cpu_usage_value"].setText(f"{cpu_usage}%")
            self.status_items["cpuUsage"].setValue(cpu_usage)
            self.health_bars["cpu_health"].setValue(cpu_usage)
            
            # Update memory usage (simulated)
            memory_usage = np.random.randint(20, 50)
            self.metric_values["memory_usage_value"].setText(f"{memory_usage}%")
            self.status_items["memoryUsage"].setValue(memory_usage)
            self.health_bars["memory_health"].setValue(memory_usage)
            
            # Update person detections
            if hasattr(self, 'person_detector'):
                detections = 0  # Get actual values from person detector
                self.metric_values["detections_value"].setText(str(detections))
            
            # Update uptime
            uptime = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))
            self.status_items["uptime"].setText(uptime)
            
            # Update emotion distribution (simulated)
            emotions = ['crying', 'laughing', 'babbling', 'silence']
            probs = np.random.random(len(emotions))
            probs = probs / probs.sum()
            
            for emotion, prob in zip(emotions, probs):
                self.emotion_percentage[emotion].setText(f"{int(prob * 100)}%")
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def toggle_notifications(self):
        """Toggle notification settings."""
        QMessageBox.information(self, "Notifications", "Notification settings not implemented yet")
    
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
            # Store start time for uptime calculation
            self.start_time = time.time()
            
            # Start video thread if not running
            if hasattr(self, 'video_thread') and not self.video_thread.isRunning():
                self.video_thread.start()
            
            # Update UI
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.statusBar().showMessage("Monitoring started")
            
            # Update status items
            self.status_items["cameraStatus"].setText("Connected")
            self.status_items["personDetectorStatus"].setText("Running")
            self.status_items["emotionDetectorStatus"].setText("Running")
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
    parser = argparse.ArgumentParser(description="Baby Monitor GUI")
    parser.add_argument("--mode", choices=["normal", "dev"], default="normal",
                      help="Operation mode: normal or developer")
    
    args = parser.parse_args()
    dev_mode = args.mode == "dev"
    
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
    
    window = BabyMonitorGUI(dev_mode=dev_mode)
    window.show()
    return app.exec_() 