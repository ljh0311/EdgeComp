#!/usr/bin/env python3
"""
Baby Monitor Client Application

This script allows connecting to a Baby Monitor system running on a Raspberry Pi or other device.
It provides a GUI interface to view the camera feed, receive alerts, and monitor the system status.

Usage:
    python baby_client.py [options]

Options:
    --host HOST             Host IP address of the Baby Monitor server [default: 192.168.1.100]
    --port PORT             Port of the Baby Monitor server [default: 5000]
"""

import sys
import os
import time
import json
import argparse
import requests
import threading
from datetime import datetime
from functools import partial

# Import PyQt5 components
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, 
                                QHBoxLayout, QPushButton, QGridLayout, QComboBox, QProgressBar, 
                                QScrollArea, QFrame, QSplitter, QSizePolicy, QMessageBox, 
                                QInputDialog, QLineEdit, QStatusBar)
    from PyQt5.QtCore import Qt, QUrl, QTimer, pyqtSignal, QSize, QThread, QObject
    from PyQt5.QtGui import QIcon, QPixmap, QImage, QPalette, QColor
    from PyQt5.QtWebEngineWidgets import QWebEngineView
except ImportError:
    print("Error: PyQt5 is required. Please install it using: pip install PyQt5 PyQtWebEngine")
    sys.exit(1)

try:
    import socketio
except ImportError:
    print("Error: python-socketio is required. Please install it using: pip install python-socketio")
    sys.exit(1)

class VideoThread(QThread):
    """Thread for fetching video frames from the server"""
    update_frame = pyqtSignal(QPixmap)
    error_signal = pyqtSignal(str)
    
    def __init__(self, server_url):
        super().__init__()
        self.server_url = server_url
        self.running = False
    
    def run(self):
        self.running = True
        while self.running:
            try:
                # Get the latest frame from the server
                response = requests.get(f"{self.server_url}/video_feed", stream=True)
                if response.status_code == 200:
                    # Process the multipart/x-mixed-replace stream
                    bytes_data = bytes()
                    for chunk in response.iter_content(chunk_size=1024):
                        bytes_data += chunk
                        a = bytes_data.find(b'\xff\xd8')  # JPEG start
                        b = bytes_data.find(b'\xff\xd9')  # JPEG end
                        if a != -1 and b != -1:
                            jpg = bytes_data[a:b+2]
                            bytes_data = bytes_data[b+2:]
                            image = QImage()
                            image.loadFromData(jpg)
                            pixmap = QPixmap.fromImage(image)
                            self.update_frame.emit(pixmap)
                            time.sleep(0.01)  # Prevent CPU overuse
                else:
                    self.error_signal.emit(f"Failed to get video feed: {response.status_code}")
                    time.sleep(5)  # Wait before retrying
            except Exception as e:
                self.error_signal.emit(f"Video stream error: {str(e)}")
                time.sleep(5)  # Wait before retrying
            
            if not self.running:
                break
    
    def stop(self):
        self.running = False
        self.wait()

class SocketClient(QObject):
    """Socket.IO client for real-time updates"""
    emotion_update = pyqtSignal(dict)
    system_info_update = pyqtSignal(dict)
    alert_signal = pyqtSignal(str, str)  # level, message
    connection_status = pyqtSignal(bool)
    
    def __init__(self, server_url):
        super().__init__()
        self.server_url = server_url
        self.socket = socketio.Client()
        self.setup_socket_events()
    
    def setup_socket_events(self):
        @self.socket.event
        def connect():
            print("Connected to server")
            self.connection_status.emit(True)
        
        @self.socket.event
        def disconnect():
            print("Disconnected from server")
            self.connection_status.emit(False)
        
        @self.socket.on('emotion_update')
        def on_emotion_update(data):
            self.emotion_update.emit(data)
        
        @self.socket.on('system_info')
        def on_system_info(data):
            self.system_info_update.emit(data)
        
        @self.socket.on('alert')
        def on_alert(data):
            self.alert_signal.emit(data.get('level', 'info'), data.get('message', ''))
        
        @self.socket.on('crying_detected')
        def on_crying_detected(data):
            message = f"Crying detected (confidence: {(data.get('confidence', 0) * 100):.1f}%)"
            self.alert_signal.emit('warning', message)
    
    def connect_to_server(self):
        try:
            self.socket.connect(self.server_url)
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False
    
    def disconnect_from_server(self):
        try:
            if self.socket.connected:
                self.socket.disconnect()
        except Exception as e:
            print(f"Error disconnecting: {e}")

class AlertWidget(QFrame):
    """Widget for displaying alerts"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title_label = QLabel("<b>Alert History</b>")
        layout.addWidget(title_label)
        
        # Scroll area for alerts
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Container for alerts
        self.alert_container = QWidget()
        self.alert_layout = QVBoxLayout(self.alert_container)
        self.alert_layout.setAlignment(Qt.AlignTop)
        self.alert_layout.setSpacing(5)
        self.alert_layout.setContentsMargins(5, 5, 5, 5)
        
        self.scroll_area.setWidget(self.alert_container)
        layout.addWidget(self.scroll_area)
        
        # Clear button
        self.clear_button = QPushButton("Clear Alerts")
        self.clear_button.clicked.connect(self.clear_alerts)
        layout.addWidget(self.clear_button)
    
    def add_alert(self, level, message):
        # Create alert frame
        alert_frame = QFrame()
        alert_frame.setFrameShape(QFrame.StyledPanel)
        
        # Set color based on level
        if level == 'warning':
            alert_frame.setStyleSheet("background-color: rgba(255, 193, 7, 0.2); border: 1px solid #ffc107;")
        elif level == 'danger':
            alert_frame.setStyleSheet("background-color: rgba(220, 53, 69, 0.2); border: 1px solid #dc3545;")
        elif level == 'info':
            alert_frame.setStyleSheet("background-color: rgba(13, 202, 240, 0.2); border: 1px solid #0dcaf0;")
        else:
            alert_frame.setStyleSheet("background-color: rgba(108, 117, 125, 0.2); border: 1px solid #6c757d;")
        
        # Create layout
        alert_layout = QVBoxLayout(alert_frame)
        alert_layout.setContentsMargins(5, 5, 5, 5)
        
        # Time
        time_label = QLabel(datetime.now().strftime("%H:%M:%S"))
        time_label.setStyleSheet("color: #aaa; font-size: 10px;")
        alert_layout.addWidget(time_label)
        
        # Message
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        alert_layout.addWidget(message_label)
        
        # Add to layout
        self.alert_layout.insertWidget(0, alert_frame)
        
        # Limit to 15 alerts
        while self.alert_layout.count() > 15:
            item = self.alert_layout.itemAt(self.alert_layout.count() - 1)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()
    
    def clear_alerts(self):
        for i in reversed(range(self.alert_layout.count())):
            item = self.alert_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()

class SystemStatusWidget(QFrame):
    """Widget for displaying system status"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title_label = QLabel("<b>System Status</b>")
        layout.addWidget(title_label)
        
        # Status items
        status_layout = QGridLayout()
        status_layout.setColumnStretch(1, 1)
        layout.addLayout(status_layout)
        
        # Uptime
        status_layout.addWidget(QLabel("Uptime:"), 0, 0)
        self.uptime_label = QLabel("00:00:00")
        status_layout.addWidget(self.uptime_label, 0, 1)
        
        # Camera
        status_layout.addWidget(QLabel("Camera:"), 1, 0)
        self.camera_status = QLabel("Connecting...")
        status_layout.addWidget(self.camera_status, 1, 1)
        
        # Person Detection
        status_layout.addWidget(QLabel("Person Detection:"), 2, 0)
        self.person_detector_status = QLabel("Initializing...")
        status_layout.addWidget(self.person_detector_status, 2, 1)
        
        # Emotion Detection
        status_layout.addWidget(QLabel("Emotion Detection:"), 3, 0)
        self.emotion_detector_status = QLabel("Initializing...")
        status_layout.addWidget(self.emotion_detector_status, 3, 1)
        
        # CPU Usage
        status_layout.addWidget(QLabel("CPU Usage:"), 4, 0)
        self.cpu_usage = QProgressBar()
        self.cpu_usage.setRange(0, 100)
        self.cpu_usage.setValue(0)
        status_layout.addWidget(self.cpu_usage, 4, 1)
        
        # Memory Usage
        status_layout.addWidget(QLabel("Memory Usage:"), 5, 0)
        self.memory_usage = QProgressBar()
        self.memory_usage.setRange(0, 100)
        self.memory_usage.setValue(0)
        status_layout.addWidget(self.memory_usage, 5, 1)
        
        # Last Update
        self.last_update_label = QLabel("Last update: Never")
        self.last_update_label.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self.last_update_label, alignment=Qt.AlignRight)
    
    def update_status(self, data):
        """Update the status widgets with new data"""
        # Update uptime
        if 'uptime' in data:
            self.uptime_label.setText(data['uptime'])
        
        # Update camera status
        if 'camera_status' in data:
            status = data['camera_status']
            self.camera_status.setText(status.capitalize())
            self.set_status_style(self.camera_status, status)
        
        # Update person detector status
        if 'person_detector_status' in data:
            status = data['person_detector_status']
            self.person_detector_status.setText(status.capitalize())
            self.set_status_style(self.person_detector_status, status)
        
        # Update emotion detector status
        if 'emotion_detector_status' in data:
            status = data['emotion_detector_status']
            self.emotion_detector_status.setText(status.capitalize())
            self.set_status_style(self.emotion_detector_status, status)
        
        # Update CPU usage
        if 'cpu_usage' in data:
            cpu_usage = data['cpu_usage']
            self.cpu_usage.setValue(int(cpu_usage))
            self.set_progress_bar_color(self.cpu_usage, cpu_usage)
        
        # Update memory usage
        if 'memory_usage' in data:
            memory_usage = data['memory_usage']
            self.memory_usage.setValue(int(memory_usage))
            self.set_progress_bar_color(self.memory_usage, memory_usage)
        
        # Update last update time
        now = datetime.now().strftime("%H:%M:%S")
        self.last_update_label.setText(f"Last update: {now}")
    
    def set_status_style(self, label, status):
        """Set the style of a status label based on status value"""
        if status == 'connected' or status == 'running':
            label.setStyleSheet("color: #28a745;")  # Green
        elif status == 'initializing':
            label.setStyleSheet("color: #17a2b8;")  # Blue
        elif status == 'disconnected':
            label.setStyleSheet("color: #dc3545;")  # Red
        elif status == 'error':
            label.setStyleSheet("color: #ffc107;")  # Yellow
        else:
            label.setStyleSheet("color: #6c757d;")  # Gray
    
    def set_progress_bar_color(self, progress_bar, value):
        """Set the color of a progress bar based on the value"""
        if value < 50:
            progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #28a745; }")  # Green
        elif value < 80:
            progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #ffc107; }")  # Yellow
        else:
            progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #dc3545; }")  # Red

class EmotionWidget(QFrame):
    """Widget for displaying emotion data"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        
        # Set up layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        self.title_layout = QHBoxLayout()
        self.title_label = QLabel("<b>Current Emotion State</b>")
        self.title_layout.addWidget(self.title_label)
        
        self.model_name_label = QLabel("Model: Loading...")
        self.title_layout.addWidget(self.model_name_label, alignment=Qt.AlignRight)
        
        self.layout.addLayout(self.title_layout)
        
        # Container for emotion bars
        self.emotions_container = QWidget()
        self.emotions_layout = QVBoxLayout(self.emotions_container)
        self.emotions_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.emotions_container)
        
        # Default emotions
        self.emotions = ['crying', 'laughing', 'babbling', 'silence']
        self.emotion_bars = {}
        
        # Create emotion progress bars
        self.create_emotion_bars()
        
        # Last Update
        self.last_update_label = QLabel("Last update: Never")
        self.last_update_label.setStyleSheet("color: #aaa; font-size: 10px;")
        self.layout.addWidget(self.last_update_label, alignment=Qt.AlignRight)
    
    def create_emotion_bars(self):
        """Create progress bars for each emotion"""
        # Clear existing bars
        for i in reversed(range(self.emotions_layout.count())):
            item = self.emotions_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        
        # Reset emotion bars dict
        self.emotion_bars = {}
        
        # Create new bars for each emotion
        for emotion in self.emotions:
            # Frame for this emotion
            frame = QFrame()
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(0, 0, 0, 0)
            
            # Create label and progress bar
            label = QLabel(emotion.capitalize())
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            
            # Set color based on emotion
            if emotion == 'crying' or emotion == 'sad':
                progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #dc3545; }")  # Red
            elif emotion == 'laughing' or emotion == 'happy':
                progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #28a745; }")  # Green
            elif emotion == 'babbling':
                progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #17a2b8; }")  # Blue
            elif emotion == 'angry':
                progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #fd7e14; }")  # Orange
            else:
                progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #6c757d; }")  # Gray
            
            # Add to layout
            frame_layout.addWidget(label)
            frame_layout.addWidget(progress_bar)
            
            # Store reference to this bar
            self.emotion_bars[emotion] = progress_bar
            
            # Add to container
            self.emotions_layout.addWidget(frame)
    
    def update_emotions(self, data):
        """Update emotion bars with new data"""
        if 'confidences' in data:
            confidences = data['confidences']
            
            # Update each emotion bar
            for emotion, value in confidences.items():
                if emotion in self.emotion_bars:
                    self.emotion_bars[emotion].setValue(int(value * 100))
            
            # Update last update time
            now = datetime.now().strftime("%H:%M:%S")
            self.last_update_label.setText(f"Last update: {now}")
        
        # Check for emotion model info
        if 'model' in data:
            model_info = data['model']
            self.model_name_label.setText(f"Model: {model_info.get('name', 'Unknown')}")
            
            # If emotions list changed, recreate the bars
            model_emotions = model_info.get('emotions', [])
            if set(model_emotions) != set(self.emotions):
                self.emotions = model_emotions
                self.create_emotion_bars()

class BabyMonitorClient(QMainWindow):
    """Main window for the Baby Monitor Client"""
    
    def __init__(self, server_host, server_port):
        super().__init__()
        
        # Store connection info
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"http://{server_host}:{server_port}"
        
        # Set up UI
        self.setup_ui()
        
        # Initialize video thread
        self.video_thread = None
        
        # Initialize socket client
        self.socket_client = SocketClient(self.server_url)
        self.socket_client.emotion_update.connect(self.update_emotion)
        self.socket_client.system_info_update.connect(self.update_system_info)
        self.socket_client.alert_signal.connect(self.add_alert)
        self.socket_client.connection_status.connect(self.update_connection_status)
        
        # Connect to server
        self.connect_to_server()
        
        # Start status check timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.check_server_status)
        self.status_timer.start(10000)  # Check every 10 seconds
    
    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle("Baby Monitor Client")
        self.setMinimumSize(800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Top bar with connection status and settings
        top_bar = QHBoxLayout()
        
        self.connection_label = QLabel("Status: Disconnected")
        self.connection_label.setStyleSheet("color: #dc3545; font-weight: bold;")
        top_bar.addWidget(self.connection_label)
        
        top_bar.addStretch()
        
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_button_clicked)
        top_bar.addWidget(self.connect_button)
        
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.show_settings)
        top_bar.addWidget(settings_button)
        
        main_layout.addLayout(top_bar)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Video feed
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
        # Video label
        self.video_label = QLabel("Connecting to video feed...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(400, 300)
        video_layout.addWidget(self.video_label)
        
        splitter.addWidget(video_widget)
        
        # Right sidebar
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        
        # Add emotion widget
        self.emotion_widget = EmotionWidget()
        sidebar_layout.addWidget(self.emotion_widget)
        
        # Add system status widget
        self.system_status_widget = SystemStatusWidget()
        sidebar_layout.addWidget(self.system_status_widget)
        
        # Add alert widget
        self.alert_widget = AlertWidget()
        sidebar_layout.addWidget(self.alert_widget)
        
        splitter.addWidget(sidebar_widget)
        
        # Set initial sizes
        splitter.setSizes([600, 200])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
    
    def connect_to_server(self):
        """Connect to the baby monitor server"""
        self.statusBar.showMessage(f"Connecting to {self.server_url}...")
        
        # Connect socket
        threading.Thread(target=self._connect_socket, daemon=True).start()
        
        # Start video thread
        if self.video_thread is None or not self.video_thread.isRunning():
            self.video_thread = VideoThread(self.server_url)
            self.video_thread.update_frame.connect(self.update_video_frame)
            self.video_thread.error_signal.connect(self.show_video_error)
            self.video_thread.start()
    
    def _connect_socket(self):
        """Connect socket in a separate thread to avoid blocking UI"""
        success = self.socket_client.connect_to_server()
        if not success:
            # If failed, update UI from the main thread
            QApplication.instance().processEvents()
            self.update_connection_status(False)
    
    def disconnect_from_server(self):
        """Disconnect from the baby monitor server"""
        # Stop video thread
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
        
        # Disconnect socket
        self.socket_client.disconnect_from_server()
        
        # Update UI
        self.update_connection_status(False)
        self.video_label.setText("Disconnected from video feed")
        self.statusBar.showMessage("Disconnected from server")
    
    def connect_button_clicked(self):
        """Handle connect/disconnect button click"""
        if self.connect_button.text() == "Connect":
            # Show server input dialog
            host, ok1 = QInputDialog.getText(self, "Server Settings", "Server Host:",
                                           QLineEdit.Normal, self.server_host)
            
            if ok1:
                port, ok2 = QInputDialog.getInt(self, "Server Settings", "Server Port:",
                                             self.server_port, 1, 65535)
                
                if ok2:
                    # Update connection info
                    self.server_host = host
                    self.server_port = port
                    self.server_url = f"http://{host}:{port}"
                    
                    # Connect to server
                    self.connect_to_server()
        else:
            # Disconnect
            self.disconnect_from_server()
    
    def update_connection_status(self, connected):
        """Update the connection status in the UI"""
        if connected:
            self.connection_label.setText("Status: Connected")
            self.connection_label.setStyleSheet("color: #28a745; font-weight: bold;")
            self.connect_button.setText("Disconnect")
            self.statusBar.showMessage(f"Connected to {self.server_url}")
            
            # Add initial alert
            self.add_alert("info", f"Connected to baby monitor server at {self.server_host}:{self.server_port}")
        else:
            self.connection_label.setText("Status: Disconnected")
            self.connection_label.setStyleSheet("color: #dc3545; font-weight: bold;")
            self.connect_button.setText("Connect")
            self.statusBar.showMessage("Disconnected from server")
    
    def update_video_frame(self, pixmap):
        """Update the video frame with a new image"""
        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def show_video_error(self, error_message):
        """Show video error in the video label"""
        self.video_label.setText(error_message)
    
    def update_emotion(self, data):
        """Update emotion widget with new data"""
        self.emotion_widget.update_emotions(data)
    
    def update_system_info(self, data):
        """Update system status widget with new data"""
        self.system_status_widget.update_status(data)
    
    def add_alert(self, level, message):
        """Add an alert to the alert widget"""
        self.alert_widget.add_alert(level, message)
        
        # Also show in status bar for a short time
        self.statusBar.showMessage(message, 5000)
    
    def check_server_status(self):
        """Check if the server is still reachable"""
        if not hasattr(self.socket_client.socket, 'connected') or not self.socket_client.socket.connected:
            try:
                # Try a quick HTTP request to see if server is up
                response = requests.get(f"{self.server_url}/api/system_info", timeout=2)
                if response.status_code == 200:
                    # Server is up but socket is disconnected, try reconnecting
                    threading.Thread(target=self._connect_socket, daemon=True).start()
            except:
                # Server is down, update UI if needed
                if self.connect_button.text() == "Disconnect":
                    self.update_connection_status(False)
    
    def show_settings(self):
        """Show settings dialog"""
        host, ok1 = QInputDialog.getText(self, "Server Settings", "Server Host:",
                                       QLineEdit.Normal, self.server_host)
        
        if ok1:
            port, ok2 = QInputDialog.getInt(self, "Server Settings", "Server Port:",
                                         self.server_port, 1, 65535)
            
            if ok2:
                # Check if settings changed
                if host != self.server_host or port != self.server_port:
                    # Disconnect from current server
                    self.disconnect_from_server()
                    
                    # Update connection info
                    self.server_host = host
                    self.server_port = port
                    self.server_url = f"http://{host}:{port}"
                    
                    # Connect to new server
                    self.connect_to_server()
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Disconnect from server
        self.disconnect_from_server()
        
        # Accept the close event
        event.accept()

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Baby Monitor Client')
    parser.add_argument('--host', type=str, default='192.168.1.100',
                        help='Host IP address of the Baby Monitor server')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port of the Baby Monitor server')
    args = parser.parse_args()
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Baby Monitor Client")
    
    # Set dark theme
    app.setStyle("Fusion")
    
    # Dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    # Create and show the main window
    window = BabyMonitorClient(args.host, args.port)
    window.show()
    
    # Run the application
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main()) 