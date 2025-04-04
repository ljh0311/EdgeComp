"""
Baby Monitor System
=================
Main application module for the Baby Monitor System.
"""

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import logging
import time
import matplotlib
from queue import Empty
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque
import psutil

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
from PIL import Image, ImageTk
import argparse

# Add the src directory to Python path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
sys.path.insert(0, src_dir)

# Local imports
from babymonitor.camera.camera import Camera
from babymonitor.detectors.person_tracker import PersonDetector
from babymonitor.audio.audio_processor import AudioProcessor
from babymonitor.emotion.emotion import EmotionRecognizer
from babymonitor.core.web_app import BabyMonitorWeb
from babymonitor.core.config import config
from babymonitor.mqtt_server import MQTTServer

# Configure logging
logging.basicConfig(
    level=config.get('logging.level', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.get('logging.file')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmotionUI:
    def __init__(self, parent):
        """Initialize emotion recognition UI."""
        self.window = tk.Toplevel(parent)
        self.window.title("Emotion Recognition")
        self.window.geometry("800x600")

        # Configure dark theme colors
        bg_dark = "#2b2b2b"
        bg_darker = "#1e1e1e"
        text_color = "#ffffff"

        self.window.configure(bg=bg_dark)

        # Create main container
        main_container = ttk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(
            main_container,
            text="Real-time Emotion Detection",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))

        # Current emotion display
        self.emotion_label = ttk.Label(
            main_container,
            text="Current Emotion: None",
            font=("Arial", 14)
        )
        self.emotion_label.pack(pady=(0, 20))

        # Create progress bars for each emotion
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
        
        self.progress_bars = {}
        for emotion in self.emotions:
            emotion_frame = ttk.Frame(main_container)
            emotion_frame.pack(fill=tk.X, pady=5)
            
            # Emotion label
            label = ttk.Label(
                emotion_frame,
                text=emotion.capitalize(),
                width=10,
                foreground=self.emotion_colors[emotion]
            )
            label.pack(side=tk.LEFT, padx=5)
            
            # Progress bar
            progress = ttk.Progressbar(
                emotion_frame,
                length=400,
                mode='determinate'
            )
            progress.pack(side=tk.LEFT, padx=5)
            
            # Percentage label
            value_label = ttk.Label(
                emotion_frame,
                text="0%",
                width=5
            )
            value_label.pack(side=tk.LEFT, padx=5)
            
            self.progress_bars[emotion] = (progress, value_label)

        # Status section
        status_frame = ttk.LabelFrame(main_container, text="Status")
        status_frame.pack(fill=tk.X, pady=20)
        
        self.status_label = ttk.Label(
            status_frame,
            text="Status: Initializing...",
            foreground="white"
        )
        self.status_label.pack(pady=5)

    def update_emotion(self, emotions):
        """Update emotion visualization."""
        try:
            # Update progress bars
            for emotion, (progress, label) in self.progress_bars.items():
                confidence = emotions.get(emotion, 0.0)
                percentage = int(confidence * 100)
                progress['value'] = percentage
                label['text'] = f"{percentage}%"
                
                # Update progress bar color
                progress['style'] = f"{emotion}.Horizontal.TProgressbar"
                style = ttk.Style()
                style.configure(
                    f"{emotion}.Horizontal.TProgressbar",
                    troughcolor=self.emotion_colors[emotion],
                    background=self.emotion_colors[emotion]
                )

            # Update current emotion (highest confidence)
            max_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            self.emotion_label['text'] = f"Current Emotion: {max_emotion.capitalize()}"
            self.emotion_label['foreground'] = self.emotion_colors[max_emotion]
            
            # Update status
            self.status_label['text'] = "Status: Active"
            self.status_label['foreground'] = "green"

        except Exception as e:
            logger.error(f"Error updating emotion UI: {str(e)}")
            self.status_label['text'] = "Status: Error"
            self.status_label['foreground'] = "red"

    def close(self):
        """Close the emotion UI window."""
        self.window.destroy()


class BabyMonitorSystem:
    def __init__(self, dev_mode=False, only_local=False, only_web=False, mqtt_host=None, mqtt_port=1883):
        """Initialize the Baby Monitor System."""
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.camera_error = False
        self.audio_enabled = True
        self.camera_enabled = True  # Start with camera enabled
        self.frame_lock = threading.Lock()
        self.dev_mode = dev_mode
        self.only_local = only_local
        self.only_web = only_web
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.initialized_components = []  # Track initialized components for cleanup
        self.web_app = None  # Initialize web_app reference as None
        self.dev_window = None  # Initialize dev window reference as None
        self.emotion_ui = None  # Initialize emotion UI reference as None
        self.mqtt_server = None  # Initialize MQTT server reference as None

        # Performance metrics
        self.metrics_lock = threading.Lock()
        self.fps_history = deque(maxlen=100)  # Store last 100 FPS values
        self.frame_times = deque(maxlen=100)  # Store last 100 frame processing times
        self.cpu_history = deque(maxlen=100)  # Store last 100 CPU usage values
        self.memory_history = deque(maxlen=100)  # Store last 100 memory usage values
        self.last_metrics_update = time.time()
        self.metrics_update_interval = 1.0  # Update metrics every second

        try:
            # Initialize camera first
            camera_config = config.get('camera', {})
            self.camera = Camera(
                width=camera_config.get('width', 640),
                height=camera_config.get('height', 480)
            )
            if not self.camera.initialize():
                self.logger.error("Failed to initialize camera")
                self.camera_error = True
                self.camera_enabled = False
            else:
                self.initialized_components.append("camera")
                self.logger.info("Camera initialized successfully")

            # Initialize MQTT server if host is provided
            if mqtt_host:
                self.mqtt_server = MQTTServer(host=mqtt_host, port=mqtt_port)
                self.logger.info(f"Initializing MQTT server at {mqtt_host}:{mqtt_port}")
                self.initialized_components.append("mqtt_server")

            # Initialize UI if not web-only
            if not only_web:
                try:
                    self.root = tk.Tk()
                    self.setup_ui()
                    self.initialized_components.append("ui")

                    # Create emotion UI
                    self.emotion_ui = EmotionUI(self.root)
                    self.initialized_components.append("emotion_ui")

                    # Update camera UI status and selections
                    if hasattr(self, "camera_status"):
                        if self.camera_error:
                            self.camera_status.configure(text="📷  Camera: Error")
                        else:
                            self.camera_status.configure(text="📷  Camera: Ready")
                            # Update camera list and select current camera
                            self.update_camera_list()
                            camera_list = self.camera.get_camera_list()
                            if camera_list:
                                current_camera = camera_list[
                                    self.camera.selected_camera_index
                                ]
                                self.camera_select.set(current_camera)
                                # Update resolution list and select current resolution
                                self.update_resolution_list(current_camera)
                                current_resolution = (
                                    self.camera.get_current_resolution()
                                )
                                if current_resolution:
                                    res_str = f"{current_resolution[0]}x{current_resolution[1]}"
                                    self.resolution_select.set(res_str)
                except Exception as e:
                    self.logger.error(f"Error initializing UI: {str(e)}")
                    if not only_local:
                        # Fall back to web-only mode
                        self.only_web = True
                        self.logger.info("Falling back to web-only mode")

            # Initialize person detector
            try:
                model_path = os.path.join(src_dir, "models", "yolov8n.pt")
                self.person_detector = PersonDetector(model_path=model_path)
                self.initialized_components.append("person_detector")
                self.logger.info("Person detector initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize person detector: {str(e)}")
                if not self.only_web and hasattr(self, "detection_status"):
                    self.detection_status.configure(text="👀  Detection: Error")
                self.person_detector = None

            # Initialize audio processor if enabled
            self.audio_processor = None
            if self.audio_enabled:
                try:
                    self.audio_processor = AudioProcessor()
                    self.initialized_components.append("audio_processor")
                except Exception as e:
                    self.logger.error(f"Error initializing audio processor: {str(e)}")
                    self.audio_enabled = False
                    self.logger.info("Audio processing disabled")

            # Initialize emotion recognizer
            self.emotion_recognizer = EmotionRecognizer()
            self.initialized_components.append("emotion_recognizer")

            # Initialize web interface if not local-only
            if not only_local:
                try:
                    self.web_app = BabyMonitorWeb(self)
                    self.web_app.set_monitor_system(self)
                    self.web_app.start()
                    self.initialized_components.append("web_app")
                except Exception as e:
                    self.logger.error(f"Failed to initialize web interface: {str(e)}")
                    if only_web:  # Web interface is required in web-only mode
                        self.cleanup()
                        raise

            self.logger.info("Baby Monitor System initialized successfully")

        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            self.cleanup()

    def setup_ui(self):
        """Setup the main UI window."""
        self.root.title("Baby Monitor")
        self.root.geometry("1200x800")

        # Configure dark theme colors
        bg_dark = "#2b2b2b"
        bg_darker = "#1e1e1e"
        text_color = "#ffffff"
        status_bg = "#333333"

        # Configure style
        style = ttk.Style()
        style.configure("Dark.TFrame", background=bg_dark)
        style.configure("Darker.TFrame", background=bg_darker)
        style.configure("Dark.TLabelframe", background=bg_dark)
        style.configure(
            "Dark.TLabelframe.Label", background=bg_dark, foreground=text_color
        )
        style.configure(
            "Status.TLabel", background=status_bg, foreground=text_color, padding=5
        )
        style.configure(
            "Dark.TCombobox",
            background=bg_darker,
            foreground=text_color,
            fieldbackground=bg_darker,
            selectbackground=bg_dark,
        )

        # Set root window background
        self.root.configure(bg=bg_dark)

        # Create main container
        main_container = ttk.Frame(self.root, style="Dark.TFrame")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left panel (video feed and waveform)
        left_panel = ttk.Frame(main_container, style="Dark.TFrame")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Video feed frame
        video_frame = ttk.Frame(left_panel, style="Darker.TFrame")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.video_canvas = tk.Canvas(video_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Camera controls frame
        camera_controls = ttk.Frame(video_frame, style="Dark.TFrame")
        camera_controls.pack(fill=tk.X, pady=5)

        # Camera selection
        self.camera_select = ttk.Combobox(
            camera_controls, state="readonly", style="Dark.TCombobox", width=30
        )
        self.camera_select.pack(side=tk.LEFT, padx=5)
        self.camera_select.bind("<<ComboboxSelected>>", self.on_camera_selected)

        # Resolution selection
        self.resolution_select = ttk.Combobox(
            camera_controls, state="readonly", style="Dark.TCombobox", width=15
        )
        self.resolution_select.pack(side=tk.LEFT, padx=5)
        self.resolution_select.bind("<<ComboboxSelected>>", self.on_resolution_selected)

        # Camera toggle button
        self.camera_btn = ttk.Button(
            camera_controls, text="Toggle Camera", command=self.toggle_camera
        )
        self.camera_btn.pack(side=tk.RIGHT, padx=5)

        # Right panel (controls and status)
        right_panel = ttk.Frame(main_container, style="Dark.TFrame", width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        right_panel.pack_propagate(False)

        # Controls section
        controls_frame = ttk.LabelFrame(
            right_panel, text="Controls", style="Dark.TLabelframe"
        )
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        # Audio toggle button
        self.audio_btn = ttk.Button(
            controls_frame, text="Audio Monitor", command=self.toggle_audio
        )
        self.audio_btn.pack(fill=tk.X, padx=5, pady=5)

        # Emotion UI toggle button
        self.emotion_btn = ttk.Button(
            controls_frame, text="Emotion Recognition", command=self.toggle_emotion_ui
        )
        self.emotion_btn.pack(fill=tk.X, padx=5, pady=5)

        # Status section
        status_frame = ttk.LabelFrame(
            right_panel, text="Status", style="Dark.TLabelframe"
        )
        status_frame.pack(fill=tk.X, pady=(0, 10))

        # Status labels with dark background
        self.camera_status = ttk.Label(
            status_frame, text="📷  Camera: Initializing...", style="Status.TLabel"
        )
        self.camera_status.pack(fill=tk.X, pady=1)

        self.audio_status = ttk.Label(
            status_frame, text="🎤  Audio: Initializing...", style="Status.TLabel"
        )
        self.audio_status.pack(fill=tk.X, pady=1)

        self.emotion_status = ttk.Label(
            status_frame, text="😊  Emotion: Initializing...", style="Status.TLabel"
        )
        self.emotion_status.pack(fill=tk.X, pady=1)

        self.detection_status = ttk.Label(
            status_frame, text="👀  Detection: Initializing...", style="Status.TLabel"
        )
        self.detection_status.pack(fill=tk.X, pady=1)

        # Add decibel level display
        self.decibel_label = ttk.Label(
            status_frame, text="Sound Level: -- dB", style="Status.TLabel"
        )
        self.decibel_label.pack(fill=tk.X, pady=1)

        # Initialize waveform
        self.setup_waveform()

        # Initialize camera selection
        self.update_camera_list()

    def setup_waveform(self):
        """Setup waveform visualization."""
        # Create waveform frame
        self.waveform_frame = ttk.Frame(self.root, style="Darker.TFrame", height=150)
        self.waveform_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        self.waveform_frame.pack_propagate(False)

        # Setup matplotlib figure with dark theme
        plt.style.use("dark_background")
        self.waveform_figure = Figure(figsize=(12, 2), dpi=100, facecolor="#1e1e1e")
        self.waveform_canvas = FigureCanvasTkAgg(
            self.waveform_figure, master=self.waveform_frame
        )
        self.waveform_canvas.draw()
        self.waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Setup the plot
        self.waveform_ax = self.waveform_figure.add_subplot(111)
        self.waveform_ax.set_facecolor("#1e1e1e")
        self.waveform_ax.set_ylim([-1, 1])
        self.waveform_ax.set_xlim([0, 1000])
        self.waveform_ax.grid(True, color="#333333", linestyle="-", alpha=0.3)

        # Remove axis labels and ticks
        self.waveform_ax.set_xticks([])
        self.waveform_ax.set_yticks([])

        # Create the line with a nice color
        (self.waveform_line,) = self.waveform_ax.plot([], [], lw=2, color="#4CAF50")
        self.waveform_data = np.zeros(1000)

        # Remove margins
        self.waveform_figure.tight_layout(pad=0.1)

        # Bind resize event
        self.waveform_frame.bind("<Configure>", self._set_waveform_height)

    def toggle_camera(self):
        """Toggle camera feed on/off."""
        self.camera_enabled = not self.camera_enabled

        try:
            if self.camera_enabled:
                if not self.only_web and hasattr(self, "camera_btn"):
                    self.camera_btn.configure(text="Camera Feed")
                    self.camera_status.configure(text="📷  Camera: Active")
                if self.web_app is not None:
                    self.web_app.emit_status({"camera_enabled": True})
            else:
                if not self.only_web and hasattr(self, "camera_btn"):
                    self.camera_btn.configure(text="Camera Feed")
                    self.camera_status.configure(text="📷  Camera: Disabled")
                    # Clear the video canvas
                    self.video_canvas.delete("all")
                if self.web_app is not None:
                    self.web_app.emit_status({"camera_enabled": False})
                    # Clear any remaining frames from the queue
                    with self.web_app.frame_lock:
                        while not self.web_app.frame_queue.empty():
                            try:
                                self.web_app.frame_queue.get_nowait()
                            except Empty:
                                break
        except Exception as e:
            self.logger.error(f"Error toggling camera: {str(e)}")
            if not self.only_web and hasattr(self, "camera_status"):
                self.camera_status.configure(text="📷  Camera: Error")

    def toggle_audio(self):
        """Toggle audio monitoring on/off."""
        self.audio_enabled = not self.audio_enabled

        try:
            if self.audio_enabled:
                if hasattr(self, "audio_processor"):
                    self.audio_processor.start()
                if hasattr(self, "emotion_recognizer"):
                    self.emotion_recognizer.start()
                if not self.only_web and hasattr(self, "audio_btn"):
                    self.audio_btn.configure(text="Audio Monitor")
                    self.audio_status.configure(text="🎤  Audio: Active")
                if self.web_app:
                    self.web_app.emit_status({"audio_enabled": True})
            else:
                if hasattr(self, "audio_processor"):
                    self.audio_processor.stop()
                if hasattr(self, "emotion_recognizer"):
                    self.emotion_recognizer.stop()
                if not self.only_web and hasattr(self, "audio_btn"):
                    self.audio_btn.configure(text="Audio Monitor")
                    self.audio_status.configure(text="🎤  Audio: Disabled")
                if self.web_app:
                    self.web_app.emit_status({"audio_enabled": False})
        except Exception as e:
            self.logger.error(f"Error toggling audio: {str(e)}")
            if not self.only_web and hasattr(self, "audio_status"):
                self.audio_status.configure(text="🎤  Audio: Error")

    def handle_alert(self, message, level="info"):
        """Handle alert and send to all connected clients."""
        if self.web_app:
            self.web_app.send_alert(message, level)
            
        # Send alert via MQTT if available
        if self.mqtt_server and self.mqtt_server.is_connected():
            self.mqtt_server.publish_alert(message, level)

    def send_status_update(self, status_data):
        """Send status update to connected clients."""
        if self.web_app:
            self.web_app.send_status_update(status_data)
        
        # Send status update via MQTT if available
        if self.mqtt_server and self.mqtt_server.is_connected():
            self.mqtt_server.publish_system_status(status_data)

    def start(self):
        """Start the Baby Monitor System."""
        self.logger.info("Starting Baby Monitor System")
        self.is_running = True

        # Start MQTT server if available
        if self.mqtt_server:
            self.mqtt_server.start()
            self.logger.info("MQTT server started")

        # Start web interface if available
        if self.web_app:
            try:
                self.web_app.start()
                self.logger.info(f"Web interface started at http://localhost:{self.web_app.port}")
            except Exception as e:
                self.logger.error(f"Error starting web interface: {str(e)}")

        # Start camera processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Start audio processing thread if enabled
        if self.audio_enabled and self.audio_processor:
            self.audio_processor.start()

        # Start UI main loop if using the local UI
        if not self.only_web and hasattr(self, "root"):
            self.root.mainloop()

    def stop(self):
        """Stop the monitoring system."""
        self.is_running = False

        try:
            self.cleanup()
            self.logger.info("Baby monitor system stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping system: {str(e)}")
            raise

    def process_frames(self):
        """Process video frames from the camera."""
        last_frame_time = time.time()
        frame_count = 0
        crying_detected_time = None
        crying_cooldown = 10  # Seconds

        try:
            while self.is_running:
                if not self.camera_enabled:
                    time.sleep(0.1)
                    continue

                # Get frame from camera
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_start_time = time.time()
                frame_h, frame_w = frame.shape[:2]

                # Create a copy for processing
                processed_frame = frame.copy()

                # Process frame with person detection
                detections = []
                if self.person_detector:
                    try:
                        detections = self.person_detector.detect(processed_frame)
                    except Exception as e:
                        self.logger.error(f"Error in person detection: {str(e)}")

                # Process emotions from audio if available
                emotions = {}
                if self.audio_enabled and self.audio_processor:
                    try:
                        # Get audio data and check for crying
                        audio_data = self.audio_processor.get_latest_audio_block()
                        if audio_data is not None:
                            # Send to emotion recognizer
                            recognition_result = self.emotion_recognizer.process_audio(audio_data)
                            
                            # Update emotions with audio result
                            if recognition_result:
                                emotions = recognition_result
                                
                                # Check if crying detected
                                if 'crying' in emotions and emotions['crying'] > 0.7:
                                    # Only trigger alert if cooldown has passed
                                    current_time = time.time()
                                    if crying_detected_time is None or (current_time - crying_detected_time) > crying_cooldown:
                                        crying_detected_time = current_time
                                        self.handle_alert(f"Crying detected (confidence: {emotions['crying']:.2f})", "warning")
                                        # Send crying alert via MQTT if available
                                        if self.mqtt_server and self.mqtt_server.is_connected():
                                            self.mqtt_server.publish_crying_detection(emotions['crying'])
                                
                                # Update emotion UI
                                self.update_emotion_ui(emotions)
                    except Exception as e:
                        self.logger.error(f"Error processing audio: {str(e)}")

                # Update dev window if available
                if self.dev_window:
                    try:
                        self.dev_window.update_detection_visualization(
                            processed_frame, detections
                        )
                        if emotions:
                            self.dev_window.update_emotion_visualization(emotions)
                    except Exception as e:
                        self.logger.error(f"Error updating dev window: {str(e)}")

                # Calculate and store FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                last_frame_time = current_time

                with self.metrics_lock:
                    self.fps_history.append(fps)
                    self.frame_times.append(1000 * (current_time - frame_start_time))  # Convert to ms

                    # Update system metrics less frequently
                    if current_time - self.last_metrics_update > self.metrics_update_interval:
                        self.cpu_history.append(psutil.cpu_percent())
                        self.memory_history.append(psutil.virtual_memory().percent)
                        self.last_metrics_update = current_time

                # Compute average FPS and other metrics
                avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
                avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
                avg_cpu = sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0
                avg_memory = sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0

                # Send status update to clients
                status_data = {
                    "fps": f"{avg_fps:.1f}",
                    "frame_time": f"{avg_frame_time:.1f} ms",
                    "cpu_usage": avg_cpu,
                    "memory_usage": avg_memory,
                    "camera_status": "connected" if self.camera_enabled else "disconnected",
                    "person_detector_status": "running" if self.person_detector else "disabled",
                    "emotion_detector_status": "running" if self.emotion_recognizer else "disabled",
                    "uptime": self.get_uptime_string(),
                }
                self.send_status_update(status_data)

                # Update UI if using local interface
                if not self.only_web and hasattr(self, "root") and hasattr(self, "camera_canvas"):
                    try:
                        # Add detection boxes
                        ui_frame = processed_frame.copy()
                        for det in detections:
                            bbox = det["bbox"]
                            x1, y1, x2, y2 = map(int, bbox)
                            confidence = det["confidence"]
                            label = det["label"]
                            cv2.rectangle(ui_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                ui_frame,
                                f"{label} {confidence:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                        # Convert to format for tkinter
                        img_rgb = cv2.cvtColor(ui_frame, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_rgb)
                        imgtk = ImageTk.PhotoImage(image=img_pil)

                        # Update canvas with image
                        with self.frame_lock:
                            if hasattr(self, "camera_canvas"):
                                self.camera_canvas.imgtk = imgtk
                                self.camera_canvas.configure(image=imgtk)
                    except Exception as e:
                        self.logger.error(f"Error updating UI: {str(e)}")

                # Convert frame to JPEG for web clients
                if self.web_app:
                    try:
                        # Convert the frame to JPEG format with quality 70 (good balance between size and quality)
                        _, jpeg_frame = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        jpeg_bytes = jpeg_frame.tobytes()
                        
                        # Send frame via web app
                        self.web_app.send_frame(jpeg_bytes)
                        
                        # Send frame via MQTT if available
                        if self.mqtt_server and self.mqtt_server.is_connected():
                            self.mqtt_server.publish_video_frame(jpeg_bytes)
                    except Exception as e:
                        self.logger.error(f"Error sending frame: {str(e)}")

                # Increment frame counter
                frame_count += 1

        except Exception as e:
            self.logger.error(f"Error in frame processing loop: {str(e)}")
        finally:
            self.logger.info("Frame processing stopped")

    def _set_waveform_height(self, event):
        """Adjust waveform figure size on window resize."""
        if hasattr(self, "waveform_figure"):
            try:
                # Get the new width and height
                width = event.width / self.waveform_figure.dpi
                height = event.height / self.waveform_figure.dpi

                # Update figure size
                self.waveform_figure.set_size_inches(width, height, forward=True)
                self.waveform_figure.tight_layout(pad=0.1)
                self.waveform_canvas.draw_idle()
            except Exception as e:
                self.logger.error(f"Error resizing waveform: {str(e)}")

    def update_audio_visualization(self, audio_data):
        """Update audio visualizations including waveform and decibel meter."""
        if self.only_web:
            return

        try:
            # Calculate RMS value
            rms = np.sqrt(np.mean(np.square(audio_data)))

            # Convert to decibels
            if rms > 0:
                db = 20 * np.log10(rms)
            else:
                db = -60  # Minimum dB level

            # Update waveform
            if hasattr(self, "waveform_data"):
                # Resize audio data to match waveform buffer size
                if len(audio_data) != len(self.waveform_data):
                    audio_data = np.interp(
                        np.linspace(0, len(audio_data), len(self.waveform_data)),
                        np.arange(len(audio_data)),
                        audio_data,
                    )

                # Update waveform data with smooth transition
                alpha = 0.5  # Smoothing factor
                self.waveform_data = (
                    alpha * audio_data + (1 - alpha) * self.waveform_data
                )

                # Update the plot
                if hasattr(self, "waveform_line"):
                    self.waveform_line.set_data(
                        range(len(self.waveform_data)), self.waveform_data
                    )
                    self.waveform_canvas.draw_idle()

            # Update decibel display
            if hasattr(self, "decibel_label"):
                self.decibel_label.configure(
                    text=f"Sound Level: {db:.1f} dB", foreground=self.get_db_color(db)
                )

            # Update web interface if available
            if self.web_app is not None:
                self.web_app.emit_audio_data(
                    {"waveform": audio_data.tolist(), "decibel": db}
                )

        except Exception as e:
            self.logger.error(f"Error updating audio visualization: {str(e)}")

    def get_db_color(self, db_level):
        """Get color for decibel level display."""
        if db_level > -20:
            return "#ff4444"  # Red for loud
        elif db_level > -40:
            return "#ffeb3b"  # Yellow for medium
        return "#4CAF50"  # Green for quiet

    def update_waveform(self):
        """Update the waveform visualization."""
        if hasattr(self, "waveform_line"):
            try:
                self.waveform_line.set_data(
                    range(len(self.waveform_data)), self.waveform_data
                )
                self.waveform_canvas.draw_idle()
            except Exception as e:
                self.logger.error(f"Error updating waveform: {str(e)}")

    def update_camera_list(self):
        """Update the camera list in the combobox."""
        try:
            if not hasattr(self, "camera") or not hasattr(self, "camera_select"):
                self.logger.warning("Camera or UI components not initialized yet")
                return

            camera_list = self.camera.get_camera_list()
            if camera_list:
                self.camera_select["values"] = camera_list
                # Don't set the current camera here as it's done in init
        except Exception as e:
            self.logger.error(f"Error updating camera list: {str(e)}")

    def update_resolution_list(self, camera_name=None):
        """Update the resolution list in the combobox."""
        try:
            if not hasattr(self, "camera") or not hasattr(self, "resolution_select"):
                self.logger.warning("Camera or UI components not initialized yet")
                return

            if camera_name is None:
                camera_name = self.camera_select.get()

            resolutions = self.camera.get_camera_resolutions(camera_name)
            if resolutions:
                # Clear current values
                self.resolution_select["values"] = []
                # Set new values
                self.resolution_select["values"] = tuple(resolutions)
                # Get current resolution
                current_resolution = self.camera.get_current_resolution()
                if current_resolution in resolutions:
                    self.resolution_select.set(current_resolution)
                elif resolutions:
                    # Set first available resolution if current not in list
                    self.resolution_select.set(resolutions[0])
        except Exception as e:
            self.logger.error(f"Error updating resolution list: {str(e)}")

    def on_camera_selected(self, event):
        """Handle camera selection."""
        try:
            selected_camera = self.camera_select.get()
            if selected_camera:
                if self.camera.select_camera(selected_camera):
                    self.camera_status.configure(text="📷  Camera: Ready")
                    # Update resolution list for the new camera
                    self.update_resolution_list(selected_camera)
                else:
                    self.camera_status.configure(text="📷  Camera: Error")
        except Exception as e:
            self.logger.error(f"Error selecting camera: {str(e)}")
            self.camera_status.configure(text="📷  Camera: Error")

    def on_resolution_selected(self, event):
        """Handle resolution selection."""
        try:
            selected_resolution = self.resolution_select.get()
            if selected_resolution and isinstance(selected_resolution, str):
                if self.camera.set_resolution(selected_resolution):
                    self.camera_status.configure(text="📷  Camera: Ready")
                    # Verify the actual resolution set
                    current_resolution = self.camera.get_current_resolution()
                    if current_resolution != selected_resolution:
                        self.resolution_select.set(current_resolution)
                        self.logger.warning(
                            f"Camera set to {current_resolution} instead of requested {selected_resolution}"
                        )
                else:
                    self.camera_status.configure(text="📷  Camera: Resolution Error")
        except Exception as e:
            self.logger.error(f"Error setting resolution: {str(e)}")
            self.camera_status.configure(text="📷  Camera: Resolution Error")

    def toggle_emotion_ui(self):
        """Toggle emotion recognition UI window."""
        if self.emotion_ui is None or not self.emotion_ui.window.winfo_exists():
            self.emotion_ui = EmotionUI(self.root)
            self.emotion_btn.configure(text="Close Emotion UI")
        else:
            self.emotion_ui.close()
            self.emotion_ui = None
            self.emotion_btn.configure(text="Emotion Recognition")

    def update_emotion_ui(self, emotions):
        """Update emotion UI with detection results."""
        if self.emotion_ui:
            self.emotion_ui.update_emotion(emotions)
            
        # Send emotion data via MQTT if available
        if self.mqtt_server and self.mqtt_server.is_connected():
            emotion_data = {"confidences": emotions, "model": {"name": "Baby Monitor Emotion", "emotions": list(emotions.keys())}}
            self.mqtt_server.publish_emotion_state(emotion_data)

    def cleanup(self):
        """Clean up resources before exiting."""
        self.logger.info("Cleaning up resources")
        self.is_running = False

        if "camera" in self.initialized_components:
            try:
                self.camera.release()
                self.logger.info("Camera released")
            except Exception as e:
                self.logger.error(f"Error releasing camera: {str(e)}")

        if "audio_processor" in self.initialized_components and self.audio_processor:
            try:
                self.audio_processor.stop()
                self.logger.info("Audio processor stopped")
            except Exception as e:
                self.logger.error(f"Error stopping audio processor: {str(e)}")

        if "web_app" in self.initialized_components and self.web_app:
            try:
                self.web_app.stop()
                self.logger.info("Web app stopped")
            except Exception as e:
                self.logger.error(f"Error stopping web app: {str(e)}")
                
        if "mqtt_server" in self.initialized_components and self.mqtt_server:
            try:
                self.mqtt_server.stop()
                self.logger.info("MQTT server stopped")
            except Exception as e:
                self.logger.error(f"Error stopping MQTT server: {str(e)}")

        # Wait for processing thread to stop
        if hasattr(self, "processing_thread") and self.processing_thread:
            try:
                if self.processing_thread.is_alive():
                    self.processing_thread.join(timeout=1.0)
                    self.logger.info("Processing thread stopped")
            except Exception as e:
                self.logger.error(f"Error stopping processing thread: {str(e)}")

        self.logger.info("Cleanup complete")


class DevWindow:
    def __init__(self, master):
        """Initialize developer debug window."""
        self.window = tk.Toplevel(master)
        self.window.title("Developer Debug Window")
        self.window.geometry("800x600")

        # Configure dark theme colors
        bg_dark = "#2b2b2b"
        bg_darker = "#1e1e1e"
        text_color = "#ffffff"

        self.window.configure(bg=bg_dark)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Audio Debug Tab
        self.audio_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.audio_frame, text="Audio Debug")

        # Audio metrics
        self.audio_metrics = tk.Text(
            self.audio_frame, height=10, bg=bg_darker, fg=text_color
        )
        self.audio_metrics.pack(fill=tk.X, padx=5, pady=5)

        # Raw waveform
        self.raw_waveform_canvas = tk.Canvas(self.audio_frame, bg=bg_darker, height=150)
        self.raw_waveform_canvas.pack(fill=tk.X, padx=5, pady=5)

        # Decibel meter
        self.decibel_canvas = tk.Canvas(self.audio_frame, bg=bg_darker, height=100)
        self.decibel_canvas.pack(fill=tk.X, padx=5, pady=5)

        # Detection Debug Tab
        self.detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.detection_frame, text="Detection Debug")

        # Detection metrics
        self.detection_metrics = tk.Text(
            self.detection_frame, height=10, bg=bg_darker, fg=text_color
        )
        self.detection_metrics.pack(fill=tk.X, padx=5, pady=5)

        # Detection visualization
        self.detection_canvas = tk.Canvas(self.detection_frame, bg=bg_darker)
        self.detection_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Emotion Debug Tab
        self.emotion_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.emotion_frame, text="Emotion Debug")

        # Emotion metrics
        self.emotion_metrics = tk.Text(
            self.emotion_frame, height=10, bg=bg_darker, fg=text_color
        )
        self.emotion_metrics.pack(fill=tk.X, padx=5, pady=5)

        # Emotion confidence bars
        self.emotion_canvas = tk.Canvas(self.emotion_frame, bg=bg_darker)
        self.emotion_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def update_audio_metrics(self, metrics):
        """Update audio debug information."""
        self.audio_metrics.delete(1.0, tk.END)
        self.audio_metrics.insert(tk.END, metrics)

    def update_detection_metrics(self, metrics):
        """Update detection debug information."""
        self.detection_metrics.delete(1.0, tk.END)
        self.detection_metrics.insert(tk.END, metrics)

    def update_emotion_metrics(self, metrics):
        """Update emotion debug information."""
        self.emotion_metrics.delete(1.0, tk.END)
        self.emotion_metrics.insert(tk.END, metrics)

    def update_raw_waveform(self, data):
        """Update raw waveform visualization."""
        canvas = self.raw_waveform_canvas
        canvas.delete("all")

        width = canvas.winfo_width()
        height = canvas.winfo_height()
        mid = height // 2

        if len(data) < 2:
            return

        points = []
        for i, val in enumerate(data):
            x = int((i / len(data)) * width)
            y = mid + int(val * mid)
            points.extend([x, y])

        if len(points) >= 4:
            canvas.create_line(points, fill="#4CAF50", width=2)

    def update_decibel_meter(self, db_level):
        """Update decibel meter visualization."""
        canvas = self.decibel_canvas
        canvas.delete("all")

        width = canvas.winfo_width()
        height = canvas.winfo_height()

        # Draw background
        canvas.create_rectangle(0, 0, width, height, fill="#1e1e1e")

        # Calculate meter width based on dB level (assuming range -60 to 0 dB)
        db_normalized = (db_level + 60) / 60  # Normalize to 0-1
        meter_width = int(width * max(0, min(1, db_normalized)))

        # Choose color based on level
        if db_level > -20:
            color = "#ff4444"  # Red for loud
        elif db_level > -40:
            color = "#ffeb3b"  # Yellow for medium
        else:
            color = "#4CAF50"  # Green for quiet

        # Draw meter
        canvas.create_rectangle(0, 0, meter_width, height, fill=color)

        # Draw dB value
        canvas.create_text(
            width // 2,
            height // 2,
            text=f"{db_level:.1f} dB",
            fill="white",
            font=("Arial", 14, "bold"),
        )

    def update_detection_visualization(self, frame, detections):
        """Update detection visualization."""
        if frame is None:
            return

        # Convert frame to PhotoImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Resize to fit canvas
        canvas_width = self.detection_canvas.winfo_width()
        canvas_height = self.detection_canvas.winfo_height()

        frame_aspect = frame.shape[1] / frame.shape[0]
        canvas_aspect = canvas_width / canvas_height

        if frame_aspect > canvas_aspect:
            new_width = canvas_width
            new_height = int(canvas_width / frame_aspect)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * frame_aspect)

        frame_pil = frame_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=frame_pil)

        # Update canvas
        self.detection_canvas.delete("all")
        self.detection_canvas.create_image(
            canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER
        )
        self.detection_canvas.image = photo  # Keep a reference

    def update_emotion_visualization(self, emotions):
        """Update emotion confidence visualization."""
        canvas = self.emotion_canvas
        canvas.delete("all")

        width = canvas.winfo_width()
        height = canvas.winfo_height()
        bar_height = 30
        spacing = 10

        y = spacing
        for emotion, confidence in emotions.items():
            # Draw bar background
            canvas.create_rectangle(
                0, y, width, y + bar_height, fill="#1e1e1e", outline="#333333"
            )

            # Draw confidence bar
            bar_width = int(width * confidence)
            canvas.create_rectangle(
                0, y, bar_width, y + bar_height, fill="#4CAF50", outline=""
            )

            # Draw label
            canvas.create_text(
                10,
                y + bar_height // 2,
                text=f"{emotion}: {confidence:.2f}",
                anchor="w",
                fill="white",
            )

            y += bar_height + spacing


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Baby Monitor System")
    parser.add_argument("--dev", action="store_true", help="Enable development mode")
    parser.add_argument("--only-web", action="store_true", help="Run only the web interface")
    parser.add_argument("--only-local", action="store_true", help="Run only the local interface")
    parser.add_argument("--mqtt-host", type=str, help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    args = parser.parse_args()

    # Initialize and start the system
    system = BabyMonitorSystem(
        dev_mode=args.dev, 
        only_local=args.only_local, 
        only_web=args.only_web,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port
    )
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()
