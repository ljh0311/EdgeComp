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
src_dir = str(Path(__file__).resolve().parent.parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Local imports
from babymonitor.camera.camera import Camera
from babymonitor.detectors.person_tracker import PersonDetector
from babymonitor.audio.audio_processor import AudioProcessor
from babymonitor.emotion.emotion import EmotionRecognizer
from babymonitor.web.web_app import BabyMonitorWeb
from .config import Config

# Configure logging
logging.basicConfig(**Config.LOGGING)
logger = logging.getLogger(__name__)


class BabyMonitorSystem:
    def __init__(self, dev_mode=False, only_local=False, only_web=False):
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
        self.initialized_components = []  # Track initialized components for cleanup
        self.web_app = None  # Initialize web_app reference as None
        self.dev_window = None  # Initialize dev window reference as None

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
            self.camera = Camera(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
            if not self.camera.initialize():
                self.logger.error("Failed to initialize camera")
                self.camera_error = True
                self.camera_enabled = False
            else:
                self.initialized_components.append("camera")
                self.logger.info("Camera initialized successfully")

            # Initialize UI if not web-only
            if not only_web:
                try:
                    self.root = tk.Tk()
                    self.setup_ui()
                    self.initialized_components.append("ui")

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
                                if current_resolution != "Not available":
                                    self.resolution_select.set(current_resolution)

                    # Create developer window if in dev mode
                    if dev_mode:
                        self.dev_window = DevWindow(self.root)
                        self.initialized_components.append("dev_window")
                except Exception as e:
                    self.logger.error(f"Failed to initialize UI: {str(e)}")
                    self.cleanup()
                    raise

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

            # Initialize web interface if not local-only
            if not only_local:
                try:
                    self.web_app = BabyMonitorWeb(dev_mode=self.dev_mode)
                    self.web_app.set_monitor_system(self)
                    self.web_app.start()
                    self.initialized_components.append("web_app")
                except Exception as e:
                    self.logger.error(f"Failed to initialize web interface: {str(e)}")
                    if only_web:  # Web interface is required in web-only mode
                        self.cleanup()
                        raise

            # Initialize audio components if enabled
            if self.audio_enabled:
                try:
                    self.audio_processor = AudioProcessor(
                        Config.AUDIO_PROCESSING, self.handle_alert
                    )
                    if not only_web:
                        self.audio_processor.set_visualization_callback(
                            self.update_audio_visualization
                        )
                    self.initialized_components.append("audio_processor")

                    # Initialize emotion recognizer
                    try:
                        self.emotion_recognizer = EmotionRecognizer()
                        if self.web_app:
                            self.emotion_recognizer.web_app = self.web_app
                        self.initialized_components.append("emotion_recognizer")
                        self.logger.info("Emotion recognizer initialized successfully")
                    except Exception as e:
                        self.logger.error(
                            f"Failed to initialize emotion recognizer: {str(e)}"
                        )
                        self.emotion_recognizer = None

                    if not only_web:
                        self.audio_status.configure(text="🎤  Audio: Ready")
                        if hasattr(self, "emotion_recognizer"):
                            self.emotion_status.configure(text="😊  Emotion: Ready")
                        else:
                            self.emotion_status.configure(
                                text="😊  Emotion: Not Available"
                            )
                except Exception as e:
                    self.logger.error(
                        f"Failed to initialize audio components: {str(e)}"
                    )
                    self.audio_enabled = False
                    if not only_web:
                        self.audio_status.configure(text="🎤  Audio: Error")

            self.logger.info("Baby Monitor System initialized successfully")

        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            self.cleanup()
            raise

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
        """Handle alerts from components."""
        if hasattr(self, "web_app"):
            self.web_app.emit_alert(level, message)

    def send_status_update(self, status_data):
        """Send status update to web interface."""
        if hasattr(self, "web_app"):
            self.web_app.emit_status(status_data)

    def start(self):
        """Start the monitoring system."""
        self.is_running = True

        # Start audio processing if enabled
        if self.audio_enabled:
            if hasattr(self, "audio_processor"):
                self.audio_processor.start()
            if hasattr(self, "emotion_recognizer"):
                self.emotion_recognizer.start()

        # Start frame processing thread
        self.process_thread = threading.Thread(target=self.process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()

        self.logger.info("Baby monitor system started")

        # Only enter tkinter mainloop if local GUI is enabled
        if not self.only_web:
            self.root.mainloop()
        else:
            # For web-only mode, keep the main thread alive
            try:
                while self.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()

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
        """Process video frames in a separate thread."""
        last_frame_time = 0
        frame_interval = 1.0 / 60.0  # Target 60 FPS
        frame_count = 0
        fps_update_time = time.time()

        # Pre-allocate reusable frame buffer
        processed_frame = None
        resized_frame = None

        while self.is_running:
            try:
                frame_start_time = time.time()

                if not self.camera_enabled:
                    time.sleep(0.001)
                    continue

                current_time = time.time()

                # Skip frame if we're processing too fast
                time_since_last_frame = current_time - last_frame_time
                if time_since_last_frame < frame_interval:
                    sleep_time = frame_interval - time_since_last_frame
                    if sleep_time > 0.001:  # Only sleep for meaningful intervals
                        time.sleep(sleep_time)
                    continue

                # Get frame from camera
                ret, frame = self.camera.get_frame()
                if not ret or frame is None:
                    if not self.only_web and hasattr(self, "camera_status"):
                        self.camera_status.configure(text="📷  Camera: No Signal")
                    time.sleep(0.001)
                    continue

                # Process frame with person detector if available
                if (
                    hasattr(self, "person_detector")
                    and self.person_detector is not None
                ):
                    try:
                        detections = self.person_detector.detect(frame)
                        if not self.only_web and hasattr(self, "detection_status"):
                            self.detection_status.configure(
                                text=f"👀  Detection: {len(detections)} people"
                            )
                    except Exception as e:
                        self.logger.error(f"Person detection error: {str(e)}")
                        if not self.only_web and hasattr(self, "detection_status"):
                            self.detection_status.configure(text="👀  Detection: Error")
                        detections = []
                else:
                    detections = []

                # Process frame with motion detector if available
                if (
                    hasattr(self, "motion_detector")
                    and self.motion_detector is not None
                ):
                    try:
                        processed_frame, rapid_motion, fall_detected = (
                            self.motion_detector.detect(frame, detections)
                        )

                        # Update web interface with detection results
                        if self.web_app is not None:
                            self.web_app.emit_detection(
                                {
                                    "people_count": len(detections),
                                    "rapid_motion": rapid_motion,
                                    "fall_detected": fall_detected,
                                }
                            )
                    except Exception as e:
                        self.logger.error(f"Motion detection error: {str(e)}")
                        processed_frame = frame
                else:
                    processed_frame = frame

                # Update performance metrics
                frame_count += 1
                frame_end_time = time.time()
                frame_processing_time = frame_end_time - frame_start_time

                with self.metrics_lock:
                    self.frame_times.append(
                        frame_processing_time * 1000
                    )  # Convert to ms

                    # Update FPS every second
                    if current_time - fps_update_time >= 1.0:
                        fps = frame_count / (current_time - fps_update_time)
                        self.fps_history.append(fps)
                        frame_count = 0
                        fps_update_time = current_time

                    # Update system metrics every second
                    if (
                        current_time - self.last_metrics_update
                        >= self.metrics_update_interval
                    ):
                        self.cpu_history.append(psutil.cpu_percent())
                        self.memory_history.append(psutil.Process().memory_percent())
                        self.last_metrics_update = current_time

                        # Emit metrics to web interface
                        if self.web_app is not None:
                            self.web_app.emit_metrics(
                                {
                                    "fps": (
                                        self.fps_history[-1] if self.fps_history else 0
                                    ),
                                    "frame_time": (
                                        sum(self.frame_times) / len(self.frame_times)
                                        if self.frame_times
                                        else 0
                                    ),
                                    "cpu_usage": (
                                        self.cpu_history[-1] if self.cpu_history else 0
                                    ),
                                    "memory_usage": (
                                        self.memory_history[-1]
                                        if self.memory_history
                                        else 0
                                    ),
                                }
                            )

                # Update web interface
                if self.web_app is not None and self.camera_enabled:
                    try:
                        self.web_app.emit_frame(processed_frame)
                    except Exception as e:
                        self.logger.error(
                            f"Error sending frame to web interface: {str(e)}"
                        )

                # Update local GUI if not in web-only mode
                if not self.only_web:
                    try:
                        # Get current canvas dimensions
                        canvas_width = self.video_canvas.winfo_width()
                        canvas_height = self.video_canvas.winfo_height()

                        if canvas_width > 1 and canvas_height > 1:
                            # Calculate target dimensions
                            frame_aspect = (
                                processed_frame.shape[1] / processed_frame.shape[0]
                            )
                            canvas_aspect = canvas_width / canvas_height

                            if frame_aspect > canvas_aspect:
                                new_width = canvas_width
                                new_height = int(canvas_width / frame_aspect)
                            else:
                                new_height = canvas_height
                                new_width = int(canvas_height * frame_aspect)

                            # Only resize if dimensions have changed
                            if (
                                resized_frame is None
                                or resized_frame.shape[1] != new_width
                                or resized_frame.shape[0] != new_height
                            ):
                                resized_frame = cv2.resize(
                                    processed_frame,
                                    (new_width, new_height),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            else:
                                # Reuse existing buffer
                                cv2.resize(
                                    processed_frame,
                                    (new_width, new_height),
                                    dst=resized_frame,
                                    interpolation=cv2.INTER_NEAREST,
                                )

                            # Convert to RGB and create PhotoImage
                            # Use numpy operations instead of cv2.cvtColor for better performance
                            frame_rgb = np.empty(resized_frame.shape, dtype=np.uint8)
                            frame_rgb[..., 0] = resized_frame[..., 2]
                            frame_rgb[..., 1] = resized_frame[..., 1]
                            frame_rgb[..., 2] = resized_frame[..., 0]

                            frame_pil = Image.fromarray(frame_rgb)
                            photo = ImageTk.PhotoImage(image=frame_pil)

                            # Update canvas in main thread
                            def update_canvas(photo):
                                if not self.is_running:
                                    return
                                try:
                                    self.video_canvas.delete("all")
                                    self.video_canvas.create_image(
                                        canvas_width // 2,
                                        canvas_height // 2,
                                        image=photo,
                                        anchor=tk.CENTER,
                                    )
                                    self.video_canvas.image = photo
                                except Exception as e:
                                    self.logger.error(
                                        f"Error updating canvas: {str(e)}"
                                    )

                            self.root.after(1, update_canvas, photo)

                    except Exception as e:
                        self.logger.error(f"Error updating video display: {str(e)}")

                last_frame_time = current_time

            except Exception as e:
                self.logger.error(f"Error in frame processing loop: {str(e)}")
                time.sleep(0.001)

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

    def cleanup(self):
        """Clean up initialized components in reverse order."""
        self.is_running = False

        for component in reversed(self.initialized_components):
            try:
                if component == "web_app":
                    if hasattr(self, "web_app") and self.web_app is not None:
                        self.web_app.stop()
                elif component == "camera":
                    if hasattr(self, "camera") and self.camera is not None:
                        self.camera.cleanup()
                elif component == "audio_processor":
                    if (
                        hasattr(self, "audio_processor")
                        and self.audio_processor is not None
                    ):
                        self.audio_processor.stop()
                elif component == "emotion_recognizer":
                    if (
                        hasattr(self, "emotion_recognizer")
                        and self.emotion_recognizer is not None
                    ):
                        self.emotion_recognizer.stop()
                elif component == "dev_window":
                    if hasattr(self, "dev_window") and self.dev_window is not None:
                        self.dev_window.window.destroy()
                elif component == "ui":
                    if hasattr(self, "root") and self.root is not None:
                        self.root.quit()
            except Exception as e:
                self.logger.error(f"Error cleaning up {component}: {str(e)}")

        self.initialized_components.clear()
        self.logger.info("Cleanup completed")


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
    """Main entry point for the Baby Monitor System."""
    parser = argparse.ArgumentParser(description="Baby Monitor System")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in developer mode with local GUI and debug window",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web interface (default: 5000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the web interface (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    try:
        # Initialize the system with web-only mode by default
        monitor = BabyMonitorSystem(
            dev_mode=args.dev,
            only_local=False,  # Always false as we want web interface
            only_web=not args.dev,  # Web-only if not in dev mode
        )

        # Start the monitor system
        monitor.start()

        if args.dev:
            # In dev mode, run with local GUI
            monitor.root.mainloop()
        else:
            # In production, run web-only mode
            print(f"\nBaby Monitor System started!")
            print(f"Web interface available at: http://{args.host}:{args.port}")
            print("\nPress Ctrl+C to stop the system...")

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                monitor.stop()

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if "monitor" in locals():
            monitor.cleanup()


if __name__ == "__main__":
    main()
