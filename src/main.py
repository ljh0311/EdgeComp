"""
Baby Monitor System
=================
Main application module for the Baby Monitor System.
"""

import argparse
from datetime import datetime
import os
import sys
import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Local imports
from src.audio.audio_processor import AudioProcessor
from src.camera.camera import Camera
from src.detectors.motion_mog2 import MotionDetector
from src.detectors.sound_hubert import EmotionDetector
from src.detectors.vision_yolo import PersonDetector
from src.utils.config import Config
from src.utils.system_monitor import SystemMonitor
from src.web.web_app import BabyMonitorWeb
from src.detectors import AVAILABLE_SOUND_DETECTORS

# Configure logging
logging.basicConfig(**Config.LOGGING)
logger = logging.getLogger(__name__)

def setup_gpu():
    """Setup GPU if available."""
    if torch.cuda.is_available():
        # Set up CUDA device
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        return device
    else:
        logger.info("No GPU available, using CPU")
        return torch.device("cpu")

class BabyMonitorSystem:
    def __init__(self, dev_mode=False):
        """Initialize the Baby Monitor System."""
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.camera_error = False
        self.audio_enabled = True
        self.camera_enabled = False  # Start with camera disabled
        self.frame_lock = threading.Lock()
        self.dev_mode = dev_mode
        self.waveform_queue = queue.Queue()

        # Setup GPU if available
        self.device = setup_gpu()

        # Initialize UI first
        self.root = tk.Tk()
        self.setup_ui()

        try:
            # Initialize web interface in a separate thread
            self.web_app = BabyMonitorWeb(dev_mode=self.dev_mode, monitor_system=self)
            self.web_thread = threading.Thread(target=self.web_app.start, daemon=True)
            self.web_thread.start()

            # Initialize camera but don't start it yet
            self.camera = Camera(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
            self.update_camera_list()
            self.update_resolution_list()
            self.camera_status.configure(text="📷  Camera: Ready")

            # Initialize person detector with GPU support
            self.person_detector = PersonDetector(device=self.device)

            # Initialize motion detector with GPU support
            self.motion_detector = MotionDetector(Config.MOTION_DETECTION, device=self.device)

            # Initialize audio components if enabled
            if self.audio_enabled:
                audio_config = Config.AUDIO_PROCESSING.copy()
                audio_config['device'] = self.device
                self.audio_processor = AudioProcessor(
                    audio_config, self.handle_alert
                )
                self.audio_processor.set_visualization_callback(
                    self.update_waveform_data
                )
                
                # Initialize with default model (HuBERT)
                self.current_emotion_detector = None
                self.initialize_emotion_detector('hubert')
                
                self.audio_status.configure(text="🎤  Audio: Ready")
                self.update_emotion_status()

            # Initialize system monitor
            self.monitor = SystemMonitor()

            # Log initialization status
            if torch.cuda.is_available():
                self.logger.info(f"Baby Monitor System initialized with GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.logger.info("Baby Monitor System initialized with CPU")

        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            raise

    def setup_ui(self):
        """Setup the main UI window."""
        self.root.title("Baby Monitor")
        self.root.geometry("1200x800")

        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left panel (video feed and waveform)
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Video feed frame
        video_frame = ttk.Frame(left_panel)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.video_canvas = tk.Canvas(video_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Waveform frame
        self.waveform_frame = ttk.Frame(left_panel, height=200)
        self.waveform_frame.pack(fill=tk.X, pady=(0, 10))
        self.waveform_frame.pack_propagate(False)

        # Right panel (controls and status)
        right_panel = ttk.Frame(main_container, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        right_panel.pack_propagate(False)

        # Controls section
        controls_frame = ttk.LabelFrame(right_panel, text="Controls")
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        # Model selection frame
        model_frame = ttk.Frame(controls_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        # Model selection combobox
        ttk.Label(model_frame, text="Emotion Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_select = ttk.Combobox(
            model_frame, state="readonly", width=20
        )
        self.model_select['values'] = [info['name'] for info in AVAILABLE_SOUND_DETECTORS.values()]
        self.model_select.set(AVAILABLE_SOUND_DETECTORS['hubert']['name'])
        self.model_select.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.model_select.bind("<<ComboboxSelected>>", self.on_model_selected)

        # Model description (for dev mode)
        self.model_desc = ttk.Label(controls_frame, text="", wraplength=250)
        self.model_desc.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Camera selection frame
        camera_selection_frame = ttk.Frame(controls_frame)
        camera_selection_frame.pack(fill=tk.X, padx=5, pady=5)

        # Camera selection combobox
        ttk.Label(camera_selection_frame, text="Camera:").pack(
            side=tk.LEFT, padx=(0, 5)
        )
        self.camera_select = ttk.Combobox(
            camera_selection_frame, state="readonly", width=15
        )
        self.camera_select.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.camera_select.bind("<<ComboboxSelected>>", self.on_camera_selected)

        # Resolution selection frame
        resolution_frame = ttk.Frame(controls_frame)
        resolution_frame.pack(fill=tk.X, padx=5, pady=5)

        # Resolution selection combobox
        ttk.Label(resolution_frame, text="Resolution:").pack(side=tk.LEFT, padx=(0, 5))
        self.resolution_select = ttk.Combobox(
            resolution_frame, state="readonly", width=15
        )
        self.resolution_select.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.resolution_select.bind("<<ComboboxSelected>>", self.on_resolution_selected)

        # Camera toggle button
        self.camera_btn = ttk.Button(
            controls_frame, text="Camera Feed", command=self.toggle_camera
        )
        self.camera_btn.pack(fill=tk.X, padx=5, pady=5)

        # Audio toggle button
        self.audio_btn = ttk.Button(
            controls_frame, text="Audio Monitor", command=self.toggle_audio
        )
        self.audio_btn.pack(fill=tk.X, padx=5, pady=5)

        # Status section
        status_frame = ttk.LabelFrame(right_panel, text="Status")
        status_frame.pack(fill=tk.X, pady=(0, 10))

        # Status labels
        self.camera_status = ttk.Label(status_frame, text="📷  Camera: Initializing...")
        self.camera_status.pack(anchor=tk.W, padx=5, pady=2)

        self.audio_status = ttk.Label(status_frame, text="🎤  Audio: Initializing...")
        self.audio_status.pack(anchor=tk.W, padx=5, pady=2)

        self.emotion_status = ttk.Label(
            status_frame, text="😊  Emotion: Initializing..."
        )
        self.emotion_status.pack(anchor=tk.W, padx=5, pady=2)

        self.detection_status = ttk.Label(
            status_frame, text="👀  Detection: Initializing..."
        )
        self.detection_status.pack(anchor=tk.W, padx=5, pady=2)

        # Initialize matplotlib for waveform
        self.setup_waveform()

        # Initialize camera selection
        self.update_camera_list()

    def setup_waveform(self):
        """Setup waveform visualization."""
        self.waveform_figure = Figure(figsize=(6, 2), dpi=100)
        self.waveform_canvas = FigureCanvasTkAgg(
            self.waveform_figure, master=self.waveform_frame
        )
        self.waveform_canvas.draw()
        self.waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Setup the plot
        self.waveform_ax = self.waveform_figure.add_subplot(111)
        self.waveform_ax.set_ylim([-1, 1])
        self.waveform_ax.set_xlim([0, 1000])
        (self.waveform_line,) = self.waveform_ax.plot([], [], lw=2)
        self.waveform_data = np.zeros(1000)

        # Remove margins and grid
        self.waveform_figure.tight_layout(pad=0.1)
        self.waveform_ax.grid(False)

        # Bind resize event
        self.waveform_frame.bind("<Configure>", self._set_waveform_height)

    def toggle_camera(self):
        """Toggle camera feed on/off."""
        try:
            if not self.camera_enabled:
                # Trying to enable camera
                self.camera_status.configure(text="📷  Camera: Initializing...")
                
                # Ensure camera is initialized
                if not hasattr(self, 'camera') or self.camera is None:
                    self.camera = Camera(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
                
                # Try to initialize camera with detailed logging
                self.logger.info("Initializing camera...")
                if not self.camera.initialize():
                    self.logger.error("Failed to initialize camera")
                    self.camera_status.configure(text="📷  Camera: Failed to Initialize")
                    return
                
                # Read a test frame to verify camera is working
                self.logger.info("Testing camera frame capture...")
                ret, frame = self.camera.get_frame()
                if not ret or frame is None:
                    self.logger.error("Camera initialized but failed to get frame")
                    self.camera_status.configure(text="📷  Camera: No Video Signal")
                    return
                
                self.logger.info(f"Test frame captured successfully: shape={frame.shape}")
                
                # Camera is working, enable it
                self.camera_enabled = True
                self.camera_btn.configure(text="Stop Camera")
                self.camera_status.configure(text="📷  Camera: Active")
                self.web_app.emit_status({"camera_enabled": True})
                
                # Start frame processing if not already running
                if not hasattr(self, 'frame_thread') or not self.frame_thread.is_alive():
                    self.start_processing_threads()
                
            else:
                # Disabling camera
                self.logger.info("Stopping camera...")
                self.camera_enabled = False
                self.camera_btn.configure(text="Start Camera")
                self.camera_status.configure(text="📷  Camera: Disabled")
                # Clear the video canvas
                self.video_canvas.delete("all")
                if hasattr(self, 'camera'):
                    self.camera.release()
                self.web_app.emit_status({"camera_enabled": False})

        except Exception as e:
            self.logger.error(f"Error toggling camera: {str(e)}")
            self.camera_enabled = False
            self.camera_btn.configure(text="Start Camera")
            self.camera_status.configure(text="📷  Camera: Error")
            if hasattr(self, 'camera'):
                self.camera.release()

    def toggle_audio(self):
        """Toggle audio monitoring on/off."""
        try:
            if not self.audio_enabled:
                # Starting audio
                self.logger.info("Starting audio monitoring...")
                
                # Initialize audio processor if needed
                if not hasattr(self, 'audio_processor'):
                    audio_config = Config.AUDIO_PROCESSING.copy()
                    audio_config['device'] = self.device
                    self.audio_processor = AudioProcessor(
                        audio_config, self.handle_alert
                    )
                    self.audio_processor.set_visualization_callback(
                        self.update_waveform_data
                    )
                
                # Start audio processing
                if self.audio_processor:
                    self.audio_processor.start()
                    self.audio_enabled = True
                    self.audio_btn.configure(text="Stop Audio")
                    self.audio_status.configure(text="🎤  Audio: Active")
                    
                    # Start emotion detector if available
                    if hasattr(self, "current_emotion_detector") and self.current_emotion_detector:
                        try:
                            self.current_emotion_detector.start()
                        except Exception as e:
                            self.logger.warning(f"Non-critical error starting emotion detector: {str(e)}")
                    
                    self.web_app.emit_status({"audio_enabled": True})
                    
            else:
                # Stopping audio
                self.logger.info("Stopping audio monitoring...")
                self.audio_enabled = False
                
                # Stop emotion detector first
                if hasattr(self, "current_emotion_detector") and self.current_emotion_detector:
                    try:
                        self.current_emotion_detector.stop()
                    except Exception as e:
                        self.logger.warning(f"Non-critical error stopping emotion detector: {str(e)}")
                
                # Stop audio processor
                if hasattr(self, "audio_processor") and self.audio_processor:
                    try:
                        self.audio_processor.stop()
                    except Exception as e:
                        self.logger.warning(f"Non-critical error stopping audio processor: {str(e)}")
                
                self.audio_btn.configure(text="Start Audio")
                self.audio_status.configure(text="🎤  Audio: Disabled")
                self.web_app.emit_status({"audio_enabled": False})
                
        except Exception as e:
            self.logger.error(f"Error toggling audio: {str(e)}")
            self.audio_status.configure(text="🎤  Audio: Error")
            # Try to cleanup on error
            if hasattr(self, "audio_processor"):
                try:
                    self.audio_processor.stop()
                except:
                    pass

    def handle_alert(self, message, level="info"):
        """Handle alerts from components."""
        if hasattr(self, "web_app"):
            self.web_app.emit_alert(level, message)

    def send_status_update(self, status_data):
        """Send status update to web interface."""
        if hasattr(self, "web_app"):
            self.web_app.emit_status(status_data)

    def start_processing_threads(self):
        """Start all processing threads."""
        try:
            # Start frame processing thread
            self.frame_thread = threading.Thread(target=self.process_frames, daemon=True)
            self.frame_thread.start()
            self.logger.info("Started frame processing thread")
            
            # Start waveform processing
            self.root.after(10, self.process_waveform_queue)
            self.logger.info("Started waveform processing")
            
        except Exception as e:
            self.logger.error(f"Error starting processing threads: {str(e)}")

    def start(self):
        """Start the monitoring system."""
        if self.is_running:
            self.logger.warning("System is already running")
            return

        try:
            self.is_running = True
            self.logger.info("Starting monitoring system...")

            # Ensure camera is initialized and enabled
            if not hasattr(self, 'camera') or self.camera is None:
                self.logger.info("Initializing camera...")
                self.camera = Camera(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
                if not self.camera.initialize():
                    self.logger.error("Failed to initialize camera")
                    self.camera_error = True
                else:
                    self.camera_enabled = True
                    self.logger.info("Camera initialized successfully")

            # Start processing threads
            self.start_processing_threads()

            # Start audio processing if enabled
            if self.audio_processor and self.audio_enabled:
                self.audio_processor.start()

            # Start emotion detector if available
            if self.current_emotion_detector:
                try:
                    self.current_emotion_detector.start()
                    self.logger.info("Started emotion detector")
                except Exception as e:
                    self.logger.error(f"Failed to start emotion detector: {str(e)}")
                    self.initialize_emotion_detector('basic')

            self.update_status()
            self.logger.info("Monitoring system started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start monitoring system: {str(e)}")
            self.stop()  # Clean up if start fails

    def stop(self):
        """Stop the monitoring system."""
        self.is_running = False

        # Stop audio components with timeout handling
        try:
            if hasattr(self, "audio_processor"):
                self.audio_processor.stop()
                # Give audio processor a short time to stop
                time.sleep(0.5)
        except Exception as e:
            self.logger.warning(f"Non-critical error stopping audio processor: {str(e)}")

        try:
            if hasattr(self, "current_emotion_detector"):
                self.current_emotion_detector.stop()
                # Give emotion recognizer a short time to stop
                time.sleep(0.5)
        except Exception as e:
            self.logger.warning(f"Non-critical error stopping emotion recognizer: {str(e)}")

        # Stop web interface
        if hasattr(self, "web_app"):
            self.web_app.stop()

        # Release camera
        if hasattr(self, "camera"):
            self.camera.release()

        self.root.quit()
        self.logger.info("Baby monitor system stopped")

    def process_frames(self):
        """Process video frames in a separate thread."""
        last_frame_time = 0
        frame_interval = 1.0 / 30.0  # 30 FPS target
        error_count = 0
        max_errors = 3
        frame_count = 0
        last_detection_time = 0
        detection_interval = 0.2  # Reduced detection frequency to 5 FPS
        last_ui_update = 0
        ui_update_interval = 0.033  # Update UI at ~30fps
        last_web_update = 0
        web_update_interval = 0.1  # Update web interface every 100ms

        # Frame buffer for smoother display
        self.current_frame = None
        self.frame_lock = threading.Lock()

        self.logger.info("Frame processing thread started")

        while self.is_running:
            try:
                if not self.camera_enabled or not hasattr(self, 'camera'):
                    time.sleep(0.1)
                    continue

                current_time = time.time()

                # Frame rate control - sleep if we're ahead of schedule
                if current_time - last_frame_time < frame_interval:
                    time.sleep(max(0.001, (frame_interval - (current_time - last_frame_time)) * 0.95))
                    continue

                # Get frame with detailed logging
                ret, frame = self.camera.get_frame()
                
                if not ret or frame is None:
                    error_count += 1
                    self.logger.warning(f"Failed to get frame (attempt {error_count}/{max_errors})")
                    if error_count >= max_errors:
                        self.root.after(0, lambda: self.camera_status.configure(text="📷  Camera: Critical Error"))
                        error_count = 0
                    time.sleep(0.1)
                    continue

                error_count = 0
                frame_count += 1
                last_frame_time = current_time

                # Process detections at a lower frequency
                if current_time - last_detection_time >= detection_interval and hasattr(self, 'person_detector'):
                    try:
                        with self.frame_lock:
                            # Process frame with person detector
                            processed_frame, status_text = self.person_detector.process_frame(frame.copy())
                            
                            # Update detection status in UI
                            self.root.after(0, lambda t=status_text: self.detection_status.configure(text=f"👀  Detection: {t}"))
                            
                            # Store the processed frame
                            self.current_frame = processed_frame
                        
                    except Exception as e:
                        if self.dev_mode:
                            self.logger.error(f"Error in detection processing: {str(e)}")
                    
                    last_detection_time = current_time

                # Update UI with camera feed
                if current_time - last_ui_update >= ui_update_interval:
                    try:
                        # Use the current processed frame if available, otherwise use the raw frame
                        display_frame = None
                        with self.frame_lock:
                            if self.current_frame is not None:
                                display_frame = self.current_frame.copy()
                            else:
                                display_frame = frame.copy()

                        # Convert frame to RGB and resize efficiently
                        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        
                        # Get canvas dimensions once
                        canvas_width = self.video_canvas.winfo_width()
                        canvas_height = self.video_canvas.winfo_height()

                        if canvas_width > 1 and canvas_height > 1:
                            # Calculate aspect ratios
                            frame_aspect = frame_rgb.shape[1] / frame_rgb.shape[0]
                            canvas_aspect = canvas_width / canvas_height

                            # Calculate new dimensions maintaining aspect ratio
                            if frame_aspect > canvas_aspect:
                                new_width = canvas_width
                                new_height = int(canvas_width / frame_aspect)
                            else:
                                new_height = canvas_height
                                new_width = int(canvas_height * frame_aspect)

                            # Only resize if dimensions have changed significantly
                            if (abs(new_width - frame_rgb.shape[1]) > 10 or 
                                abs(new_height - frame_rgb.shape[0]) > 10):
                                frame_rgb = cv2.resize(
                                    frame_rgb, 
                                    (new_width, new_height),
                                    interpolation=cv2.INTER_LINEAR
                                )

                            # Update UI in the main thread
                            frame_pil = Image.fromarray(frame_rgb)
                            photo = ImageTk.PhotoImage(image=frame_pil)
                            self.root.after(0, lambda p=photo, w=canvas_width, h=canvas_height: 
                                          self.update_video_frame(p, w, h))

                            last_ui_update = current_time

                    except Exception as e:
                        if self.dev_mode:
                            self.logger.error(f"Error updating video display: {str(e)}")

                # Update web interface
                if current_time - last_web_update >= web_update_interval and hasattr(self, 'web_app'):
                    try:
                        with self.frame_lock:
                            if self.current_frame is not None:
                                self.web_app.emit_frame(self.current_frame)
                        last_web_update = current_time
                    except Exception as e:
                        if self.dev_mode:
                            self.logger.error(f"Error sending frame to web interface: {str(e)}")

            except Exception as e:
                self.logger.error(f"Error in frame processing loop: {str(e)}")
                error_count += 1
                if error_count >= max_errors:
                    self.root.after(0, lambda: self.camera_status.configure(text="📷  Camera: Critical Error"))
                    error_count = 0
                time.sleep(0.1)

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

    def update_waveform_data(self, audio_data):
        """Queue the waveform data for update in the main thread."""
        try:
            # Resize audio data to match waveform buffer size
            if len(audio_data) != len(self.waveform_data):
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), len(self.waveform_data)),
                    np.arange(len(audio_data)),
                    audio_data,
                )

            # Put the data in the queue
            self.waveform_queue.put(audio_data)

            # Schedule the waveform update in the main thread
            self.root.after(1, self.process_waveform_queue)
        except Exception as e:
            self.logger.error(f"Error queueing waveform data: {str(e)}")

    def process_waveform_queue(self):
        """Process queued waveform data in the main thread."""
        try:
            while not self.waveform_queue.empty():
                audio_data = self.waveform_queue.get_nowait()

                # Update waveform data
                self.waveform_data = np.roll(self.waveform_data, -len(audio_data))
                self.waveform_data[-len(audio_data) :] = audio_data

                # Update the plot
                if hasattr(self, "waveform_line"):
                    self.waveform_line.set_data(
                        range(len(self.waveform_data)), self.waveform_data
                    )
                    self.waveform_canvas.draw_idle()
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing waveform queue: {str(e)}")

        # Schedule the next queue check if the system is still running
        if self.is_running:
            self.root.after(10, self.process_waveform_queue)

    def update_camera_list(self):
        """Update the camera list in the combobox."""
        try:
            if not hasattr(self, 'camera') or self.camera is None:
                self.camera_select["values"] = []
                self.camera_select.set("")
                return

            camera_list = self.camera.get_camera_list()
            if camera_list:
                self.camera_select["values"] = camera_list
                current_camera = camera_list[self.camera.selected_camera_index]
                self.camera_select.set(current_camera)
                self.update_resolution_list(current_camera)
        except Exception as e:
            self.logger.error(f"Error updating camera list: {str(e)}")
            self.camera_select["values"] = []
            self.camera_select.set("")

    def update_resolution_list(self, camera_name=None):
        """Update the resolution list in the combobox."""
        try:
            if not hasattr(self, 'camera') or self.camera is None:
                self.resolution_select["values"] = []
                self.resolution_select.set("")
                return

            if camera_name is None:
                camera_name = self.camera_select.get()
            resolutions = self.camera.get_camera_resolutions(camera_name)
            if resolutions:
                self.resolution_select["values"] = resolutions
                current_resolution = self.camera.get_current_resolution()
                self.resolution_select.set(current_resolution)
        except Exception as e:
            self.logger.error(f"Error updating resolution list: {str(e)}")
            self.resolution_select["values"] = []
            self.resolution_select.set("")

    def on_camera_selected(self, event):
        """Handle camera selection."""
        try:
            selected_camera = self.camera_select.get()
            if selected_camera:
                if self.camera.select_camera(selected_camera):
                    self.camera_status.configure(text="📷  Camera: Ready")
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
            if selected_resolution:
                self.camera_status.configure(text="📷  Camera: Changing Resolution...")
                if self.camera.set_resolution(selected_resolution):
                    # Verify camera is still working with new resolution
                    ret, frame = self.camera.get_frame()
                    if ret and frame is not None:
                        self.camera_status.configure(text="📷  Camera: Active")
                    else:
                        self.camera_status.configure(text="📷  Camera: No Video Signal")
                else:
                    self.camera_status.configure(text="📷  Camera: Resolution Error")
        except Exception as e:
            self.logger.error(f"Error setting resolution: {str(e)}")
            self.camera_status.configure(text="📷  Camera: Resolution Error")

    def update_video_frame(self, photo, canvas_width, canvas_height):
        """Update video frame in the main thread."""
        if not self.is_running or not self.camera_enabled:
            return
        
        try:
            # Clear previous frame
            self.video_canvas.delete("all")
            
            # Calculate center position
            x = canvas_width // 2
            y = canvas_height // 2
            
            # Create new image
            self.video_canvas.create_image(x, y, image=photo, anchor=tk.CENTER)
            
            # Keep a reference to prevent garbage collection
            self.video_canvas.image = photo
            
            # Log frame update (in dev mode)
            if self.dev_mode:
                self.logger.debug(f"Updated frame display: size={photo.width()}x{photo.height()}")
            
        except Exception as e:
            self.logger.error(f"Error updating video frame: {str(e)}")

    def initialize_emotion_detector(self, model_key):
        """Initialize the selected emotion detector."""
        try:
            if self.current_emotion_detector:
                # Stop the current detector if it's running
                try:
                    self.current_emotion_detector.stop()
                except:
                    pass
                self.current_emotion_detector = None

            # Get the detector class and create instance
            detector_info = AVAILABLE_SOUND_DETECTORS.get(model_key)
            if not detector_info:
                raise ValueError(f"Unknown model key: {model_key}")
                
            detector_class = detector_info['class']
            
            try:
                # Initialize detector with proper config
                config = Config.EMOTION_DETECTION.copy()
                config['device'] = self.device  # Ensure device is set
                
                if model_key == 'hubert':
                    # Ensure model path is absolute
                    config['model_path'] = str(Config.MODELS_DIR)
                
                self.current_emotion_detector = detector_class(config, self.web_app)
                
                # Update status and UI
                self.update_emotion_status()
                self.model_select.set(detector_info['name'])
                
                if self.dev_mode:
                    # Show supported emotions in dev mode
                    emotions = ", ".join(self.current_emotion_detector.supported_emotions)
                    desc = f"{detector_info['description']}\nEmotions: {emotions}"
                    self.model_desc.configure(text=desc)
                else:
                    self.model_desc.configure(text="")
                    
                self.logger.info(f"Successfully initialized {detector_info['name']} model")
                    
            except Exception as model_error:
                self.logger.error(f"Failed to initialize {detector_info['name']} model: {str(model_error)}")
                
                # If this was HuBERT, try falling back to Basic model
                if model_key == 'hubert':
                    self.logger.info("Falling back to Basic model...")
                    return self.initialize_emotion_detector('basic')
                else:
                    # If we're already on the basic model or another model, raise the error
                    raise
                
        except Exception as e:
            self.logger.error(f"Error initializing emotion detector: {str(e)}")
            self.emotion_status.configure(text="😢 Emotion: Error")
            if self.dev_mode:
                self.model_desc.configure(text=f"Error: {str(e)}")
            
            # Ensure we don't have a None detector
            if not self.current_emotion_detector:
                self.logger.info("Initializing with Basic model as last resort...")
                try:
                    config = Config.EMOTION_DETECTION.copy()
                    config['device'] = self.device
                    self.current_emotion_detector = AVAILABLE_SOUND_DETECTORS['basic']['class'](
                        config, self.web_app
                    )
                except Exception as basic_error:
                    self.logger.error(f"Failed to initialize Basic model: {str(basic_error)}")
                    self.current_emotion_detector = None

    def update_emotion_status(self):
        """Update the emotion detection status in the UI."""
        if not self.current_emotion_detector:
            self.emotion_status.configure(text="😐 Emotion: Not Initialized")
            return
            
        status = f"😊 Emotion: {self.current_emotion_detector.model_name}"
        if self.dev_mode:
            status += f" ({self.current_emotion_detector.__class__.__name__})"
        self.emotion_status.configure(text=status)

    def on_model_selected(self, event):
        """Handle emotion model selection."""
        try:
            # Find the model key from the selected name
            selected_name = self.model_select.get()
            model_key = next(
                key for key, info in AVAILABLE_SOUND_DETECTORS.items()
                if info['name'] == selected_name
            )
            
            # Initialize the selected model
            self.initialize_emotion_detector(model_key)
            
        except Exception as e:
            self.logger.error(f"Error changing emotion model: {str(e)}")
            self.emotion_status.configure(text="😢 Emotion: Error")
            if self.dev_mode:
                self.model_desc.configure(text=f"Error: {str(e)}")

    def update_status(self):
        """Update all status indicators."""
        try:
            # Update camera status
            if self.camera_enabled:
                self.camera_status.configure(text="  Camera: Active")
            else:
                self.camera_status.configure(text="📷  Camera: Disabled")

            # Update audio status
            if self.audio_enabled:
                if hasattr(self, 'audio_processor') and self.audio_processor:
                    self.audio_status.configure(text="🎤  Audio: Active")
                else:
                    self.audio_status.configure(text="🎤  Audio: Error")
            else:
                self.audio_status.configure(text="🎤  Audio: Disabled")

            # Update emotion status
            if hasattr(self, 'current_emotion_detector') and self.current_emotion_detector:
                self.update_emotion_status()
            else:
                self.emotion_status.configure(text="😐 Emotion: Not Initialized")

            # Update detection status
            if hasattr(self, 'person_detector'):
                self.detection_status.configure(text="👀  Detection: Ready")
            else:
                self.detection_status.configure(text="👀  Detection: Not Initialized")

            # Send status to web interface
            if hasattr(self, 'web_app'):
                self.web_app.emit_status({
                    'camera_enabled': self.camera_enabled,
                    'audio_enabled': self.audio_enabled,
                    'emotion_enabled': hasattr(self, 'current_emotion_detector') and self.current_emotion_detector is not None,
                    'detection_enabled': hasattr(self, 'person_detector')
                })

        except Exception as e:
            self.logger.error(f"Error updating status: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Baby Monitor System")
    parser.add_argument("--dev", action="store_true", help="Run in developer mode")
    args = parser.parse_args()

    try:
        app = BabyMonitorSystem(dev_mode=args.dev)
        app.start()

        # Start the Tkinter event loop
        try:
            app.root.mainloop()
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        finally:
            app.stop()

    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
