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
from src.detectors.motion_detector import MotionDetector
from src.emotion.emotion_recognizer import EmotionRecognizer
from src.detectors.person_detector import PersonDetector
from src.utils.config import Config
from src.utils.system_monitor import SystemMonitor
from src.web.web_app import BabyMonitorWeb

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
        self.camera_enabled = True
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

            # Initialize camera in a separate thread
            def init_camera():
                self.camera = Camera(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
                if not self.camera.initialize():
                    self.logger.error("Failed to initialize camera")
                    self.camera_error = True
                    self.root.after(0, lambda: self.camera_status.configure(text="📷  Camera: Error"))
                else:
                    self.root.after(0, lambda: self.camera_status.configure(text="📷  Camera: Ready"))
                    # Update camera selection UI
                    self.root.after(0, self.update_camera_list)
                    self.root.after(0, self.update_resolution_list)

            camera_thread = threading.Thread(target=init_camera, daemon=True)
            camera_thread.start()

            # Initialize person detector with GPU support
            self.person_detector = PersonDetector(device=self.device)

            # Initialize motion detector with GPU support
            self.motion_detector = MotionDetector(Config.MOTION_DETECTION, device=self.device)

            # Initialize audio components if enabled
            if self.audio_enabled:
                self.audio_processor = AudioProcessor(
                    Config.AUDIO_PROCESSING, self.handle_alert
                )
                self.audio_processor.set_visualization_callback(
                    self.update_waveform_data
                )
                self.emotion_recognizer = EmotionRecognizer(web_app=self.web_app)
                self.audio_status.configure(text="🎤  Audio: Ready")
                self.emotion_status.configure(text="😊  Emotion: Ready")

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
                self.camera_enabled = True
                self.camera_btn.configure(text="Stop Camera")
                self.camera_status.configure(text="📷  Camera: Initializing...")

                # Ensure camera is initialized
                if not hasattr(self, 'camera') or self.camera is None:
                    self.camera = Camera(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
                
                if not self.camera.initialize():
                    self.camera_enabled = False
                    self.camera_btn.configure(text="Start Camera")
                    self.camera_status.configure(text="📷  Camera: Failed to Initialize")
                    return

                self.camera_status.configure(text="📷  Camera: Active")
                self.web_app.emit_status({"camera_enabled": True})
            else:
                # Disabling camera
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
        self.audio_enabled = not self.audio_enabled

        try:
            if self.audio_enabled:
                self.audio_btn.configure(text="Audio Monitor")
                self.audio_status.configure(text="🎤  Audio: Active")
                if hasattr(self, "audio_processor"):
                    self.audio_processor.start()
                if hasattr(self, "emotion_recognizer"):
                    self.emotion_recognizer.start()
                self.web_app.emit_status({"audio_enabled": True})
            else:
                self.audio_btn.configure(text="Audio Monitor")
                self.audio_status.configure(text="🎤  Audio: Disabled")
                if hasattr(self, "audio_processor"):
                    self.audio_processor.stop()
                if hasattr(self, "emotion_recognizer"):
                    self.emotion_recognizer.stop()
                self.web_app.emit_status({"audio_enabled": False})
        except Exception as e:
            self.logger.error(f"Error toggling audio: {str(e)}")
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
        if self.is_running:
            return

        self.is_running = True

        # Start frame processing thread
        self.frame_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.frame_thread.start()

        # Start audio processing if enabled
        if self.audio_enabled:
            if hasattr(self, "audio_processor"):
                self.audio_processor.start()
            if hasattr(self, "emotion_recognizer"):
                self.emotion_recognizer.start()

        # Start processing the waveform queue
        self.process_waveform_queue()

        self.logger.info("Baby monitor system started")

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
            if hasattr(self, "emotion_recognizer"):
                self.emotion_recognizer.stop()
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
        frame_interval = 1.0 / 60.0  # Increased target FPS to 60
        error_count = 0
        max_errors = 3
        frame_count = 0
        last_ui_update = 0
        ui_update_interval = 0.1  # Update UI every 100ms

        while self.is_running:
            try:
                if not self.camera_enabled:
                    time.sleep(0.01)
                    continue

                if not hasattr(self, 'camera') or not self.camera.is_available():
                    if error_count == 0:  # Only log once
                        self.logger.warning("Camera not available")
                        self.root.after(0, lambda: self.camera_status.configure(text="📷  Camera: Error - Not Available"))
                        self.root.after(0, lambda: self.camera_btn.configure(text="Start Camera"))
                    error_count += 1
                    time.sleep(0.1)
                    continue

                # Frame rate control
                current_time = time.time()
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.0001)  # Minimal sleep
                    continue

                # Get frame
                ret, frame = self.camera.get_frame()
                if not ret or frame is None:
                    if error_count == 0:  # Only log once
                        self.logger.warning("Failed to get frame")
                    error_count += 1
                    if error_count >= max_errors:
                        self.root.after(0, lambda: self.camera_status.configure(text="📷  Camera: Error - No Frame"))
                        error_count = 0
                    time.sleep(0.01)
                    continue

                error_count = 0
                frame_count += 1
                last_frame_time = current_time

                # Process frame with minimal copying
                detection_results = {
                    'people_count': 0,
                    'rapid_motion': False,
                    'fall_detected': False,
                    'position': 'unknown',
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }

                try:
                    # Process detection every 3rd frame
                    if frame_count % 3 == 0 and hasattr(self, 'person_detector'):
                        detections = self.person_detector.detect(frame)
                        detection_results['people_count'] = int(len(detections))

                        if hasattr(self, 'motion_detector'):
                            processed_frame, rapid_motion, fall_detected = self.motion_detector.detect(frame, detections)
                            detection_results.update({
                                'rapid_motion': bool(rapid_motion),
                                'fall_detected': bool(fall_detected)
                            })

                            # Update UI status less frequently
                            if current_time - last_ui_update >= ui_update_interval:
                                status_text = "👀  Detection: "
                                if detection_results['fall_detected']:
                                    status_text += "Fall Detected!"
                                elif detection_results['rapid_motion']:
                                    status_text += "Rapid Motion"
                                elif detection_results['people_count'] > 0:
                                    status_text += f"{detection_results['people_count']} Person(s)"
                                else:
                                    status_text += "No Activity"
                                
                                self.root.after(0, lambda: self.detection_status.configure(text=status_text))
                                last_ui_update = current_time

                            # Emit detection results every 5th processed frame
                            if frame_count % 5 == 0 and hasattr(self, 'web_app'):
                                self.web_app.emit_detection(detection_results)
                    else:
                        processed_frame = frame

                    # Update web interface with reduced frequency
                    if frame_count % 2 == 0 and hasattr(self, 'web_app'):
                        self.web_app.emit_frame(frame)

                except Exception as e:
                    if self.dev_mode:
                        self.logger.error(f"Error in frame processing: {str(e)}")
                    processed_frame = frame

                # Update UI elements with optimized image processing
                try:
                    # Skip frame update if UI is too busy
                    if self.root.tk.dooneevent(0):  # Check if UI is responsive
                        # Convert and resize frame more efficiently
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Get current canvas dimensions
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

                            # Use more efficient resizing method
                            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), 
                                                interpolation=cv2.INTER_NEAREST)  # Faster interpolation

                        # Convert to PIL more efficiently
                        frame_pil = Image.fromarray(frame_rgb)
                        photo = ImageTk.PhotoImage(image=frame_pil)

                        # Update canvas in main thread
                        self.root.after(1, lambda p=photo, w=canvas_width, h=canvas_height: 
                                      self.update_video_frame(p, w, h))

                except Exception as e:
                    if self.dev_mode:
                        self.logger.error(f"Error updating video display: {str(e)}")

            except Exception as e:
                if self.dev_mode:
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
                if self.camera.set_resolution(selected_resolution):
                    self.camera_status.configure(text="📷  Camera: Ready")
                else:
                    self.camera_status.configure(text="📷  Camera: Resolution Error")
        except Exception as e:
            self.logger.error(f"Error setting resolution: {str(e)}")
            self.camera_status.configure(text="📷  Camera: Resolution Error")

    def update_video_frame(self, photo, canvas_width, canvas_height):
        """Update video frame in the main thread."""
        if not self.is_running:
            return
        self.video_canvas.delete("all")
        self.video_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=photo,
            anchor=tk.CENTER
        )
        self.video_canvas.image = photo  # Keep a reference


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
