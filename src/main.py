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

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
from PIL import Image, ImageTk
import argparse

# Add src directory to path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Local imports
from camera.camera import Camera
from detectors.person_detector import PersonDetector
from audio.audio_processor import AudioProcessor
from emotion.emotion_recognizer import EmotionRecognizer
from web.web_app import BabyMonitorWeb
from config import Config
from detectors.motion_detector import MotionDetector

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
        self.camera_enabled = True
        self.frame_lock = threading.Lock()
        self.dev_mode = dev_mode
        self.only_local = only_local
        self.only_web = only_web
        self.initialized_components = []  # Track initialized components for cleanup

        # Initialize UI first if not web-only
        if not only_web:
            try:
                self.root = tk.Tk()
                self.setup_ui()
                self.initialized_components.append('ui')
            except Exception as e:
                self.logger.error(f"Failed to initialize UI: {str(e)}")
                self.cleanup()
                raise

        try:
            # Initialize camera
            self.camera = Camera(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
            if not self.camera.initialize():
                self.logger.error("Failed to initialize camera")
                self.camera_error = True
                if not only_web:
                    self.camera_status.configure(text="📷  Camera: Error")
            else:
                self.initialized_components.append('camera')
                if not only_web:
                    self.camera_status.configure(text="📷  Camera: Ready")
                    self.update_camera_list()
                    self.update_resolution_list()

            # Initialize person detector
            try:
                self.person_detector = PersonDetector(
                    model_path=os.path.join(src_dir, "models", "yolov8n.pt")
                )
                self.initialized_components.append('person_detector')
            except Exception as e:
                self.logger.error(f"Failed to initialize person detector: {str(e)}")
                # Continue without person detection

            # Initialize motion detector
            try:
                self.motion_detector = MotionDetector(Config.MOTION_DETECTION)
                self.initialized_components.append('motion_detector')
            except Exception as e:
                self.logger.error(f"Failed to initialize motion detector: {str(e)}")
                # Continue without motion detection

            # Initialize web interface if not local-only
            if not only_local:
                try:
                    self.web_app = BabyMonitorWeb(dev_mode=self.dev_mode)
                    self.web_app.set_monitor_system(self)
                    self.web_app.start()
                    self.initialized_components.append('web_app')
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
                            self.update_waveform_data
                        )
                    self.initialized_components.append('audio_processor')

                    self.emotion_recognizer = EmotionRecognizer(
                        model_path=os.path.join(src_dir, "models", "emotion_model.pt"),
                        web_app=self.web_app if not only_local else None,
                    )
                    self.initialized_components.append('emotion_recognizer')

                    if not only_web:
                        self.audio_status.configure(text="🎤  Audio: Ready")
                        self.emotion_status.configure(text="😊  Emotion: Ready")
                except Exception as e:
                    self.logger.error(f"Failed to initialize audio components: {str(e)}")
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
        style.configure("Dark.TLabelframe.Label", background=bg_dark, foreground=text_color)
        style.configure("Status.TLabel", background=status_bg, foreground=text_color, padding=5)
        
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

        # Waveform frame
        self.waveform_frame = ttk.Frame(left_panel, style="Darker.TFrame", height=200)
        self.waveform_frame.pack(fill=tk.X, pady=(0, 10))
        self.waveform_frame.pack_propagate(False)

        # Right panel (controls and status)
        right_panel = ttk.Frame(main_container, style="Dark.TFrame", width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        right_panel.pack_propagate(False)

        # Controls section
        controls_frame = ttk.LabelFrame(right_panel, text="Controls", style="Dark.TLabelframe")
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        # Camera selection frame
        camera_selection_frame = ttk.Frame(controls_frame, style="Dark.TFrame")
        camera_selection_frame.pack(fill=tk.X, padx=5, pady=5)

        # Camera selection combobox
        ttk.Label(camera_selection_frame, text="Camera:", style="Status.TLabel").pack(side=tk.LEFT, padx=(0, 5))
        self.camera_select = ttk.Combobox(camera_selection_frame, state="readonly", width=15)
        self.camera_select.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.camera_select.bind("<<ComboboxSelected>>", self.on_camera_selected)

        # Resolution selection frame
        resolution_frame = ttk.Frame(controls_frame, style="Dark.TFrame")
        resolution_frame.pack(fill=tk.X, padx=5, pady=5)

        # Resolution selection combobox
        ttk.Label(resolution_frame, text="Resolution:", style="Status.TLabel").pack(side=tk.LEFT, padx=(0, 5))
        self.resolution_select = ttk.Combobox(resolution_frame, state="readonly", width=15)
        self.resolution_select.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.resolution_select.bind("<<ComboboxSelected>>", self.on_resolution_selected)

        # Camera toggle button
        self.camera_btn = ttk.Button(controls_frame, text="Camera Feed", command=self.toggle_camera)
        self.camera_btn.pack(fill=tk.X, padx=5, pady=5)

        # Audio toggle button
        self.audio_btn = ttk.Button(controls_frame, text="Audio Monitor", command=self.toggle_audio)
        self.audio_btn.pack(fill=tk.X, padx=5, pady=5)

        # Status section
        status_frame = ttk.LabelFrame(right_panel, text="Status", style="Dark.TLabelframe")
        status_frame.pack(fill=tk.X, pady=(0, 10))

        # Status labels with dark background
        self.camera_status = ttk.Label(status_frame, text="📷  Camera: Initializing...", style="Status.TLabel")
        self.camera_status.pack(fill=tk.X, pady=1)

        self.audio_status = ttk.Label(status_frame, text="🎤  Audio: Initializing...", style="Status.TLabel")
        self.audio_status.pack(fill=tk.X, pady=1)

        self.emotion_status = ttk.Label(status_frame, text="😊  Emotion: Initializing...", style="Status.TLabel")
        self.emotion_status.pack(fill=tk.X, pady=1)

        self.detection_status = ttk.Label(status_frame, text="👀  Detection: Initializing...", style="Status.TLabel")
        self.detection_status.pack(fill=tk.X, pady=1)

        # Initialize matplotlib for waveform with dark theme
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
        self.camera_enabled = not self.camera_enabled

        try:
            if self.camera_enabled:
                self.camera_btn.configure(text="Camera Feed")
                self.camera_status.configure(text="📷  Camera: Active")
                self.web_app.emit_status({"camera_enabled": True})
            else:
                self.camera_btn.configure(text="Camera Feed")
                self.camera_status.configure(text="📷  Camera: Disabled")
                # Clear the video canvas
                self.video_canvas.delete("all")
                self.web_app.emit_status({"camera_enabled": False})
        except Exception as e:
            self.logger.error(f"Error toggling camera: {str(e)}")
            self.camera_status.configure(text="📷  Camera: Error")

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
        frame_interval = 1.0 / 30.0  # Target 30 FPS
        max_frame_time = 1.0 / 15.0  # Maximum acceptable frame time before skipping

        while self.is_running:
            try:
                if not self.camera_enabled:
                    time.sleep(0.1)
                    continue

                current_time = time.time()
                time_since_last_frame = current_time - last_frame_time

                # Skip frame if we're falling too far behind
                if time_since_last_frame < frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent CPU overload
                    continue
                elif time_since_last_frame > max_frame_time:
                    # We're falling behind, skip this frame
                    ret, _ = self.camera.get_frame()  # Just read and discard frame
                    last_frame_time = current_time
                    continue

                ret, frame = self.camera.get_frame()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Store original frame with synchronization
                with self.frame_lock:
                    self.current_frame = frame.copy()

                # Process frame with person detector
                if hasattr(self, "person_detector"):
                    detections = self.person_detector.detect(frame)
                    
                    # Process frame with motion detector
                    if hasattr(self, "motion_detector"):
                        processed_frame, rapid_motion, fall_detected = self.motion_detector.detect(frame, detections)
                        
                        # Update web interface with detection results
                        if hasattr(self, "web_app"):
                            self.web_app.emit_detection({
                                'people_count': len(detections),
                                'rapid_motion': rapid_motion,
                                'fall_detected': fall_detected
                            })
                    else:
                        processed_frame = frame
                else:
                    processed_frame = frame

                # Update web interface with reduced frequency
                if hasattr(self, "web_app") and time_since_last_frame >= frame_interval * 2:
                    try:
                        self.web_app.emit_frame(processed_frame)
                    except Exception as e:
                        self.logger.error(f"Error sending frame to web interface: {str(e)}")

                # Only update local GUI if not in web-only mode
                if not self.only_web:
                    try:
                        # Get current canvas dimensions first to avoid unnecessary processing
                        canvas_width = self.video_canvas.winfo_width()
                        canvas_height = self.video_canvas.winfo_height()

                        if canvas_width <= 1 or canvas_height <= 1:
                            continue

                        # Calculate target dimensions once
                        frame_aspect = processed_frame.shape[1] / processed_frame.shape[0]
                        canvas_aspect = canvas_width / canvas_height

                        if frame_aspect > canvas_aspect:
                            new_width = canvas_width
                            new_height = int(canvas_width / frame_aspect)
                        else:
                            new_height = canvas_height
                            new_width = int(canvas_height * frame_aspect)

                        # Resize BGR frame directly using cv2 for better performance
                        resized_frame = cv2.resize(processed_frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                        
                        # Convert to RGB and PIL Image
                        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)

                        # Create new PhotoImage in the main thread
                        def update_canvas(photo):
                            if not self.is_running:
                                return
                            self.video_canvas.delete("all")
                            self.video_canvas.create_image(
                                canvas_width // 2,
                                canvas_height // 2,
                                image=photo,
                                anchor=tk.CENTER,
                            )
                            self.video_canvas.image = photo  # Keep a reference

                        photo = ImageTk.PhotoImage(image=frame_pil)
                        self.root.after(1, update_canvas, photo)

                    except Exception as e:
                        self.logger.error(f"Error updating video display: {str(e)}")

                last_frame_time = current_time

            except Exception as e:
                self.logger.error(f"Error in frame processing loop: {str(e)}")
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
        """Update the waveform data with new audio samples."""
        # Skip waveform updates in web-only mode
        if self.only_web:
            return

        try:
            # Resize audio data to match waveform buffer size
            if len(audio_data) != len(self.waveform_data):
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), len(self.waveform_data)),
                    np.arange(len(audio_data)),
                    audio_data,
                )

            # Roll existing data and append new data
            self.waveform_data = np.roll(self.waveform_data, -len(audio_data))
            self.waveform_data[-len(audio_data) :] = audio_data

            # Update the plot
            self.update_waveform()
        except Exception as e:
            self.logger.error(f"Error updating waveform data: {str(e)}")

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
            camera_list = self.camera.get_camera_list()
            if camera_list:
                self.camera_select['values'] = camera_list
                current_camera = camera_list[self.camera.selected_camera_index]
                self.camera_select.set(current_camera)
                self.update_resolution_list(current_camera)
        except Exception as e:
            self.logger.error(f"Error updating camera list: {str(e)}")

    def update_resolution_list(self, camera_name=None):
        """Update the resolution list in the combobox."""
        try:
            if camera_name is None:
                camera_name = self.camera_select.get()
            resolutions = self.camera.get_camera_resolutions(camera_name)
            if resolutions:
                self.resolution_select['values'] = resolutions
                current_resolution = self.camera.get_current_resolution()
                self.resolution_select.set(current_resolution)
        except Exception as e:
            self.logger.error(f"Error updating resolution list: {str(e)}")

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

    def cleanup(self):
        """Clean up initialized components in reverse order."""
        for component in reversed(self.initialized_components):
            try:
                if component == 'web_app':
                    if hasattr(self, 'web_app'):
                        self.web_app.stop()
                elif component == 'camera':
                    if hasattr(self, 'camera'):
                        self.camera.release()
                elif component == 'audio_processor':
                    if hasattr(self, 'audio_processor'):
                        self.audio_processor.stop()
                elif component == 'emotion_recognizer':
                    if hasattr(self, 'emotion_recognizer'):
                        self.emotion_recognizer.stop()
                elif component == 'ui':
                    if hasattr(self, 'root'):
                        self.root.quit()
            except Exception as e:
                self.logger.error(f"Error cleaning up {component}: {str(e)}")
        self.initialized_components.clear()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Baby Monitor System")
    parser.add_argument("--dev", action="store_true", help="Run in developer mode")
    parser.add_argument("--onlyLocal", action="store_true", help="Run only the local GUI interface")
    parser.add_argument("--onlyWeb", action="store_true", help="Run only the web interface")
    args = parser.parse_args()

    if args.onlyLocal and args.onlyWeb:
        print("Error: Cannot specify both --onlyLocal and --onlyWeb")
        sys.exit(1)

    try:
        app = BabyMonitorSystem(
            dev_mode=args.dev,
            only_local=args.onlyLocal,
            only_web=args.onlyWeb
        )
        app.start()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
