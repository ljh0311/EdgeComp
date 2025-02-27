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
matplotlib.use('TkAgg')
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

# Configure logging
logging.basicConfig(**Config.LOGGING)
logger = logging.getLogger(__name__)

class BabyMonitorSystem:
    def __init__(self, dev_mode=False):
        """Initialize the Baby Monitor System.
        
        Args:
            dev_mode (bool): If True, runs in developer mode with video feed and waveform.
                           If False, runs in production mode with only status and alerts.
        """
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.camera_error = False
        self.audio_enabled = True
        self.camera_enabled = True
        self.frame_lock = threading.Lock()
        self.dev_mode = dev_mode

        # Initialize UI first
        self.root = tk.Tk()
        self.setup_ui()
        
        try:
            # Initialize camera
            self.camera = Camera(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
            if not self.camera.initialize():
                self.logger.error("Failed to initialize camera")
                self.camera_error = True
                self.camera_status.configure(text="📷  Camera: Error")
            else:
                self.camera_status.configure(text="📷  Camera: Ready")

            # Initialize person detector
            self.person_detector = PersonDetector(
                model_path=os.path.join(src_dir, "models", "yolov8n.pt")
            )
            
            # Initialize web interface with dev_mode
            self.web_app = BabyMonitorWeb(dev_mode=self.dev_mode)
            self.web_app.set_monitor_system(self)
            self.web_app.start()

            # Initialize audio components if enabled
            if self.audio_enabled:
                self.audio_processor = AudioProcessor(
                    Config.AUDIO_PROCESSING,
                    self.handle_alert
                )
                self.audio_processor.set_visualization_callback(self.update_waveform_data)
                self.emotion_recognizer = EmotionRecognizer(
                    model_path=os.path.join(src_dir, "models", "emotion_model.pt"),
                    web_app=self.web_app
                )
                self.audio_status.configure(text="🎤  Audio: Ready")
                self.emotion_status.configure(text="😊  Emotion: Ready")

            self.logger.info("Baby Monitor System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            self.detection_status.configure(text="👀  Detection: Error")
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

        # Camera toggle button
        self.camera_btn = ttk.Button(
            controls_frame,
            text="Camera Feed",
            command=self.toggle_camera
        )
        self.camera_btn.pack(fill=tk.X, padx=5, pady=5)

        # Audio toggle button
        self.audio_btn = ttk.Button(
            controls_frame,
            text="Audio Monitor",
            command=self.toggle_audio
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

        self.emotion_status = ttk.Label(status_frame, text="😊  Emotion: Initializing...")
        self.emotion_status.pack(anchor=tk.W, padx=5, pady=2)

        self.detection_status = ttk.Label(status_frame, text="👀  Detection: Initializing...")
        self.detection_status.pack(anchor=tk.W, padx=5, pady=2)

        # Initialize matplotlib for waveform
        self.setup_waveform()

    def setup_waveform(self):
        """Setup waveform visualization."""
        self.waveform_figure = Figure(figsize=(6, 2), dpi=100)
        self.waveform_canvas = FigureCanvasTkAgg(self.waveform_figure, master=self.waveform_frame)
        self.waveform_canvas.draw()
        self.waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Setup the plot
        self.waveform_ax = self.waveform_figure.add_subplot(111)
        self.waveform_ax.set_ylim([-1, 1])
        self.waveform_ax.set_xlim([0, 1000])
        self.waveform_line, = self.waveform_ax.plot([], [], lw=2)
        self.waveform_data = np.zeros(1000)
        
        # Remove margins and grid
        self.waveform_figure.tight_layout(pad=0.1)
        self.waveform_ax.grid(False)
        
        # Bind resize event
        self.waveform_frame.bind("<Configure>", self._set_waveform_height)

    def update_waveform(self):
        """Update the waveform visualization."""
        if hasattr(self, 'waveform_line'):
            try:
                self.waveform_line.set_data(range(len(self.waveform_data)), self.waveform_data)
                self.waveform_canvas.draw_idle()
            except Exception as e:
                self.logger.error(f"Error updating waveform: {str(e)}")

    def update_waveform_data(self, audio_data):
        """Update the waveform data with new audio samples."""
        try:
            # Resize audio data to match waveform buffer size
            if len(audio_data) != len(self.waveform_data):
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), len(self.waveform_data)),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Roll existing data and append new data
            self.waveform_data = np.roll(self.waveform_data, -len(audio_data))
            self.waveform_data[-len(audio_data):] = audio_data
            
            # Update the plot
            self.update_waveform()
        except Exception as e:
            logging.error(f"Error updating waveform data: {str(e)}")

    def _set_waveform_height(self, event):
        """Adjust waveform figure size on window resize."""
        if hasattr(self, 'waveform_figure'):
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
                if hasattr(self, 'audio_processor'):
                    self.audio_processor.start()
                if hasattr(self, 'emotion_recognizer'):
                    self.emotion_recognizer.start()
                self.web_app.emit_status({"audio_enabled": True})
            else:
                self.audio_btn.configure(text="Audio Monitor")
                self.audio_status.configure(text="🎤  Audio: Disabled")
                if hasattr(self, 'audio_processor'):
                    self.audio_processor.stop()
                if hasattr(self, 'emotion_recognizer'):
                    self.emotion_recognizer.stop()
                self.web_app.emit_status({"audio_enabled": False})
        except Exception as e:
            self.logger.error(f"Error toggling audio: {str(e)}")
            self.audio_status.configure(text="🎤  Audio: Error")

    def handle_alert(self, message, level="info"):
        """Handle alerts from components."""
        if hasattr(self, 'web_app'):
            self.web_app.emit_alert(level, message)

    def send_status_update(self, status_data):
        """Send status update to web interface."""
        if hasattr(self, 'web_app'):
            self.web_app.emit_status(status_data)

    def start(self):
        """Start the monitoring system."""
        self.is_running = True
        
        # Start audio processing if enabled
        if self.audio_enabled:
            if hasattr(self, 'audio_processor'):
                self.audio_processor.start()
            if hasattr(self, 'emotion_recognizer'):
                self.emotion_recognizer.start()

        # Start frame processing thread
        self.process_thread = threading.Thread(target=self.process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()

        self.logger.info("Baby monitor system started")
        self.root.mainloop()

    def stop(self):
        """Stop the monitoring system."""
        self.is_running = False
        
        # Stop audio components
        if hasattr(self, 'audio_processor'):
            self.audio_processor.stop()
        if hasattr(self, 'emotion_recognizer'):
            self.emotion_recognizer.stop()
        
        # Stop web interface
        if hasattr(self, 'web_app'):
            self.web_app.stop()
        
        # Release camera
        if hasattr(self, 'camera'):
            self.camera.release()
        
        self.root.quit()
        self.logger.info("Baby monitor system stopped")

    def process_frames(self):
        """Process video frames in a separate thread."""
        last_frame_time = 0
        frame_interval = 1.0 / 30.0  # Target 30 FPS
        
        while self.is_running:
            try:
                if not self.camera_enabled:
                    time.sleep(0.1)
                    continue

                # Control frame rate
                current_time = time.time()
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent CPU overload
                    continue
                
                last_frame_time = current_time

                ret, frame = self.camera.get_frame()
                if not ret or frame is None:
                    continue

                # Store original frame with synchronization
                with self.frame_lock:
                    self.current_frame = frame.copy()

                # Process frame with person detector
                if hasattr(self, 'person_detector'):
                    processed_frame = self.person_detector.process_frame(frame)
                else:
                    processed_frame = frame

                # Update web interface
                if hasattr(self, 'web_app'):
                    try:
                        self.web_app.emit_frame(processed_frame)
                    except Exception as e:
                        self.logger.error(f"Error sending frame to web interface: {str(e)}")

                # Convert frame for display with double buffering
                try:
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Get current canvas dimensions
                    canvas_width = self.video_canvas.winfo_width()
                    canvas_height = self.video_canvas.winfo_height()
                    
                    if canvas_width > 1 and canvas_height > 1:
                        # Calculate aspect ratios
                        frame_aspect = frame_pil.width / frame_pil.height
                        canvas_aspect = canvas_width / canvas_height
                        
                        # Calculate new dimensions maintaining aspect ratio
                        if frame_aspect > canvas_aspect:
                            new_width = canvas_width
                            new_height = int(canvas_width / frame_aspect)
                        else:
                            new_height = canvas_height
                            new_width = int(canvas_height * frame_aspect)
                        
                        # Resize frame using LANCZOS resampling
                        frame_pil = frame_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Create new PhotoImage in the main thread
                    def update_canvas(photo):
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
                    
                    photo = ImageTk.PhotoImage(image=frame_pil)
                    self.root.after(1, update_canvas, photo)
                
                except Exception as e:
                    self.logger.error(f"Error updating video display: {str(e)}")

            except Exception as e:
                self.logger.error(f"Error in frame processing loop: {str(e)}")
                time.sleep(0.1)

    def process_audio(self):
        """Process audio input and update waveform."""
        try:
            while self.is_running and self.audio_enabled:
                # Read audio data
                audio_data = self.audio_stream.read(self.CHUNK_SIZE)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Normalize audio data
                normalized_audio = audio_array.astype(np.float32) / 32768.0
                
                # Update waveform visualization
                self.update_waveform_data(normalized_audio)
                
                # Process audio for emotion detection
                if self.emotion_recognizer:
                    # Move input tensor to same device as model
                    audio_tensor = torch.FloatTensor(normalized_audio).to(self.emotion_recognizer.device)
                    self.emotion_recognizer.process_audio(audio_tensor)
                
                time.sleep(0.01)  # Small delay to prevent high CPU usage
        except Exception as e:
            logging.error(f"Error processing audio: {str(e)}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Baby Monitor System')
    parser.add_argument('--dev', action='store_true', help='Run in developer mode')
    args = parser.parse_args()

    try:
        app = BabyMonitorSystem(dev_mode=args.dev)
        app.start()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 