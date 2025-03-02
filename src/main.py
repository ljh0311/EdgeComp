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
from utils.camera import Camera
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
        self.current_frame = None

        try:
            # Initialize web interface first
            self.web_app = BabyMonitorWeb(dev_mode=self.dev_mode)
            self.web_app.set_monitor_system(self)

            # Initialize camera
            self.camera = Camera(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
            if not self.camera.initialize():
                self.logger.error("Failed to initialize camera")
                self.camera_error = True
                self.camera_enabled = False
                self.web_app.emit_alert("error", "Failed to initialize camera")

            # Initialize person detector
            self.person_detector = PersonDetector(
                model_path=os.path.join(src_dir, "models", "yolov8n.pt")
            )

            # Initialize audio components if enabled
            if self.audio_enabled:
                self.audio_processor = AudioProcessor(
                    Config.AUDIO_PROCESSING, self.handle_alert
                )
                self.emotion_recognizer = EmotionRecognizer(
                    model_path=os.path.join(src_dir, "models", "emotion_model.pt"),
                    web_app=self.web_app,
                )

            # Start web interface
            self.web_app.start()

            # Send initial status
            self.web_app.emit_status(
                {
                    "camera_enabled": self.camera_enabled,
                    "audio_enabled": self.audio_enabled,
                    "emotion_enabled": hasattr(self, "emotion_recognizer"),
                    "detection_enabled": hasattr(self, "person_detector"),
                }
            )

            self.logger.info("Baby Monitor System initialized successfully")

        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            raise

    def start(self):
        """Start the monitoring system."""
        if self.is_running:
            return

        self.is_running = True

        # Start frame processing thread
        self.frame_thread = threading.Thread(target=self.process_frames)
        self.frame_thread.daemon = True
        self.frame_thread.start()

        # Start audio processing if enabled
        if self.audio_enabled and hasattr(self, "audio_processor"):
            self.audio_processor.start()
        if self.audio_enabled and hasattr(self, "emotion_recognizer"):
            self.emotion_recognizer.start()

        self.logger.info("Baby monitor system started")

    def stop(self):
        """Stop the monitoring system."""
        self.is_running = False

        # Stop audio components
        if hasattr(self, "audio_processor"):
            self.audio_processor.stop()
        if hasattr(self, "emotion_recognizer"):
            self.emotion_recognizer.stop()

        # Stop web interface
        if hasattr(self, "web_app"):
            self.web_app.stop()

        # Release camera
        if hasattr(self, "camera"):
            self.camera.release()

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
                    time.sleep(0.001)
                    continue

                last_frame_time = current_time

                ret, frame = self.camera.get_frame()
                if not ret or frame is None:
                    continue

                # Store original frame with synchronization
                with self.frame_lock:
                    self.current_frame = frame.copy()

                # Process frame with person detector
                if hasattr(self, "person_detector"):
                    processed_frame = self.person_detector.process_frame(frame)
                else:
                    processed_frame = frame

                # Update web interface
                if hasattr(self, "web_app"):
                    try:
                        self.web_app.emit_frame(processed_frame)
                    except Exception as e:
                        self.logger.error(
                            f"Error sending frame to web interface: {str(e)}"
                        )

            except Exception as e:
                self.logger.error(f"Error in frame processing loop: {str(e)}")
                time.sleep(0.1)

    def toggle_camera(self):
        """Toggle camera feed on/off."""
        try:
            if not self.camera_error:
                self.camera_enabled = not self.camera_enabled

                if self.camera_enabled:
                    # Try to initialize camera if needed
                    if not self.camera.is_initialized:
                        if not self.camera.initialize():
                            self.camera_error = True
                            self.camera_enabled = False
                            raise Exception("Failed to initialize camera")
                # Update web interface
                self.web_app.emit_status(
                    {
                        "camera_enabled": self.camera_enabled,
                        "audio_enabled": self.audio_enabled,
                    }
                )

        except Exception as e:
            self.logger.error(f"Error toggling camera: {str(e)}")
            self.camera_error = True
            self.camera_enabled = False
            # Update web interface with error state
            self.web_app.emit_status(
                {"camera_enabled": False, "audio_enabled": self.audio_enabled}
            )
            self.web_app.emit_alert("error", f"Camera error: {str(e)}")

    def toggle_audio(self):
        """Toggle audio monitoring on/off."""
        try:
            self.audio_enabled = not self.audio_enabled

            if self.audio_enabled:
                if hasattr(self, "audio_processor"):
                    self.audio_processor.start()
                if hasattr(self, "emotion_recognizer"):
                    self.emotion_recognizer.start()
            else:
                if hasattr(self, "audio_processor"):
                    self.audio_processor.stop()
                if hasattr(self, "emotion_recognizer"):
                    self.emotion_recognizer.stop()

            # Update web interface
            self.web_app.emit_status(
                {
                    "camera_enabled": self.camera_enabled,
                    "audio_enabled": self.audio_enabled,
                }
            )

        except Exception as e:
            self.logger.error(f"Error toggling audio: {str(e)}")
            self.web_app.emit_alert("error", f"Audio error: {str(e)}")

    def handle_alert(self, message, level="info"):
        """Handle system alerts."""
        if hasattr(self, "web_app"):
            self.web_app.emit_alert(level, message)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Baby Monitor System")
    parser.add_argument("--dev", action="store_true", help="Run in developer mode")
    args = parser.parse_args()

    try:
        app = BabyMonitorSystem(dev_mode=args.dev)
        app.start()

        # Keep the main thread running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            app.stop()

    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
