# #!/usr/bin/env python3

import cv2
import numpy as np
import pyaudio
import threading
import logging
import platform
from datetime import datetime
import io
import tkinter as tk
from tkinter import ttk
import PIL.Image, PIL.ImageTk
from pathlib import Path
from ultralytics import YOLO
import torch

# Determine the platform
IS_RASPBERRY_PI = platform.machine() in ('armv7l', 'aarch64')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baby_monitor.log'),
        logging.StreamHandler()
    ]
)

class BabyMonitor:
    def __init__(self, camera_width=1280, camera_height=720):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        
        # Video settings
        self.camera = None
        self.frame = None
        self.frame_lock = threading.Lock()
        
        # Audio settings
        self.audio = pyaudio.PyAudio()
        self.audio_stream = None
        self.audio_frames = []
        
        # Detection settings
        self.person_detector = None
        self.pose_detector = None
        self.cry_analyzer = None
        
        # Initialize YOLO models
        try:
            self.person_detector = YOLO('yolov8n.pt')  # for person detection
            self.pose_detector = YOLO('yolov8n-pose.pt')  # for pose detection
            self.logger.info("YOLO models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO models: {str(e)}")
            raise
        
        self.logger.info(f"Baby Monitor system initialized on {'Raspberry Pi' if IS_RASPBERRY_PI else 'Windows'}")

        self.camera_width = camera_width
        self.camera_height = camera_height

        # Initialize UI
        self.setup_ui()
        
        # Add UI elements
        self.status_text = "Monitoring..."
        self.alert_text = ""
        self.person_count = 0

    def setup_ui(self):
        """Setup the UI window and elements"""
        self.root = tk.Tk()
        self.root.title("Baby Monitor System")
        self.root.configure(bg='#2C3E50')
        
        # Set minimum window size and make it full screen by default
        self.root.minsize(1280, 720)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Video frame
        self.video_label = ttk.Label(self.main_container)
        self.video_label.pack(pady=5, fill=tk.BOTH, expand=True)
        
        # Status frame
        self.status_frame = ttk.Frame(self.main_container)
        self.status_frame.pack(fill=tk.X, pady=5)
        
        # Status indicators with larger font
        self.status_label = ttk.Label(self.status_frame, 
                                    text="Status: Monitoring", 
                                    font=('Arial', 14))
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.person_label = ttk.Label(self.status_frame, 
                                    text="People in room: 0", 
                                    font=('Arial', 14))
        self.person_label.pack(side=tk.RIGHT, padx=5)
        
        # Alert frame at the bottom
        self.alert_frame = ttk.Frame(self.main_container, style='Alert.TFrame')
        self.alert_frame.pack(fill=tk.X, pady=5)
        
        self.alert_label = ttk.Label(self.alert_frame, 
                                   text="", 
                                   font=('Arial', 12, 'bold'), 
                                   foreground='red')
        self.alert_label.pack(pady=5)
        
        # Style configuration
        style = ttk.Style()
        style.configure('Alert.TFrame', background='#FFE5E5')
        
        # Configure window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    def initialize_camera(self):
        """Initialize the camera with appropriate settings."""
        try:
            if IS_RASPBERRY_PI:
                # Raspberry Pi camera setup
                from picamera2 import Picamera2
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (1280, 720)},
                    lores={"size": (640, 480)},
                    display="lores"
                )
                self.camera.configure(config)
                self.camera.start()
            else:
                # Windows/laptop camera setup
                self.camera = cv2.VideoCapture(0)
                
                # Set default resolution (16:9 aspect ratio)
                default_width = 1280
                default_height = 720
                
                # Set initial size but allow resizing
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, default_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, default_height)
                
                # Try to enable camera properties dialog (may not work on all cameras)
                try:
                    self.camera.set(cv2.CAP_PROP_SETTINGS, 1)
                except:
                    self.logger.warning("Camera settings dialog not available")
            
            self.logger.info("Camera initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {str(e)}")
            raise

    def get_frame(self):
        """Get frame from camera based on platform."""
        if IS_RASPBERRY_PI:
            return True, self.camera.capture_array()
        else:
            return self.camera.read()

    def initialize_audio(self):
        """Initialize the audio stream with appropriate settings."""
        try:
            self.audio_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024
            )
            self.logger.info("Audio initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize audio: {str(e)}")
            raise

    def detect_person(self, frame):
        """
        Detect people in the frame using YOLO.
        Returns frame with bounding boxes, position information and presence information.
        """
        try:
            # Run YOLO detection
            results = self.person_detector(frame, conf=0.5)  # person detection
            
            # Process results
            person_count = 0
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Filter for person class (class 0 in COCO dataset)
                    if box.cls == 0:  # person class
                        person_count += 1
                        x1, y1, x2, y2 = box.xyxy[0]  # get box coordinates
                        conf = box.conf[0]  # confidence score
                        
                        # Convert coordinates to integers
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # Calculate box dimensions
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Determine if person is standing, sitting, or lying down
                        if height > width * 1.5:  # Person is significantly taller than wide
                            position = "standing"
                        elif width > height * 1.5:  # Person is significantly wider than tall
                            position = "lying down"
                        else:  # Dimensions are relatively similar
                            position = "sitting"
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add confidence label with position
                        label = f'Person {person_count} ({position}): {conf:.2f}'
                        cv2.putText(frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Log person detection with position
                        self.logger.info(f"Person {person_count} detected in {position} position with confidence {conf:.2f}")

            # Display total person count in top-left corner
            cv2.putText(frame, f'People in room: {person_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Update the person count in the UI
            self.person_count = person_count
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error in person detection: {str(e)}")
            return frame  # Return original frame if detection fails

    def detect_baby_position(self, frame):
        """
        Detect and track baby's position in the frame.
        Returns frame with baby position marked.
        """
        # TODO: Implement baby detection and tracking
        return frame

    def analyze_cry(self, audio_data):
        """
        Analyze audio data to detect and classify baby cries.
        Returns cry type if detected (e.g., hunger, discomfort, tiredness).
        """
        # TODO: Implement cry detection and classification
        return None

    def process_video_stream(self):
        """Process video stream in a separate thread."""
        while self.is_running:
            ret, frame = self.get_frame()
            if ret:
                # Process frame
                frame = self.detect_person(frame)
                frame = self.detect_baby_position(frame)
                
                with self.frame_lock:
                    self.frame = frame

    def process_audio_stream(self):
        """Process audio stream in a separate thread."""
        while self.is_running:
            audio_data = np.frombuffer(
                self.audio_stream.read(1024, exception_on_overflow=False),
                dtype=np.float32
            )
            cry_type = self.analyze_cry(audio_data)
            if cry_type:
                self.logger.info(f"Detected cry type: {cry_type}")

    def update_display(self, frame):
        """Update the video display"""
        if frame is not None:
            # Resize frame to fit the window
            frame = cv2.resize(frame, (800, 600))
            
            # Convert frame to PhotoImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = PIL.Image.fromarray(frame)
            frame = PIL.ImageTk.PhotoImage(image=frame)
            
            # Update video label
            self.video_label.configure(image=frame)
            self.video_label.image = frame
            
            # Update status with current person count
            self.status_label.configure(text=f"Status: {self.status_text}")
            self.person_label.configure(text=f"People in room: {self.person_count}")
            
            # Update alert if any
            if self.alert_text:
                self.alert_label.configure(text=self.alert_text)
                self.alert_frame.pack(fill=tk.X, pady=5)
            else:
                self.alert_frame.pack_forget()

    def start(self):
        """Start the baby monitor system"""
        try:
            self.initialize_camera()
            self.initialize_audio()
            self.is_running = True
            
            # Start processing threads
            video_thread = threading.Thread(target=self.process_video_stream)
            audio_thread = threading.Thread(target=self.process_audio_stream)
            
            video_thread.start()
            audio_thread.start()
            
            self.logger.info("Baby Monitor system started")
            
            # Start UI update loop
            self.update_ui()
            self.root.mainloop()
            
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
            self.stop()

    def update_ui(self):
        """Update UI elements"""
        if self.is_running:
            with self.frame_lock:
                if self.frame is not None:
                    self.update_display(self.frame)
            
            # Schedule next update
            self.root.after(30, self.update_ui)

    def stop(self):
        """Stop the system and close UI"""
        self.is_running = False
        self.root.quit()
        self.root.destroy()
        
        if self.camera is not None:
            if IS_RASPBERRY_PI:
                self.camera.stop()
            else:
                self.camera.release()
        
        if self.audio_stream is not None:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        self.audio.terminate()
        self.logger.info("Baby Monitor system stopped")

def main():
    # Check for required packages
    if IS_RASPBERRY_PI:
        try:
            from picamera2 import Picamera2
        except ImportError:
            print("Please install picamera2 for Raspberry Pi: sudo apt install -y python3-picamera2")
            return
    
    monitor = BabyMonitor()
    monitor.start()

if __name__ == "__main__":
    main()

