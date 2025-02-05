"""
Baby Monitor System
=================

This script implements a real-time baby monitoring system using computer vision and audio analysis.
Key features:
- Person detection and tracking
- Motion and fall detection
- Audio monitoring (cry detection - to be implemented)
- Alert system with history
- Modern UI with dark theme

Dependencies:
- OpenCV (cv2) for video processing
- PyAudio for audio processing
- Ultralytics YOLO for object detection
- Tkinter for UI
- NumPy for numerical operations

Configuration Options:
- camera_width/height: Set the camera resolution
- motion_threshold: Adjust sensitivity of motion detection
- fall_threshold: Adjust sensitivity of fall detection
- alert_duration: Set how long alerts stay visible
- max_alert_history: Set number of alerts to keep in history
"""

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
import time
import collections

# Platform detection for different camera implementations
IS_RASPBERRY_PI = platform.machine() in ('armv7l', 'aarch64')

# Configure logging - modify level and handlers as needed
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baby_monitor.log'),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

class BabyMonitor:
    """
    Main class for the Baby Monitor system.
    
    This class handles:
    1. Video capture and processing
    2. Person detection and tracking
    3. Motion and fall detection
    4. Audio monitoring
    5. Alert management
    6. UI rendering and updates
    
    Configuration is done through class attributes and can be modified
    during runtime if needed.
    """
    
    def __init__(self, camera_width=1280, camera_height=720):
        """
        Initialize the Baby Monitor system.
        
        Args:
            camera_width (int): Desired camera capture width (default: 1280)
            camera_height (int): Desired camera capture height (default: 720)
            
        Note: Actual camera resolution may differ based on hardware support
        """
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        
        # Video settings - modify these to change camera behavior
        self.camera = None
        self.frame = None
        self.frame_lock = threading.Lock()  # Prevents frame corruption during multi-threading
        
        # Audio settings - modify these to change audio behavior
        self.audio = pyaudio.PyAudio()
        self.audio_stream = None
        self.audio_frames = []
        
        # Detection settings - modify these to change detection behavior
        self.person_detector = None
        self.pose_detector = None
        self.cry_analyzer = None
        
        # Motion detection settings - adjust these to fine-tune detection sensitivity
        self.previous_positions = {}  # Tracks person positions between frames
        self.motion_threshold = 100   # Pixel distance for rapid movement (lower = more sensitive)
        self.fall_threshold = 1.5     # Aspect ratio change for fall detection (lower = more sensitive)
        self.previous_frame = None
        self.motion_history = []      # Stores recent motion data for analysis
        
        # Initialize YOLO models - modify model paths or parameters as needed
        # Model configurations - modify these to use different models
        self.model_configs = {
            'person': {
                'path': 'yolov8n.pt',      # Default small model
                'conf': 0.25,              # Confidence threshold
                'iou': 0.45,               # IOU threshold
                'device': 'cpu'            # Device to run on ('cpu', 'cuda', etc)
            },
            'pose': {
                'path': 'yolov8n-pose.pt', # Default pose model
                'conf': 0.25,              # Confidence threshold
                'iou': 0.45,               # IOU threshold
                'device': 'cpu'            # Device to run on
            }
        }

        try:
            # Initialize models using configurations
            self.person_detector = YOLO(
                self.model_configs['person']['path'],
                conf=self.model_configs['person']['conf'],
                iou=self.model_configs['person']['iou'],
                device=self.model_configs['person']['device']
            )
            
            self.pose_detector = YOLO(
                self.model_configs['pose']['path'],
                conf=self.model_configs['pose']['conf'],
                iou=self.model_configs['pose']['iou'],
                device=self.model_configs['pose']['device']
            )
            
            self.logger.info("YOLO models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO models: {str(e)}")
            raise

        # Store camera resolution for later use
        self.camera_width = camera_width
        self.camera_height = camera_height

        # Initialize UI components
        self.setup_ui()
        
        # UI state variables - can be modified to change UI behavior
        self.status_text = "Stopped"  # Will be updated based on camera status
        self.alert_text = ""
        self.person_count = 0
        
        # Camera status states
        self.CAMERA_DISCONNECTED = "Camera Disconnected"
        self.CAMERA_RUNNING = "Monitoring..."
        self.CAMERA_STOPPED = "Stopped"

        # Alert settings - modify these to change alert behavior
        self.alert_duration = 10000  # Alert duration in milliseconds
        self.current_alert_timer = None
        self.alert_history = []
        self.max_alert_history = 5  # Maximum number of alerts to show in history

    def setup_ui(self):
        """Setup the UI window and elements with improved accessibility and modern design"""
        self.root = tk.Tk()
        self.root.title("Baby Monitor System")
        self.root.configure(bg='#1a1a1a')  # Darker background
        
        # Set minimum window size and calculate initial size
        self.root.minsize(800, 600)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        initial_width = int(screen_width * 0.8)
        initial_height = int(screen_height * 0.8)
        self.root.geometry(f"{initial_width}x{initial_height}")
        
        # Create main container with padding
        self.main_container = ttk.Frame(self.root, style='Main.TFrame')
        self.main_container.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Configure grid weights for responsiveness
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Title frame
        self.title_frame = ttk.Frame(self.main_container, style='Title.TFrame')
        self.title_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.title_label = ttk.Label(self.title_frame, 
                                    text="Baby Monitor System", 
                                    font=('Arial', int(initial_height * 0.03), 'bold'),
                                    foreground='white',
                                    background='#1a1a1a')
        self.title_label.grid(row=0, column=0, padx=10)
        
        # Status indicators in title frame
        self.status_frame = ttk.Frame(self.title_frame, style='Status.TFrame')
        self.status_frame.grid(row=0, column=1, padx=10, sticky="e")
        
        self.status_label = ttk.Label(self.status_frame, 
                                    text="● MONITORING", 
                                    font=('Arial', int(initial_height * 0.02), 'bold'),
                                    foreground='#00ff00',  # Green
                                    background='#1a1a1a')
        self.status_label.grid(row=0, column=0, padx=10)
        
        self.person_label = ttk.Label(self.status_frame, 
                                    text="People in room: 0", 
                                    font=('Arial', int(initial_height * 0.02)),
                                    foreground='white',
                                    background='#1a1a1a')
        self.person_label.grid(row=0, column=1, padx=10)
        
        # Video frame with border, maintaining 16:9 aspect ratio
        self.video_frame = ttk.Frame(self.main_container, style='Video.TFrame')
        self.video_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")  # Allow expansion
        
        # Bind resize event to maintain aspect ratio
        self.video_frame.bind('<Configure>', self._maintain_aspect_ratio)
        
        # Alert container to hold both current alert and history
        self.alert_container = ttk.Frame(self.main_container, style='AlertContainer.TFrame')
        self.alert_container.grid(row=1, column=1, sticky="ns", padx=5)
        
        # Current alert frame
        self.alert_frame = ttk.Frame(self.alert_container, style='Alert.TFrame')
        self.alert_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.alert_frame.grid_remove()  # Hidden by default
        
        self.alert_label = ttk.Label(self.alert_frame, 
                                   text="", 
                                   font=('Arial', int(initial_height * 0.02), 'bold'),
                                   foreground='white',
                                   background='#ff3333')
        self.alert_label.grid(row=0, column=0, pady=10, padx=10, sticky="ew")
        
        # Add dismiss button to alert frame
        self.dismiss_button = ttk.Button(self.alert_frame, 
                                       text="Dismiss Alert", 
                                       command=self.dismiss_alert)
        self.dismiss_button.grid(row=1, column=0, pady=(0, 5))
        
        # Alert history section
        self.alert_history_container = ttk.Frame(self.alert_container, style='AlertHistory.TFrame')
        self.alert_history_container.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        
        # History title
        self.history_title = ttk.Label(self.alert_history_container,
                                     text="Alert History",
                                     font=('Arial', int(initial_height * 0.02), 'bold'),
                                     foreground='white',
                                     background='#1a1a1a')
        self.history_title.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 0))
        
        # Create canvas for scrolling
        self.alert_canvas = tk.Canvas(self.alert_history_container, 
                                    background='#1a1a1a', 
                                    highlightthickness=0)
        self.alert_canvas.grid(row=1, column=0, sticky="nsew", padx=(10, 0))
        
        # Add scrollbar
        self.alert_scrollbar = ttk.Scrollbar(self.alert_history_container, 
                                           orient=tk.VERTICAL, 
                                           command=self.alert_canvas.yview)
        self.alert_scrollbar.grid(row=1, column=1, sticky="ns")
        
        # Configure canvas
        self.alert_canvas.configure(yscrollcommand=self.alert_scrollbar.set)
        
        # Create frame inside canvas for alerts
        self.alert_history_frame = ttk.Frame(self.alert_canvas, style='AlertHistory.TFrame')
        self.alert_canvas.create_window((0, 0), 
                                      window=self.alert_history_frame, 
                                      anchor='nw', 
                                      width=self.alert_canvas.winfo_width())
        
        # Configure styles
        style = ttk.Style()
        style.configure('Main.TFrame', background='#1a1a1a')
        style.configure('Title.TFrame', background='#1a1a1a')
        style.configure('Status.TFrame', background='#1a1a1a')
        style.configure('Video.TFrame', background='#333333')
        style.configure('Alert.TFrame', background='#ff3333')
        style.configure('AlertHistory.TFrame', background='#1a1a1a')
        style.configure('AlertContainer.TFrame', background='#1a1a1a')
        
        # Configure window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        
        # Bind window resize event
        self.root.bind('<Configure>', self._on_window_resize)

    def _maintain_aspect_ratio(self, event):
        """Maintain a 16:9 aspect ratio for the video frame."""
        desired_aspect_ratio = 16 / 9
        current_width = event.width
        current_height = event.height
        if current_width / current_height > desired_aspect_ratio:
            new_width = int(current_height * desired_aspect_ratio)
            self.video_frame.config(width=new_width)
        else:
            new_height = int(current_width / desired_aspect_ratio)
            self.video_frame.config(height=new_height)

    def _on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.alert_canvas.configure(scrollregion=self.alert_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """When canvas is resized, resize the inner frame to match"""
        self.alert_canvas.itemconfig('window', width=event.width)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.alert_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_window_resize(self, event):
        """Handle window resize events"""
        # Update wraplength for all history labels
        for label in self.alert_history:
            label.configure(wraplength=self.alert_canvas.winfo_width() - 20)

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
            results = self.person_detector(frame, conf=0.5)
            
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
            
            # Add motion and fall detection
            frame = self.detect_motion_and_falls(frame, boxes)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error in person detection: {str(e)}")
            return frame

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
            try:
                ret, frame = self.get_frame()
                if ret and frame is not None:
                    # Create a copy of the frame for processing to avoid modifying the original
                    process_frame = frame.copy()
                    
                    # Use a separate thread for detection to avoid blocking the video feed
                    detection_thread = threading.Thread(
                        target=self._process_detection,
                        args=(process_frame,)
                    )
                    detection_thread.start()
                    
                    # Update the display frame immediately
                    with self.frame_lock:
                        self.frame = frame
                    
                    # Don't wait for detection to complete
                    detection_thread.join(timeout=0.1)
                    
                    # Add a small sleep to prevent CPU overload
                    time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Error in video processing: {str(e)}")
                time.sleep(0.1)  # Prevent rapid error loops

    def _process_detection(self, frame):
        """Process detection in a separate thread to avoid blocking the video feed."""
        try:
            # Process frame
            frame = self.detect_person(frame)
            
            # Store the processed frame for motion detection
            if self.previous_frame is None:
                self.previous_frame = frame.copy()
            else:
                # Detect motion and falls
                frame = self.detect_motion_and_falls(frame, self.person_detector(frame)[0].boxes)
                self.previous_frame = frame.copy()
        except Exception as e:
            self.logger.error(f"Error in detection processing: {str(e)}")

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
        # self.root.quit()
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

    def detect_motion_and_falls(self, frame, boxes):
        """
        Detect rapid movements and potential falls in the video feed.
        
        This function uses two main detection methods:
        1. Frame difference analysis for general motion detection
        2. Aspect ratio analysis for fall detection
        
        Adjustable parameters:
        - self.motion_threshold: Threshold for rapid movement detection
        - self.fall_threshold: Threshold for fall detection
        - Motion spike threshold (current_motion > avg_previous_motion * 3)
        
        Returns:
            frame: Annotated frame with detection results
        """
        current_positions = {}
        rapid_motion = False
        fall_detected = False
        
        try:
            # Motion detection using frame difference
            if self.previous_frame is not None:
                # Convert frames to grayscale for more efficient processing
                current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate absolute difference between frames
                frame_diff = cv2.absdiff(current_gray, prev_gray)
                motion_score = np.mean(frame_diff)  # Higher score = more motion
                
                # Maintain motion history for spike detection
                if not hasattr(self, 'motion_history'):
                    self.motion_history = collections.deque(maxlen=10)
                self.motion_history.append(motion_score)
                
                # Detect sudden motion spikes by comparing current motion to average
                if len(self.motion_history) > 2:
                    current_motion = self.motion_history[-1]
                    avg_previous_motion = np.mean(list(self.motion_history)[:-1])
                    # Adjust multiplier (3) to change sensitivity
                    if current_motion > avg_previous_motion * 3:
                        rapid_motion = True
                        cv2.putText(frame, "RAPID MOTION DETECTED!", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.logger.warning("Rapid motion detected")
            
            # Process each detected person for fall detection
            if boxes is not None:
                for i, box in enumerate(boxes):
                    if box.cls == 0:  # person class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Calculate position metrics
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        height = y2 - y1
                        width = x2 - x1
                        
                        current_positions[i] = (center_x, center_y, width, height)
                        
                        # Fall detection using aspect ratio changes
                        if i in self.previous_positions:
                            prev_x, prev_y, prev_width, prev_height = self.previous_positions[i]
                            
                            # Calculate and compare aspect ratios
                            current_ratio = width / height if height != 0 else 0
                            previous_ratio = prev_width / prev_height if prev_height != 0 else 0
                            
                            if previous_ratio != 0:
                                # Detect significant changes in aspect ratio
                                if (current_ratio > previous_ratio * self.fall_threshold or
                                    current_ratio < previous_ratio / self.fall_threshold):
                                    fall_detected = True
                                    cv2.putText(frame, "FALL DETECTED!", (10, 90),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    self.logger.warning(f"Potential fall detected for person {i+1}")
                            
                            # Calculate movement speed
                            movement = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                            if movement > self.motion_threshold:
                                rapid_motion = True
                                cv2.putText(frame, f"RAPID MOVEMENT - Person {i+1}!", (10, 120 + i*30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                self.logger.warning(f"Rapid movement detected for person {i+1}")
            
            # Update tracking data
            self.previous_positions = current_positions
            
            # Trigger appropriate alerts
            if fall_detected:
                self.show_alert("FALL DETECTED! Check on person immediately!", "critical")
            elif rapid_motion:
                self.show_alert("RAPID MOTION DETECTED! Monitor situation", "warning")
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error in motion and fall detection: {str(e)}")
            return frame

    def show_alert(self, message, level="warning"):
        """Show alert with specified message and level"""
        self.alert_text = message
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"{timestamp} - {message}"
        
        # Cancel any existing alert timer
        if self.current_alert_timer:
            self.root.after_cancel(self.current_alert_timer)
        
        # Configure alert appearance based on level
        if level == "critical":
            bg_color = '#ff3333'  # Red
            font_size = 18
            self.play_alert_sound("critical")
            self.flash_alert()
        else:
            bg_color = '#ff9900'  # Orange
            font_size = 16
            self.play_alert_sound("warning")
        
        # Configure and show current alert
        self.alert_label.configure(
            text=full_message,
            background=bg_color,
            font=('Arial', font_size, 'bold')
        )
        self.alert_frame.pack(fill=tk.X, pady=(0, 5))  # Ensure the alert frame is packed
        
        # Add to history
        self.add_to_alert_history(full_message, level)
        
        # Set timer for auto-hide
        self.current_alert_timer = self.root.after(self.alert_duration, self.hide_main_alert)

    def add_to_alert_history(self, message, level):
        """Add alert to history at the top"""
        # Create new label for alert history
        history_label = ttk.Label(
            self.alert_history_frame,
            text=message,
            font=('Arial', 12),
            foreground='white' if level == "warning" else '#ff9999',
            background='#1a1a1a',
            wraplength=self.alert_canvas.winfo_width() - 20  # Allow text wrapping
        )
        
        # Insert at the top
        history_label.pack(fill=tk.X, padx=10, pady=2, before=self.alert_history_frame.winfo_children()[0] if self.alert_history else None)
        
        # Add to history list
        self.alert_history.append(history_label)
        
        # Remove oldest alert if exceeding maximum
        if len(self.alert_history) > self.max_alert_history:
            oldest_alert = self.alert_history.pop(0)
            oldest_alert.destroy()
        
        # Update scroll region
        self._on_frame_configure()
        
        # Scroll to top
        self.alert_canvas.yview_moveto(0)

    def hide_main_alert(self):
        """Hide the main alert but keep history"""
        self.alert_frame.pack_forget()
        self.alert_text = ""
        self.current_alert_timer = None

    def dismiss_alert(self):
        """Manually dismiss the current alert"""
        self.hide_main_alert()
        if self.current_alert_timer:
            self.root.after_cancel(self.current_alert_timer)

    def clear_alert_history(self):
        """Clear all alerts from history"""
        for alert in self.alert_history:
            alert.destroy()
        self.alert_history.clear()
        self._on_frame_configure()

    def hide_alert(self):
        """Hide the alert frame"""
        self.alert_text = ""
        self.alert_frame.pack_forget()

    def flash_alert(self):
        """Flash the alert frame for critical alerts"""
        if not self.alert_text:
            return
        
        current_bg = self.alert_label.cget('background')
        new_bg = '#ffffff' if current_bg == '#ff3333' else '#ff3333'
        self.alert_label.configure(background=new_bg)
        
        # Schedule next flash
        self.root.after(500, self.flash_alert)

    def play_alert_sound(self, level="warning"):
        """Play alert sound based on severity level"""
        if level == "critical":
            frequency = 1000  # Hz
            duration = 500   # ms
        else:
            frequency = 500   # Hz
            duration = 200   # ms
        
        # Generate and play sound (Windows only)
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(frequency, duration)

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




