# #!/usr/bin/env python3

import cv2
import numpy as np
import pyaudio
import threading
import logging
import platform
from datetime import datetime
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
    def __init__(self):
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
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
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
        Detect people and their pose in the frame using YOLO.
        Returns frame with bounding boxes, pose keypoints, and detected objects.
        """
        try:
            # Run YOLO detection
            results = self.person_detector(frame, conf=0.5)  # person detection
            pose_results = self.pose_detector(frame, conf=0.5)  # pose detection
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Filter for person class (class 0 in COCO dataset)
                    if box.cls == 0:  # person class
                        x1, y1, x2, y2 = box.xyxy[0]  # get box coordinates
                        conf = box.conf[0]  # confidence score
                        
                        # Convert coordinates to integers
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # Calculate box dimensions
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Determine if person is standing, sitting, or lying down
                        if height > width:
                            position = "standing up"
                        elif abs(height - width) <= min(height, width) * 0.2:
                            position = "sitting down"
                        else:
                            position = "lying down"

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add confidence label with position
                        label = f'Person ({position}): {conf:.2f}'
                        cv2.putText(frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Process pose detection results
            for pose in pose_results:
                if pose.keypoints is not None:
                    keypoints = pose.keypoints.data[0]
                    
                    # Draw hands (wrists and elbows)
                    hand_indices = [9, 10]  # wrists
                    elbow_indices = [7, 8]  # elbows
                    
                    # Draw legs (ankles and knees)
                    ankle_indices = [15, 16]  # ankles
                    knee_indices = [13, 14]  # knees
                    
                    # Colors for different body parts
                    HAND_COLOR = (255, 0, 0)  # Blue
                    LEG_COLOR = (0, 0, 255)   # Red
                    
                    # Draw hands
                    for wrist, elbow in zip(hand_indices, elbow_indices):
                        if keypoints[wrist][2] > 0.5 and keypoints[elbow][2] > 0.5:
                            wrist_point = tuple(map(int, keypoints[wrist][:2]))
                            elbow_point = tuple(map(int, keypoints[elbow][:2]))
                            cv2.line(frame, wrist_point, elbow_point, HAND_COLOR, 2)
                            cv2.circle(frame, wrist_point, 4, HAND_COLOR, -1)
                            cv2.putText(frame, "Hand", wrist_point, 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, HAND_COLOR, 2)
                    
                    # Draw legs
                    for ankle, knee in zip(ankle_indices, knee_indices):
                        if keypoints[ankle][2] > 0.5 and keypoints[knee][2] > 0.5:
                            ankle_point = tuple(map(int, keypoints[ankle][:2]))
                            knee_point = tuple(map(int, keypoints[knee][:2]))
                            cv2.line(frame, ankle_point, knee_point, LEG_COLOR, 2)
                            cv2.circle(frame, ankle_point, 4, LEG_COLOR, -1)
                            cv2.putText(frame, "Leg", ankle_point, 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, LEG_COLOR, 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error in person/pose detection: {str(e)}")
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

    def start(self):
        """Start the baby monitor system."""
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
            
            # Main loop for displaying results
            while self.is_running:
                with self.frame_lock:
                    if self.frame is not None:
                        cv2.imshow('Baby Monitor', self.frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            self.stop()
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
            self.stop()

    def stop(self):
        """Stop the baby monitor system and clean up resources."""
        self.is_running = False
        
        if self.camera is not None:
            if IS_RASPBERRY_PI:
                self.camera.stop()
            else:
                self.camera.release()
        
        if self.audio_stream is not None:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        self.audio.terminate()
        cv2.destroyAllWindows()
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

