"""
Baby Monitor System - Main Module
===============================
Main entry point for the Baby Monitor System.
Coordinates all components and handles the main application loop.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import logging
import threading
import time
import PIL.Image, PIL.ImageTk

from utils.config import Config
from utils.camera import Camera
from detectors.person_detector import PersonDetector
from detectors.motion_detector import MotionDetector
from ui.alert_manager import AlertManager

class BabyMonitorApp:
    def __init__(self):
        """Initialize the Baby Monitor application."""
        # Setup logging
        logging.basicConfig(**Config.LOGGING)
        self.logger = logging.getLogger(__name__)
        
        # Initialize state
        self.is_running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        
        # Initialize UI
        self.root = tk.Tk()
        self.setup_ui()
        
        # Initialize components
        self.camera = Camera(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
        self.person_detector = PersonDetector(Config.PERSON_DETECTION)
        self.motion_detector = MotionDetector(Config.MOTION_DETECTION)
        self.alert_manager = AlertManager(self.root, Config.ALERT)
        
        self.logger.info("Baby Monitor application initialized")
    
    def setup_ui(self):
        """Setup the main UI window."""
        self.root.title("Baby Monitor System")
        self.root.configure(bg=Config.UI['dark_theme']['background'])
        
        # Set minimum window size
        self.root.minsize(Config.UI['min_width'], Config.UI['min_height'])
        
        # Calculate initial size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        initial_width = int(screen_width * Config.UI['window_scale'])
        initial_height = int(screen_height * Config.UI['window_scale'])
        self.root.geometry(f"{initial_width}x{initial_height}")
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create video display frame
        self.video_frame = ttk.Frame(self.main_container)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create video label for displaying the camera feed
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Configure window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
    
    def start(self):
        """Start the monitoring system."""
        try:
            # Initialize camera
            if not self.camera.initialize():
                self.logger.error("Failed to initialize camera")
                return
            
            self.is_running = True
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self.process_frames)
            self.process_thread.start()
            
            # Start UI update loop
            self.update_ui()
            
            # Start main loop
            self.root.mainloop()
            
        except Exception as e:
            self.logger.error(f"Error starting application: {str(e)}")
            self.stop()
    
    def process_frames(self):
        """Process video frames in a separate thread."""
        while self.is_running:
            try:
                # Get frame from camera
                ret, frame = self.camera.get_frame()
                if not ret or frame is None:
                    continue
                
                # Create a copy for processing
                process_frame = frame.copy()
                
                # Detect people
                frame, person_count, boxes = self.person_detector.detect(process_frame)
                
                # Detect motion and falls
                frame, rapid_motion, fall_detected = self.motion_detector.detect(frame, boxes)
                
                # Update frame
                with self.frame_lock:
                    self.frame = frame
                
                # Handle alerts
                if fall_detected:
                    self.alert_manager.show_alert(
                        "FALL DETECTED! Check on person immediately!",
                        "critical"
                    )
                elif rapid_motion:
                    self.alert_manager.show_alert(
                        "RAPID MOTION DETECTED! Monitor situation",
                        "warning"
                    )
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error processing frame: {str(e)}")
                time.sleep(0.1)  # Prevent rapid error loops
    
    def update_ui(self):
        """Update the UI with the latest frame."""
        if self.is_running:
            with self.frame_lock:
                if self.frame is not None:
                    # Convert frame to PhotoImage
                    frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    frame = PIL.Image.fromarray(frame)
                    
                    # Resize frame to fit the window while maintaining aspect ratio
                    display_width = self.video_frame.winfo_width()
                    display_height = self.video_frame.winfo_height()
                    
                    if display_width > 0 and display_height > 0:
                        # Calculate scaling factor
                        scale_width = display_width / frame.width
                        scale_height = display_height / frame.height
                        scale = min(scale_width, scale_height)
                        
                        # Calculate new dimensions
                        new_width = int(frame.width * scale)
                        new_height = int(frame.height * scale)
                        
                        # Resize frame
                        frame = frame.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage and update display
                    photo = PIL.ImageTk.PhotoImage(image=frame)
                    self.video_label.configure(image=photo)
                    self.video_label.image = photo  # Keep a reference
            
            # Schedule next update
            self.root.after(30, self.update_ui)
    
    def stop(self):
        """Stop the monitoring system."""
        self.is_running = False
        
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        
        self.camera.release()
        self.root.destroy()
        self.logger.info("Baby Monitor system stopped")

def main():
    """Main entry point."""
    app = BabyMonitorApp()
    app.start()

if __name__ == "__main__":
    main() 