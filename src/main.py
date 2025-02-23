"""
Baby Monitor System - Main Module
===============================
Main entry point for the Baby Monitor System.
Coordinates all components and handles the main application loop.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import logging
import threading
import time
from datetime import datetime
import PIL.Image, PIL.ImageTk
import numpy as np

from utils.config import Config
from utils.camera import Camera
from detectors.person_detector import PersonDetector
from detectors.motion_detector import MotionDetector
from detectors.emotion_detector import EmotionDetector
from ui.alert_manager import AlertManager

class BabyMonitorApp:
    def __init__(self):
        """Initialize the Baby Monitor application."""
        # Setup logging
        logging.basicConfig(**Config.LOGGING)
        
        # Add custom filter to root logger
        yolo_filter = Config.YOLOFilter()
        for handler in logging.getLogger().handlers:
            handler.addFilter(yolo_filter)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize state
        self.is_running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.camera_error = False
        
        # Initialize components
        self.camera = Camera(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
        self.person_detector = PersonDetector(Config.PERSON_DETECTION)
        self.motion_detector = MotionDetector(Config.MOTION_DETECTION)
        
        # Try to initialize emotion detector (optional)
        try:
            self.emotion_detector = EmotionDetector(Config.EMOTION_DETECTION)
            self.has_emotion_detector = True
        except Exception as e:
            self.logger.warning(f"Emotion detection disabled: {str(e)}")
            self.has_emotion_detector = False
        
        # Initialize audio capture if emotion detection is available
        if self.has_emotion_detector:
            self.audio_buffer = []
            self.setup_audio()
        
        # Initialize UI
        self.root = tk.Tk()
        self.setup_ui()
        self.alert_manager = AlertManager(self.root, Config.ALERT, self.alert_container)
        
        # Start clock update
        self.update_clock()
        
        self.logger.info("Baby Monitor application initialized")
    
    def setup_audio(self):
        """Setup audio capture for emotion detection."""
        try:
            import pyaudio
            
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=Config.EMOTION_DETECTION['sampling_rate'],
                input=True,
                frames_per_buffer=Config.EMOTION_DETECTION['chunk_size'],
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            self.logger.info("Audio capture initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio capture: {str(e)}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Handle incoming audio data."""
        try:
            # Skip processing if emotion detection is not available
            if not self.has_emotion_detector:
                return (in_data, pyaudio.paContinue)
                
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Add to buffer
            self.audio_buffer.append(audio_data)
            
            # Keep only last second of audio
            max_chunks = Config.EMOTION_DETECTION['sampling_rate'] // Config.EMOTION_DETECTION['chunk_size']
            if len(self.audio_buffer) > max_chunks:
                self.audio_buffer.pop(0)
            
            # Process emotions if we have enough data
            if len(self.audio_buffer) == max_chunks:
                # Combine chunks
                audio_data = np.concatenate(self.audio_buffer)
                
                # Detect emotions
                emotion, confidence = self.emotion_detector.detect(audio_data)
                
                if emotion:
                    # Get alert level
                    level = self.emotion_detector.get_emotion_level(emotion, confidence)
                    
                    if level:
                        # Show alert in UI thread
                        message = f"Detected {emotion.lower()} emotion (confidence: {confidence:.2f})"
                        self.root.after(0, lambda: self.show_alert(message, level))
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
        
        return (in_data, pyaudio.paContinue)
    
    def setup_ui(self):
        """Setup the main UI window."""
        self.root.title("Baby Monitor System")
        self.root.configure(bg=Config.UI['dark_theme']['background'])
        
        # Configure styles
        style = ttk.Style()
        
        # Main styles
        style.configure('TFrame', background=Config.UI['dark_theme']['background'])
        style.configure('VideoFrame.TFrame', background=Config.UI['dark_theme']['accent'])
        style.configure('Alert.TFrame', background=Config.UI['dark_theme']['alert'])
        
        # Camera selection dialog styles
        style.configure('Camera.TFrame', 
            background=Config.UI['dark_theme']['accent'],
            borderwidth=1,
            relief='solid'
        )
        style.configure('CameraHover.TFrame',
            background=Config.UI['dark_theme']['accent_text'],
            borderwidth=1,
            relief='solid'
        )
        
        # Button styles
        style.configure('TButton',
            padding=6,
            background=Config.UI['dark_theme']['accent'],
            foreground=Config.UI['dark_theme']['text']
        )
        style.map('TButton',
            background=[('active', Config.UI['dark_theme']['accent_text'])],
            foreground=[('active', Config.UI['dark_theme']['background'])]
        )
        
        # Create menu bar
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)
        
        # Camera menu
        self.camera_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Camera", menu=self.camera_menu)
        self.camera_menu.add_command(label="Select Camera", command=self.show_camera_selection)
        self.camera_menu.add_command(label="Camera Properties", command=self.show_camera_properties)
        self.camera_menu.add_separator()
        self.camera_menu.add_command(label="Retry Connection", command=self.retry_camera)
        
        # Set minimum window size and calculate initial size
        self.root.minsize(Config.UI['min_width'], Config.UI['min_height'])
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        initial_width = int(screen_width * Config.UI['window_scale'])
        initial_height = int(screen_height * Config.UI['window_scale'])
        self.root.geometry(f"{initial_width}x{initial_height}")
        
        # Create header frame
        self.header_frame = ttk.Frame(self.root)
        self.header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        # Title with baby icon
        self.title_label = ttk.Label(
            self.header_frame,
            text="👶 Baby Monitor",
            font=('Segoe UI', 28, 'bold'),
            foreground=Config.UI['dark_theme']['accent_text'],
            background=Config.UI['dark_theme']['background']
        )
        self.title_label.pack(side=tk.LEFT)
        
        # Time and date display
        self.datetime_frame = ttk.Frame(self.header_frame)
        self.datetime_frame.pack(side=tk.RIGHT)
        
        self.time_label = ttk.Label(
            self.datetime_frame,
            text="",
            font=('Segoe UI', 24),
            foreground=Config.UI['dark_theme']['accent_text'],
            background=Config.UI['dark_theme']['background']
        )
        self.time_label.pack()
        
        self.date_label = ttk.Label(
            self.datetime_frame,
            text="",
            font=('Segoe UI', 14),
            foreground=Config.UI['dark_theme']['text'],
            background=Config.UI['dark_theme']['background']
        )
        self.date_label.pack()
        
        # Create main container with grid layout
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Configure grid weights for proper scaling
        self.main_container.grid_columnconfigure(0, weight=3)  # Video takes 75% of space
        self.main_container.grid_columnconfigure(1, weight=1)  # Alerts take 25% of space
        self.main_container.grid_rowconfigure(0, weight=1)     # Allow vertical expansion
        
        # Left side - Video feed
        self.video_container = ttk.Frame(self.main_container)
        self.video_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Configure video container grid
        self.video_container.grid_rowconfigure(1, weight=1)    # Video frame expands vertically
        self.video_container.grid_columnconfigure(0, weight=1) # Video frame expands horizontally
        
        # Video feed title
        self.video_title = ttk.Label(
            self.video_container,
            text="📹 Live View",
            font=('Segoe UI', 18, 'bold'),
            foreground=Config.UI['dark_theme']['accent_text'],
            background=Config.UI['dark_theme']['background']
        )
        self.video_title.grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # Video frame with dark background
        self.video_frame = ttk.Frame(
            self.video_container,
            style='VideoFrame.TFrame'
        )
        self.video_frame.grid(row=1, column=0, sticky="nsew")
        
        # Create a container frame for the video label to center it
        self.video_label_container = ttk.Frame(
            self.video_frame,
            style='VideoFrame.TFrame'
        )
        self.video_label_container.place(relx=0.5, rely=0.5, anchor="center")
        
        # Video display label
        self.video_label = ttk.Label(self.video_label_container)
        self.video_label.pack(padx=2, pady=2)
        
        # Bind resize event to video frame
        self.video_frame.bind('<Configure>', self._on_video_frame_resize)
        
        # Right side - Alerts and Status
        self.alert_container = ttk.Frame(self.main_container)
        self.alert_container.grid(row=0, column=1, sticky="nsew")
        
        # Configure alert container grid
        self.alert_container.grid_rowconfigure(2, weight=1)  # History section expands
        self.alert_container.grid_columnconfigure(0, weight=1)
        
        # Alerts & Status title
        self.alert_title = ttk.Label(
            self.alert_container,
            text="🔔 Alerts & Status",
            font=('Segoe UI', 18, 'bold'),
            foreground=Config.UI['dark_theme']['accent_text'],
            background=Config.UI['dark_theme']['background']
        )
        self.alert_title.grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # Status indicators frame
        self.status_frame = ttk.Frame(self.alert_container)
        self.status_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        # Monitoring status with icon
        self.monitoring_status = ttk.Label(
            self.status_frame,
            text="✅ MONITORING",
            font=('Segoe UI', 14),
            foreground='#00ff00',
            background=Config.UI['dark_theme']['background']
        )
        self.monitoring_status.pack(side=tk.LEFT, padx=5)
        
        # Person count with icon
        self.person_count_label = ttk.Label(
            self.status_frame,
            text="👥 People: 0",
            font=('Segoe UI', 14),
            foreground=Config.UI['dark_theme']['text'],
            background=Config.UI['dark_theme']['background']
        )
        self.person_count_label.pack(side=tk.RIGHT, padx=5)
        
        # Create alert manager container that will expand to fill remaining space
        self.alert_manager_container = ttk.Frame(self.alert_container)
        self.alert_manager_container.grid(row=2, column=0, sticky="nsew")
        
        # Initialize alert manager
        self.alert_manager = AlertManager(self.root, Config.ALERT, self.alert_manager_container)
        
        # Configure window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
    
    def update_clock(self):
        """Update the time and date display."""
        now = datetime.now()
        self.time_label.configure(text=now.strftime("%I:%M:%S %p"))
        self.date_label.configure(text=now.strftime("%A, %B %d, %Y"))
        self.root.after(1000, self.update_clock)
    
    def show_camera_error(self):
        """Display camera error message with icon."""
        if not self.camera_error:  # Only show error if not already shown
            self.video_label.pack_forget()
            self.error_label.pack(expand=True)
            self.retry_button.pack(pady=10)
            self.monitoring_status.configure(
                text="❌ CAMERA ERROR",
                foreground='red'
            )
            self.camera_error = True
            
            # Show alert
            self.show_alert(
                "Camera connection lost! Click 'Retry Connection' to reconnect.",
                "critical"
            )
    
    def hide_camera_error(self):
        """Hide camera error message."""
        self.error_label.pack_forget()
        self.retry_button.pack_forget()
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.monitoring_status.configure(
            text="✅ MONITORING",
            foreground='#00ff00'
        )
        self.camera_error = False
    
    def retry_camera(self):
        """Attempt to reconnect to the camera."""
        try:
            # Update status
            self.monitoring_status.configure(
                text="🔄 RECONNECTING...",
                foreground='yellow'
            )
            self.root.update()
            
            # Release existing camera if any
            self.camera.release()
            
            # Try to initialize camera
            if self.camera.initialize():
                self.hide_camera_error()
                self.is_running = True
                self.process_thread = threading.Thread(target=self.process_frames)
                self.process_thread.start()
                self.logger.info("Camera reconnected successfully")
                
                # Show success message
                self.show_alert(
                    "Camera reconnected successfully!",
                    "warning"
                )
            else:
                self.show_camera_error()
                messagebox.showerror(
                    "Camera Error",
                    "Failed to connect to camera. Please check:\n\n" +
                    "1. Camera is properly connected\n" +
                    "2. No other application is using the camera\n" +
                    "3. Camera drivers are installed correctly\n\n" +
                    "Try disconnecting and reconnecting the camera."
                )
        except Exception as e:
            self.logger.error(f"Error retrying camera connection: {str(e)}")
            self.show_camera_error()
            messagebox.showerror(
                "Camera Error",
                f"Failed to connect to camera:\n{str(e)}\n\n" +
                "Please check your camera connection and drivers."
            )
    
    def start(self):
        """Start the monitoring system."""
        try:
            # Initialize camera
            if not self.camera.initialize():
                self.logger.error("Failed to initialize camera")
                self.show_camera_error()
                self.root.mainloop()
                return
            
            self.is_running = True
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self.process_frames)
            self.process_thread.start()
            
            # Start main loop
            self.root.mainloop()
            
        except Exception as e:
            self.logger.error(f"Error starting application: {str(e)}")
            self.show_camera_error()
            self.root.mainloop()
    
    def process_frames(self):
        """Process video frames in a separate thread."""
        consecutive_errors = 0
        max_consecutive_errors = 5
        error_cooldown = 0
        
        while self.is_running:
            try:
                # Get frame from camera
                ret, frame = self.camera.get_frame()
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        if time.time() > error_cooldown:
                            self.logger.error("Multiple consecutive camera errors detected")
                            self.root.after(0, self.show_camera_error)
                            error_cooldown = time.time() + 30
                        time.sleep(0.5)
                        continue
                    time.sleep(0.1)
                    continue
                
                # Reset error counter on successful frame
                consecutive_errors = 0
                
                # Create a copy for processing
                process_frame = frame.copy()
                
                try:
                    # Detect people
                    frame, person_count, boxes = self.person_detector.detect(process_frame)
                    
                    # Update person count in UI
                    self.person_count_label.configure(text=f"👥 People: {person_count}")
                    
                    # Detect motion and falls
                    frame, rapid_motion, fall_detected = self.motion_detector.detect(frame, boxes)
                    
                    # Update frame
                    with self.frame_lock:
                        self.frame = frame
                    
                    # Convert frame for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = PIL.Image.fromarray(frame_rgb)
                    
                    # Get current video frame dimensions
                    video_frame_width = self.video_frame.winfo_width()
                    video_frame_height = self.video_frame.winfo_height()
                    
                    if video_frame_width > 1 and video_frame_height > 1:  # Ensure valid dimensions
                        # Calculate aspect ratios
                        frame_aspect = frame_pil.width / frame_pil.height
                        container_aspect = video_frame_width / video_frame_height
                        
                        # Calculate new dimensions maintaining aspect ratio
                        if frame_aspect > container_aspect:
                            # Frame is wider than container
                            new_width = video_frame_width
                            new_height = int(video_frame_width / frame_aspect)
                        else:
                            # Frame is taller than container
                            new_height = video_frame_height
                            new_width = int(video_frame_height * frame_aspect)
                        
                        # Ensure dimensions don't exceed container
                        new_width = min(new_width, video_frame_width)
                        new_height = min(new_height, video_frame_height)
                        
                        # Resize frame
                        frame_pil = frame_pil.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage and update display
                    photo = PIL.ImageTk.PhotoImage(image=frame_pil)
                    self.video_label.configure(image=photo)
                    self.video_label.image = photo
                    
                    # Handle alerts
                    if fall_detected:
                        self.show_alert(
                            "FALL DETECTED! Check on person immediately!",
                            "critical"
                        )
                    elif rapid_motion:
                        self.show_alert(
                            "RAPID MOTION DETECTED! Monitor situation",
                            "warning"
                        )
                
                except Exception as e:
                    self.logger.error(f"Error processing detection: {str(e)}")
                    continue
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error processing frame: {str(e)}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    if time.time() > error_cooldown:
                        self.logger.error("Multiple consecutive camera errors detected")
                        self.root.after(0, self.show_camera_error)
                        error_cooldown = time.time() + 30
                    time.sleep(0.5)
                    continue
                time.sleep(0.1)
    
    def show_alert(self, message, level):
        """Show alert in the GUI."""
        self.alert_manager.show_alert(message, level)
    
    def stop(self):
        """Stop the monitoring system."""
        self.is_running = False
        
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        
        # Stop audio
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        
        self.camera.release()
        self.root.destroy()
        self.logger.info("Baby Monitor system stopped")
    
    def show_camera_selection(self):
        """Show camera selection dialog."""
        cameras = self.camera.get_available_cameras()
        if not cameras:
            messagebox.showwarning(
                "No Cameras",
                "No cameras were found.\nPlease connect a camera and try again."
            )
            return
        
        # Create camera selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Camera")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Configure dialog style
        dialog.configure(bg=Config.UI['dark_theme']['background'])
        
        # Add title label with icon
        title_frame = ttk.Frame(dialog)
        title_frame.pack(fill=tk.X, padx=20, pady=10)
        
        title_label = ttk.Label(
            title_frame,
            text="📹 Select Camera",
            font=('Arial', 20, 'bold'),
            foreground=Config.UI['dark_theme']['accent_text'],
            background=Config.UI['dark_theme']['background']
        )
        title_label.pack(side=tk.LEFT)
        
        # Add refresh button
        refresh_button = ttk.Button(
            title_frame,
            text="🔄 Refresh",
            command=lambda: self._refresh_camera_list(buttons_frame, cameras)
        )
        refresh_button.pack(side=tk.RIGHT)
        
        # Create main container
        main_container = ttk.Frame(dialog)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        # Create scrollable frame for camera list
        canvas = tk.Canvas(
            main_container,
            bg=Config.UI['dark_theme']['background'],
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(main_container, orient=tk.VERTICAL, command=canvas.yview)
        
        # Create frame for camera buttons
        buttons_frame = ttk.Frame(canvas)
        buttons_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )
        
        # Create window in canvas
        canvas.create_window((0, 0), window=buttons_frame, anchor='nw', width=440)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def select_camera(index):
            if self.select_camera(index):
                dialog.destroy()
                self.restart_camera()
            else:
                messagebox.showerror(
                    "Camera Error",
                    "Failed to initialize selected camera.\nPlease try another camera."
                )
        
        # Add camera buttons
        self._populate_camera_list(buttons_frame, cameras, select_camera)
        
        # Add bottom button bar
        button_bar = ttk.Frame(dialog)
        button_bar.pack(fill=tk.X, padx=20, pady=10)
        
        # Add help button
        help_button = ttk.Button(
            button_bar,
            text="❓ Help",
            command=self._show_camera_help
        )
        help_button.pack(side=tk.LEFT)
        
        # Add close button
        close_button = ttk.Button(
            button_bar,
            text="Close",
            command=dialog.destroy
        )
        close_button.pack(side=tk.RIGHT)
        
        # Center dialog on screen
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')
        
        # Make dialog resizable
        dialog.resizable(True, True)
        dialog.minsize(400, 300)
    
    def _populate_camera_list(self, parent, cameras, callback):
        """Populate the camera list with camera buttons."""
        # Clear existing buttons
        for widget in parent.winfo_children():
            widget.destroy()
        
        # Add camera buttons
        for i, cam in enumerate(cameras):
            # Create frame for camera item
            camera_frame = ttk.Frame(parent, style='Camera.TFrame')
            camera_frame.pack(fill=tk.X, pady=5)
            
            # Create inner frame for content
            content_frame = ttk.Frame(camera_frame, style='Camera.TFrame')
            content_frame.pack(fill=tk.X, padx=10, pady=10)
            
            # Camera name with icon and index
            name_label = ttk.Label(
                content_frame,
                text=f"📹 {cam['name']} (ID: {cam['index']})",
                font=('Arial', 12, 'bold'),
                foreground=Config.UI['dark_theme']['text'],
                background=Config.UI['dark_theme']['background']
            )
            name_label.pack(anchor='w')
            
            # Camera details frame
            details_frame = ttk.Frame(content_frame)
            details_frame.pack(fill=tk.X, pady=(5, 0))
            
            # Resolution info
            resolution_label = ttk.Label(
                details_frame,
                text=f"🖥️ Resolution: {cam.get('resolution', 'Unknown')}",
                font=('Arial', 10),
                foreground=Config.UI['dark_theme']['text'],
                background=Config.UI['dark_theme']['background']
            )
            resolution_label.pack(side=tk.LEFT)
            
            # Status indicator (current/available)
            is_current = i == self.camera.selected_camera_index
            status_label = ttk.Label(
                details_frame,
                text="✅ Current" if is_current else "⚪ Available",
                font=('Arial', 10),
                foreground='#00ff00' if is_current else Config.UI['dark_theme']['text'],
                background=Config.UI['dark_theme']['background']
            )
            status_label.pack(side=tk.RIGHT)
            
            # Button frame
            button_frame = ttk.Frame(content_frame)
            button_frame.pack(fill=tk.X, pady=(10, 0))
            
            # Select button
            select_button = ttk.Button(
                button_frame,
                text="Select Camera",
                command=lambda idx=cam['index']: callback(idx)  # Use actual camera index
            )
            select_button.pack(side=tk.LEFT)
            
            # Properties button (only enabled for current camera)
            properties_button = ttk.Button(
                button_frame,
                text="Properties",
                command=self.show_camera_properties if is_current else lambda: None,
                state='normal' if is_current else 'disabled'
            )
            properties_button.pack(side=tk.RIGHT)
            
            # Make the frame clickable (except for buttons)
            for widget in [camera_frame, content_frame, name_label, resolution_label, status_label]:
                widget.bind('<Button-1>', lambda e, idx=cam['index']: callback(idx))
                widget.bind('<Enter>', lambda e, f=camera_frame: self._on_camera_hover(f, True))
                widget.bind('<Leave>', lambda e, f=camera_frame: self._on_camera_hover(f, False))
    
    def _on_camera_hover(self, frame, is_hover):
        """Handle camera item hover effect."""
        if is_hover:
            frame.configure(style='CameraHover.TFrame')
        else:
            frame.configure(style='Camera.TFrame')
    
    def _refresh_camera_list(self, buttons_frame, cameras):
        """Refresh the camera list."""
        # Re-enumerate cameras
        self.camera._enumerate_cameras()
        cameras = self.camera.get_available_cameras()
        
        # Repopulate the list
        self._populate_camera_list(
            buttons_frame,
            cameras,
            lambda idx: self.select_camera(idx)
        )
    
    def _show_camera_help(self):
        """Show camera help dialog."""
        help_text = """
Camera Selection Help

1. Available Cameras
   • Each camera shows its name and resolution
   • Current camera is marked with ✅
   • Click any camera to switch to it

2. Troubleshooting
   • If a camera isn't listed, click 🔄 Refresh
   • Make sure the camera is properly connected
   • Check if other applications are using the camera
   • Try disconnecting and reconnecting the camera

3. Camera Types
   • USB Webcams
   • Built-in Laptop Cameras
   • IP Cameras (if configured)
   • Raspberry Pi Cameras

Need more help? Check the documentation or contact support.
"""
        
        messagebox.showinfo("Camera Selection Help", help_text)
    
    def restart_camera(self):
        """Restart camera processing after camera change."""
        # Stop current processing
        self.is_running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        
        # Start new processing
        self.is_running = True
        self.process_thread = threading.Thread(target=self.process_frames)
        self.process_thread.start()
        
        # Update UI
        self.hide_camera_error()
        self.show_alert(
            "Camera changed successfully!",
            "warning"
        )

    def select_camera(self, index):
        """Select and initialize a camera."""
        try:
            # Show loading indicator
            loading_window = tk.Toplevel(self.root)
            loading_window.title("Switching Camera")
            loading_window.geometry("300x100")
            loading_window.transient(self.root)
            loading_window.grab_set()
            loading_window.configure(bg=Config.UI['dark_theme']['background'])
            
            # Center loading window
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 150
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 50
            loading_window.geometry(f"+{x}+{y}")
            
            # Add loading message
            loading_label = ttk.Label(
                loading_window,
                text="🔄 Switching Camera...\nPlease wait",
                font=('Arial', 12),
                foreground=Config.UI['dark_theme']['text'],
                background=Config.UI['dark_theme']['background'],
                justify=tk.CENTER
            )
            loading_label.pack(expand=True)
            
            # Update UI
            self.root.update()
            
            # Stop current camera processing
            self.is_running = False
            if hasattr(self, 'process_thread'):
                self.process_thread.join(timeout=1.0)  # Wait max 1 second
            
            # Release current camera
            self.camera.release()
            
            # Select new camera
            success = self.camera.select_camera(index)
            
            if success:
                # Start new camera processing
                self.is_running = True
                self.process_thread = threading.Thread(target=self.process_frames)
                self.process_thread.start()
                
                # Update UI
                self.hide_camera_error()
                self.show_alert(
                    "Camera changed successfully!",
                    "warning"
                )
                loading_window.destroy()
                return True
            else:
                loading_window.destroy()
                messagebox.showerror(
                    "Camera Error",
                    "Failed to initialize selected camera.\nPlease try another camera."
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Error switching camera: {str(e)}")
            loading_window.destroy()
            messagebox.showerror(
                "Camera Error",
                f"Failed to switch camera:\n{str(e)}"
            )
            return False

    def show_camera_properties(self):
        """Show native camera properties dialog on Windows."""
        if not self.camera or not self.camera.camera:
            messagebox.showwarning(
                "No Camera",
                "No camera is currently active.\nPlease select a camera first."
            )
            return
        
        try:
            # Try to open native camera properties dialog
            self.camera.camera.set(cv2.CAP_PROP_SETTINGS, 1)
        except Exception as e:
            self.logger.error(f"Error opening camera properties: {str(e)}")
            messagebox.showerror(
                "Error",
                "Failed to open camera properties.\nThis feature may not be supported by your camera."
            )

    def _on_video_frame_resize(self, event):
        """Handle video frame resize events to maintain aspect ratio."""
        if hasattr(self, 'frame') and self.frame is not None:
            # Get current frame
            frame = self.frame
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = PIL.Image.fromarray(frame_rgb)
            
            # Get container dimensions
            container_width = self.video_frame.winfo_width()
            container_height = self.video_frame.winfo_height()
            
            # Calculate aspect ratios
            frame_aspect = frame_pil.width / frame_pil.height
            container_aspect = container_width / container_height
            
            # Calculate new dimensions to fit container while maintaining aspect ratio
            if frame_aspect > container_aspect:
                # Frame is wider than container
                new_width = container_width
                new_height = int(container_width / frame_aspect)
            else:
                # Frame is taller than container
                new_height = container_height
                new_width = int(container_height * frame_aspect)
            
            # Ensure dimensions don't exceed container
            new_width = min(new_width, container_width)
            new_height = min(new_height, container_height)
            
            # Resize frame
            frame_pil = frame_pil.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
            
            # Update display
            photo = PIL.ImageTk.PhotoImage(image=frame_pil)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
            
            # Update video label container size
            self.video_label_container.configure(width=new_width, height=new_height)

def main():
    """Main entry point."""
    app = BabyMonitorApp()
    app.start()

if __name__ == "__main__":
    main() 