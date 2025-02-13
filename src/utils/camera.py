"""
Camera Module
============
Handles camera initialization and frame capture for different platforms.
"""

import cv2
import platform
import logging
import time

class Camera:
    def __init__(self, width=1280, height=720):
        """
        Initialize camera with specified resolution.
        
        Args:
            width (int): Desired camera width
            height (int): Desired camera height
        """
        self.logger = logging.getLogger(__name__)
        self.width = width
        self.height = height
        self.camera = None
        self.is_raspberry_pi = platform.machine() in ('armv7l', 'aarch64')
        self.backend_index = 0
        self.available_backends = self._get_available_backends()
        self.selected_camera_index = 0
        self.available_cameras = []
        self._enumerate_cameras()
    
    def _get_available_backends(self):
        """Get list of available camera backends for the current platform."""
        backends = []
        if platform.system() == 'Windows':
            backends = [
                cv2.CAP_DSHOW,      # DirectShow (preferred for Windows)
                cv2.CAP_MSMF,       # Microsoft Media Foundation
                cv2.CAP_ANY         # Auto-detect
            ]
        else:
            backends = [
                cv2.CAP_V4L2,       # Video4Linux2 (preferred for Linux)
                cv2.CAP_ANY         # Auto-detect
            ]
        return backends
    
    def _enumerate_cameras(self):
        """Enumerate all available cameras."""
        self.available_cameras = []
        
        if self.is_raspberry_pi:
            try:
                from picamera2 import Picamera2
                cameras = Picamera2.global_camera_info()
                for i, cam in enumerate(cameras):
                    self.available_cameras.append({
                        'index': i,
                        'name': f"Raspberry Pi Camera {i}",
                        'info': cam
                    })
            except ImportError:
                self.logger.warning("PiCamera2 module not found")
        else:
            # Simple camera enumeration
            for i in range(5):  # Check first 5 indices
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            # Get camera info
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            self.available_cameras.append({
                                'index': i,
                                'name': f"Camera {i}",
                                'resolution': f"{width}x{height}"
                            })
                        cap.release()
                except Exception as e:
                    self.logger.debug(f"Failed to check camera {i}: {str(e)}")
                    continue
        
        self.logger.info(f"Found {len(self.available_cameras)} cameras: {[c['index'] for c in self.available_cameras]}")
        return self.available_cameras
    
    def get_available_cameras(self):
        """Return list of available cameras."""
        return self.available_cameras
    
    def select_camera(self, index):
        """
        Select camera by index.
        
        Args:
            index (int): Camera index to select
            
        Returns:
            bool: True if camera was selected successfully
        """
        if 0 <= index < len(self.available_cameras):
            self.selected_camera_index = index
            # Release current camera if any
            self.release()
            # Initialize new camera
            return self.initialize()
        return False
    
    def initialize(self):
        """Initialize the camera based on platform."""
        if self.is_raspberry_pi:
            return self._initialize_raspberry_pi()
        else:
            return self._initialize_webcam()
    
    def _initialize_raspberry_pi(self):
        """Initialize camera for Raspberry Pi."""
        try:
            from picamera2 import Picamera2
            self.camera = Picamera2(self.selected_camera_index)
            config = self.camera.create_preview_configuration(
                main={"size": (self.width, self.height)},
                lores={"size": (640, 480)},
                display="lores"
            )
            self.camera.configure(config)
            self.camera.start()
            self.logger.info(f"Raspberry Pi camera {self.selected_camera_index} initialized successfully")
            return True
        except ImportError:
            self.logger.error("PiCamera2 module not found. Please install python3-picamera2")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize Raspberry Pi camera: {str(e)}")
            return False
    
    def _initialize_webcam(self):
        """Initialize webcam with fallback to different backends."""
        if not self.available_cameras:
            self.logger.error("No cameras available")
            return False
        
        try:
            # Try direct index first (simpler approach)
            self.camera = cv2.VideoCapture(self.selected_camera_index)
            if self.camera.isOpened():
                # Configure camera settings
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Test if camera actually works
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    self.logger.info(f"Camera initialized successfully with index {self.selected_camera_index}")
                    return True
            
            # If direct index failed, try with backend
            self.camera.release()
            camera_info = self.available_cameras[self.selected_camera_index]
            backend = camera_info.get('backend', cv2.CAP_ANY)
            
            self.logger.info(f"Trying camera {self.selected_camera_index} with backend {backend}")
            self.camera = cv2.VideoCapture(self.selected_camera_index + backend)
            
            if not self.camera.isOpened():
                self.logger.error("Failed to open camera with backend")
                return False
            
            # Configure camera
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret or frame is None:
                self.logger.error("Camera opened but failed to provide frames")
                self.camera.release()
                return False
            
            self.logger.info(f"Camera initialized successfully with backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {str(e)}")
            if self.camera is not None:
                self.camera.release()
            return False
    
    def get_frame(self):
        """
        Get a frame from the camera with error handling.
        
        Returns:
            tuple: (success, frame)
        """
        if self.camera is None:
            return False, None
            
        try:
            if self.is_raspberry_pi:
                try:
                    frame = self.camera.capture_array()
                    return True, frame
                except Exception as e:
                    self.logger.error(f"Error capturing frame from Pi camera: {str(e)}")
                    return False, None
            else:
                for _ in range(3):  # Try up to 3 times to get a valid frame
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        return True, frame
                    time.sleep(0.1)  # Short delay between retries
                
                self.logger.warning("Failed to get valid frame after 3 attempts")
                return False, None
                
        except Exception as e:
            self.logger.error(f"Error capturing frame: {str(e)}")
            return False, None
    
    def release(self):
        """Release the camera resources with error handling."""
        if self.camera is not None:
            try:
                if self.is_raspberry_pi:
                    self.camera.stop()
                else:
                    self.camera.release()
                self.logger.info("Camera released successfully")
            except Exception as e:
                self.logger.error(f"Error releasing camera: {str(e)}")
            finally:
                self.camera = None 