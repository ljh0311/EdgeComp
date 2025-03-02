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
    def __init__(self, width=640, height=480):
        """Initialize camera with specified resolution."""
        self.width = width
        self.height = height
        self.cap = None
        self._is_initialized = False
        self.logger = logging.getLogger(__name__)
        self.is_raspberry_pi = platform.machine() in ("armv7l", "aarch64")
        self.backend_index = 0
        self.available_backends = self._get_available_backends()
        self.selected_camera_index = 0
        self.available_cameras = []
        
        # Log platform information
        self.logger.info(f"Initializing camera system on {platform.system()}")
        self.logger.info(f"Available backends: {[str(b) for b in self.available_backends]}")
        
        self._enumerate_cameras()
        # Visualization settings
        self.ENABLE_VISUALIZATION = False  # Set to False for Raspberry Pi headless operation

    def _get_available_backends(self):
        """Get list of available camera backends for the current platform."""
        backends = []
        if platform.system() == "Windows":
            backends = [
                cv2.CAP_DSHOW,  # DirectShow (preferred for Windows)
                cv2.CAP_MSMF,  # Microsoft Media Foundation
                cv2.CAP_ANY,  # Auto-detect
            ]
        else:
            backends = [
                cv2.CAP_V4L2,  # Video4Linux2 (preferred for Linux)
                cv2.CAP_ANY,  # Auto-detect
            ]
        return backends

    def _enumerate_cameras(self):
        """Enumerate all available cameras."""
        self.available_cameras = []
        max_cameras = 10  # Maximum number of cameras to check
        self.logger.info("Scanning for available cameras...")

        for i in range(max_cameras):
            try:
                self.logger.debug(f"Checking camera index {i}...")
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
                if cap.isOpened():
                    self.available_cameras.append(i)
                    self.logger.info(f"Found camera at index {i}")
                    # Try to get camera name/description if possible
                    try:
                        if platform.system() == "Windows":
                            # On Windows, try to get camera name
                            name = cap.get(cv2.CAP_PROP_BACKEND)
                            self.logger.info(f"Camera {i} details - Backend: {name}")
                    except:
                        pass
                    cap.release()
                else:
                    self.logger.debug(f"No camera found at index {i}")
            except Exception as e:
                self.logger.debug(f"Error checking camera {i}: {str(e)}")

        if not self.available_cameras:
            self.logger.warning("No cameras found!")
        else:
            self.logger.info(f"Found {len(self.available_cameras)} camera(s) at indices: {self.available_cameras}")

    def get_available_cameras(self):
        """Get list of available camera indices."""
        return self.available_cameras

    def select_camera(self, index):
        """Select a specific camera by index."""
        if index in self.available_cameras:
            self.selected_camera_index = index
            self.logger.info(f"Selected camera {index}")
            return True
        self.logger.warning(f"Camera index {index} is not available")
        return False

    def try_next_camera(self):
        """Try to select the next available camera."""
        current_index = self.selected_camera_index
        self.logger.info(f"Attempting to find next available camera after index {current_index}")
        
        for index in self.available_cameras:
            if index > current_index:
                try:
                    self.logger.debug(f"Trying camera {index}...")
                    cap = cv2.VideoCapture(index, self.available_backends[self.backend_index])
                    if cap.isOpened():
                        cap.release()
                        self.selected_camera_index = index
                        self.logger.info(f"Successfully switched to camera {index}")
                        return True
                except Exception as e:
                    self.logger.debug(f"Error trying camera {index}: {str(e)}")
                    continue
        
        self.logger.warning("No more cameras available to try")
        return False

    def initialize(self):
        """Initialize the camera with the current settings."""
        if self._is_initialized:
            return True

        self.logger.info(f"Attempting to initialize camera {self.selected_camera_index}")
        
        # Try to initialize with current camera index
        for backend in self.available_backends:
            try:
                self.logger.debug(f"Trying backend {backend}...")
                self.cap = cv2.VideoCapture(self.selected_camera_index, backend)
                if self.cap.isOpened():
                    # Set resolution
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                    # Get actual resolution (may be different from requested)
                    actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.logger.info(f"Camera {self.selected_camera_index} initialized successfully at {actual_width}x{actual_height}")

                    self._is_initialized = True
                    return True
                else:
                    self.logger.debug(f"Failed to open camera {self.selected_camera_index} with backend {backend}")
                    self.cap.release()
            except Exception as e:
                self.logger.debug(f"Failed to initialize camera {self.selected_camera_index} with backend {backend}: {str(e)}")
                if self.cap:
                    self.cap.release()
                    self.cap = None

        # If initialization failed, try next available camera
        self.logger.warning(f"Camera {self.selected_camera_index} is in use or not available, trying next camera...")
        if self.try_next_camera():
            return self.initialize()  # Recursively try to initialize with new camera

        self.logger.error("Failed to initialize any camera")
        return False

    def get_frame(self):
        """Get a frame from the camera."""
        if not self._is_initialized or not self.cap:
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.logger.error(f"Failed to read frame from camera {self.selected_camera_index}")
                # Try to reinitialize with next camera
                if self.try_next_camera():
                    self.release()
                    if self.initialize():
                        return self.get_frame()
                return False, None
            return True, frame

        except Exception as e:
            self.logger.error(f"Error getting frame from camera {self.selected_camera_index}: {str(e)}")
            return False, None

    def release(self):
        """Release the camera."""
        if self.cap:
            self.logger.info(f"Releasing camera {self.selected_camera_index}")
            self.cap.release()
            self.cap = None
        self._is_initialized = False

    @property
    def is_initialized(self):
        """Check if camera is initialized."""
        return self._is_initialized
