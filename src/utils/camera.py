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

        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
                if cap.isOpened():
                    # Get camera name if possible
                    if platform.system() == "Windows":
                        cap.set(cv2.CAP_PROP_SETTINGS, 1)
                    
                    # Get supported resolutions
                    resolutions = self._get_supported_resolutions(cap)
                    
                    self.available_cameras.append({
                        'id': i,
                        'name': f'Camera {i}',
                        'resolutions': resolutions
                    })
                cap.release()
            except Exception as e:
                self.logger.debug(f"Error checking camera {i}: {str(e)}")

        self.logger.info(f"Found {len(self.available_cameras)} camera(s)")

    def get_available_cameras(self):
        """Get list of available camera indices."""
        return self.available_cameras

    def select_camera(self, index):
        """Select a specific camera by index."""
        if index in self.available_cameras:
            self.selected_camera_index = index
            return True
        return False

    def initialize(self):
        """Initialize the camera with the current settings."""
        if self._is_initialized:
            return True

        try:
            # Try to open the camera with the current backend
            self.cap = cv2.VideoCapture(self.selected_camera_index, self.available_backends[self.backend_index])
            
            if not self.cap.isOpened():
                self.logger.error("Failed to open camera")
                return False

            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Get actual resolution (may be different from requested)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.logger.info(f"Camera initialized at {actual_width}x{actual_height}")

            self._is_initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Error initializing camera: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def get_frame(self):
        """Get a frame from the camera."""
        if not self._is_initialized or not self.cap:
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.logger.error("Failed to read frame")
                return False, None
            return True, frame

        except Exception as e:
            self.logger.error(f"Error getting frame: {str(e)}")
            return False, None

    def release(self):
        """Release the camera."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self._is_initialized = False

    @property
    def is_initialized(self):
        """Check if camera is initialized."""
        return self._is_initialized
