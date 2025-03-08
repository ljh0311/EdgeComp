"""
Camera module for video capture and processing.
"""

import cv2
import logging
import platform
import numpy as np


class Camera:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.cap = None
        self.logger = logging.getLogger(__name__)
        self.is_raspberry_pi = platform.machine() in ("armv7l", "aarch64")
        self.backend_index = 0
        self.available_backends = self._get_available_backends()
        self.selected_camera_index = 0
        self.available_cameras = []
        self.standard_resolutions = [
            (640, 480),    # VGA
            (1280, 720),   # HD
            (1920, 1080)   # Full HD
        ]
        self._enumerate_cameras()

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
                backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
                cap = cv2.VideoCapture(i, backend)
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

    def _get_supported_resolutions(self, cap):
        """Get supported resolutions for a camera."""
        standard_resolutions = [
            (640, 480),   # VGA
            (1280, 720),  # HD
            (1920, 1080)  # Full HD
        ]
        
        supported = []
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        for width, height in standard_resolutions:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if (actual_width, actual_height) not in [(r['width'], r['height']) for r in supported]:
                    supported.append({
                        'width': actual_width,
                        'height': actual_height
                    })
            except Exception as e:
                self.logger.debug(f"Error checking resolution {width}x{height}: {str(e)}")
        
        # Restore original resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)
        
        return supported

    def get_camera_list(self):
        """Get list of available camera names."""
        return [cam['name'] for cam in self.available_cameras]

    def get_camera_resolutions(self, camera_name):
        """Get supported resolutions for a camera."""
        for cam in self.available_cameras:
            if cam['name'] == camera_name:
                return [f"{res['width']}x{res['height']}" for res in cam['resolutions']]
        return [f"{width}x{height}" for width, height in self.standard_resolutions]

    def select_camera(self, camera_name):
        """Select a camera by name."""
        try:
            for cam in self.available_cameras:
                if cam['name'] == camera_name:
                    if self.cap is not None:
                        self.release()
                    self.selected_camera_index = cam['id']
                    success = self.initialize()
                    if success:
                        self.logger.info(f"Selected camera: {camera_name}")
                        return True
            return False
        except Exception as e:
            self.logger.error(f"Error selecting camera: {str(e)}")
            return False

    def set_resolution(self, width, height=None):
        """Set camera resolution.
        
        Args:
            width: Either the width as an integer, or a resolution string in format 'WIDTHxHEIGHT'
            height: The height as an integer (optional if width is a resolution string)
            
        Returns:
            bool: True if resolution was set successfully
        """
        try:
            # Handle resolution string format (e.g. '1280x720')
            if isinstance(width, str):
                width, height = map(int, width.split('x'))
            
            # Validate height parameter is provided
            if height is None:
                raise ValueError("Height parameter is required when width is not a resolution string")
            
            self.width = width
            self.height = height
            
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return actual_width == width and actual_height == height
            return False
            
        except Exception as e:
            self.logger.error(f"Error setting resolution: {str(e)}")
            return False

    def get_current_resolution(self):
        """Get current camera resolution."""
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return f"{width}x{height}"
        return f"{self.width}x{self.height}"

    def initialize(self):
        """Initialize the camera with the current settings."""
        if self.cap is not None:
            self.release()

        try:
            # Try each available backend until one works
            for backend in self.available_backends:
                try:
                    self.cap = cv2.VideoCapture(self.selected_camera_index, backend)
                    if self.cap.isOpened():
                        break
                except Exception:
                    continue

            if not self.cap or not self.cap.isOpened():
                self.logger.error("Failed to open camera")
                return False

            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Get actual resolution (may be different from requested)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.logger.info(f"Camera initialized at {actual_width}x{actual_height}")

            return True

        except Exception as e:
            self.logger.error(f"Error initializing camera: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def get_frame(self):
        """Get a frame from the camera."""
        if not self.cap or not self.cap.isOpened():
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

    def get_available_cameras(self):
        """Get list of available cameras."""
        return self.available_cameras

    def set_resolution(self, width, height):
        """Set camera resolution."""
        self.width = width
        self.height = height
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return actual_width == width and actual_height == height
        return False 