"""
Camera module for video capture and processing.
"""

import cv2
import logging
import platform
import numpy as np
import time
from threading import Lock


class CameraError(Exception):
    """Custom exception for camera-related errors."""

    pass


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
            (640, 480),  # VGA
            (1280, 720),  # HD
            (1920, 1080),  # Full HD
        ]
        self.lock = Lock()
        self.consecutive_failures = 0
        self.max_failures = 3  # Maximum consecutive failures before reinit
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

        # Try different backends for Windows
        if platform.system() == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        for backend in backends:
            for i in range(max_cameras):
                try:
                    self.logger.info(f"Checking camera {i} with backend {backend}")
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        # Try to read a test frame
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            # Get supported resolutions
                            resolutions = self._get_supported_resolutions(cap)

                        camera_info = {
                            "id": i,
                            "name": f"Camera {i}",
                            "backend": backend,
                            "resolutions": resolutions,
                        }
                        # Only add if not already found
                        if not any(c["id"] == i for c in self.available_cameras):
                            self.available_cameras.append(camera_info)
                            self.logger.info(f"Found camera {i} with backend {backend}")
                        cap.release()
                except Exception as e:
                    self.logger.debug(
                        f"Error checking camera {i} with backend {backend}: {str(e)}"
                    )

        self.logger.info(f"Found {len(self.available_cameras)} camera(s)")
        if not self.available_cameras:
            self.logger.error("No cameras found!")

    def _get_supported_resolutions(self, cap):
        """Get supported resolutions for a camera."""
        standard_resolutions = [
            (640, 480),  # VGA
            (1280, 720),  # HD
            (1920, 1080),  # Full HD
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

                if (actual_width, actual_height) not in [
                    (r["width"], r["height"]) for r in supported
                ]:
                    supported.append({"width": actual_width, "height": actual_height})
            except Exception as e:
                self.logger.debug(
                    f"Error checking resolution {width}x{height}: {str(e)}"
                )

        # Restore original resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)

        return supported

    def get_camera_list(self):
        """Get list of available camera names."""
        return [cam["name"] for cam in self.available_cameras]

    def get_camera_resolutions(self, camera_name):
        """Get supported resolutions for a camera."""
        for cam in self.available_cameras:
            if cam["name"] == camera_name:
                return [f"{res['width']}x{res['height']}" for res in cam["resolutions"]]
        return [f"{width}x{height}" for width, height in self.standard_resolutions]

    def select_camera(self, camera_name):
        """Select a camera by name."""
        try:
            for cam in self.available_cameras:
                if cam["name"] == camera_name:
                    if self.cap is not None:
                        self.release()
                    self.selected_camera_index = cam["id"]
                    success = self.initialize()
                    if success:
                        self.logger.info(f"Selected camera: {camera_name}")
                        return True
            return False
        except Exception as e:
            self.logger.error(f"Error selecting camera: {str(e)}")
            return False

    def set_resolution(self, resolution_str=None, width=None, height=None):
        """Set camera resolution from either string format (e.g., '1280x720') or width/height values."""
        try:
            # Parse resolution from string if provided
            if resolution_str is not None:
                try:
                    width, height = map(int, resolution_str.split("x"))
                except Exception as e:
                    self.logger.error(f"Invalid resolution format: {resolution_str}")
                    return False

            # Validate width and height
            if not width or not height:
                self.logger.error("Invalid resolution values")
                return False

            self.width = width
            self.height = height

            # If camera is open, try to set resolution
            if self.cap and self.cap.isOpened():
                # Release and reinitialize camera with new resolution
                self.release()
                if self.initialize():
                    actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if actual_width == width and actual_height == height:
                        self.logger.info(
                            f"Successfully set resolution to {width}x{height}"
                        )
                        return True
                    else:
                        self.logger.warning(
                            f"Camera set to {actual_width}x{actual_height} instead of requested {width}x{height}"
                        )
                        return False
                else:
                    self.logger.error(
                        "Failed to reinitialize camera with new resolution"
                    )
                    return False
            else:
                self.logger.warning(
                    "Camera not open, resolution will be set on next initialization"
                )
                return True

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

    def _try_reinitialize(self):
        """Attempt to reinitialize the camera after failures."""
        self.logger.warning("Attempting to reinitialize camera...")
        with self.lock:
            if self.cap:
                self.release()
            time.sleep(1)  # Wait before retrying

            # Try to initialize with different backends
            for backend in self._get_available_backends():
                try:
                    self.logger.info(f"Trying backend: {backend}")
                    self.cap = cv2.VideoCapture(self.selected_camera_index, backend)
                    if self.cap.isOpened():
                        # Set resolution
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                        # Try to read a test frame
                        ret, frame = self.cap.read()
                        if ret and frame is not None and frame.size > 0:
                            self.logger.info(
                                f"Successfully reinitialized with backend {backend}"
                            )
                            return True

                        self.cap.release()
                except Exception as e:
                    self.logger.warning(
                        f"Failed to initialize with backend {backend}: {str(e)}"
                    )
                    continue

            self.logger.error("Failed to reinitialize with any backend")
            return False

    def initialize(self):
        """Initialize the camera with the current settings."""
        with self.lock:
            if self.cap is not None:
                self.release()

            # Re-enumerate cameras if none are available
            if not self.available_cameras:
                self._enumerate_cameras()
                if not self.available_cameras:
                    self.logger.error("No cameras available to initialize")
                    return False

            try:
                # Get camera info
                camera_info = next(
                    (
                        cam
                        for cam in self.available_cameras
                        if cam["id"] == self.selected_camera_index
                    ),
                    None,
                )
                if not camera_info:
                    self.logger.error(
                        f"No camera found with index {self.selected_camera_index}"
                    )
                    return False

                # Try to initialize with the camera's preferred backend
                self.logger.info(
                    f"Initializing camera {self.selected_camera_index} with backend {camera_info['backend']}"
                )
                self.cap = cv2.VideoCapture(
                    self.selected_camera_index, camera_info["backend"]
                )

                if not self.cap.isOpened():
                    self.logger.error("Failed to open camera")
                    return False

                    # Set resolution
                self.logger.info(f"Setting resolution to {self.width}x{self.height}")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                # Set buffer size to minimize latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Read multiple test frames to ensure camera is working
                for _ in range(5):
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        # Get actual resolution
                        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        self.logger.info(
                            f"Camera initialized successfully at {actual_width}x{actual_height}"
                        )
                        self.consecutive_failures = 0
                        return True
                    time.sleep(0.1)

                self.logger.error("Failed to read valid test frames")
                self.cap.release()
                self.cap = None
                return False

            except Exception as e:
                self.logger.error(f"Error initializing camera: {str(e)}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return False

    def get_frame(self):
        """Get a frame from the camera with error handling and recovery."""
        if not self.cap:
            self.logger.error("Camera not initialized")
            if not self.initialize():
                self.logger.error("Camera initialization failed")
                return False, None

        try:
            with self.lock:
                if not self.cap.isOpened():
                    self.logger.error("Camera is not opened")
                    if not self._try_reinitialize():
                        return False, None

                # Try to read frame multiple times
                for attempt in range(3):
                    ret, frame = self.cap.read()

                    if ret and frame is not None and frame.size > 0:
                        self.consecutive_failures = 0
                        return True, frame

                    self.logger.warning(
                        f"Failed to read frame (attempt {attempt + 1}/3)"
                    )
                    time.sleep(0.1)  # Short delay between attempts

                # If we get here, all attempts failed
                self.consecutive_failures += 1
                self.logger.error(
                    f"Failed to read frame after 3 attempts (consecutive failures: {self.consecutive_failures})"
                )

                if self.consecutive_failures >= self.max_failures:
                    self.logger.error(
                        "Too many consecutive failures, attempting recovery..."
                    )
                    if self._try_reinitialize():
                        self.consecutive_failures = 0
                        return self.get_frame()  # Try again after reinit
                    else:
                        self.logger.error("Camera recovery failed")
                return False, None

        except Exception as e:
            self.logger.error(f"Error getting frame: {str(e)}")
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_failures:
                self._try_reinitialize()
            return False, None

    def release(self):
        """Release the camera."""
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception as e:
                    self.logger.warning(f"Error releasing camera: {str(e)}")
                finally:
                    self.cap = None
                    self.consecutive_failures = 0

    def is_available(self):
        """Check if the camera is available and working."""
        return self.cap is not None and self.cap.isOpened()

    def get_status(self):
        """Get detailed camera status."""
        return {
            "available": self.is_available(),
            "resolution": self.get_current_resolution(),
            "failures": self.consecutive_failures,
            "selected_camera": self.selected_camera_index,
        }

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
