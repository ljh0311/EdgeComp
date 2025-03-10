"""
Camera Module
============
Handles camera initialization and frame capture for the Baby Monitor System.
"""

import cv2
import numpy as np
import logging
import threading
import time
from queue import Queue, Empty
from typing import Tuple, List, Optional, Dict

class Camera:
    def __init__(self, width: int = 640, height: int = 480):
        """Initialize the camera with specified resolution."""
        self.logger = logging.getLogger(__name__)
        self.width = width
        self.height = height
        self.cap = None
        self.is_running = False
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.frame_queue = Queue(maxsize=2)  # Reduced queue size for lower latency
        self.frame_thread = None
        self.selected_camera_index = 0
        self.available_cameras = self._enumerate_cameras()

    def _enumerate_cameras(self) -> List[Dict]:
        """Enumerate all available cameras."""
        cameras = []
        for i in range(10):  # Check first 10 indices
            try:
                # Try to get camera name using DirectShow
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(i)  # Fallback to default
                
                if cap.isOpened():
                    # Try to get a more descriptive name
                    try:
                        # Set camera properties to get name (Windows only)
                        cap.set(cv2.CAP_PROP_SETTINGS, 1)
                        # Get camera name from backend if possible
                        backend_name = cap.getBackendName()
                        name = f"Camera {i} ({backend_name})"
                    except:
                        name = f"Camera {i}"
                    
                    # Get supported resolutions
                    resolutions = self._get_supported_resolutions(cap)
                    cameras.append({
                        'id': i,
                        'name': name,
                        'resolutions': resolutions
                    })
                    cap.release()
            except Exception as e:
                self.logger.debug(f"Error checking camera {i}: {str(e)}")
                continue
        return cameras

    def _get_supported_resolutions(self, cap) -> List[str]:
        """Get supported resolutions for a camera."""
        standard_resolutions = [
            (640, 480),    # VGA
            (800, 600),    # SVGA
            (1280, 720),   # HD
            (1920, 1080)   # Full HD
        ]
        
        supported = []
        try:
            # Get current resolution to restore later
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Add current resolution first
            current_res = f"{original_width}x{original_height}"
            if current_res not in supported:
                supported.append(current_res)
            
            # Check standard resolutions
            for width, height in standard_resolutions:
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    resolution = f"{actual_width}x{actual_height}"
                    if resolution not in supported:
                        supported.append(resolution)
                except Exception as e:
                    self.logger.debug(f"Resolution {width}x{height} not supported: {str(e)}")
            
            # Restore original resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)
            
        except Exception as e:
            self.logger.error(f"Error getting supported resolutions: {str(e)}")
            # Return at least the current resolution
            if current_res:
                supported = [current_res]
            
        return supported

    def initialize(self) -> bool:
        """Initialize the camera capture."""
        try:
            if self.cap is not None:
                self.cleanup()

            # Try DirectShow first (better for Windows)
            self.cap = cv2.VideoCapture(self.selected_camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default
                self.cap = cv2.VideoCapture(self.selected_camera_index)

            if not self.cap.isOpened():
                self.logger.error("Failed to open camera")
                return False

            # Set high-performance camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if actual_width != self.width or actual_height != self.height:
                self.logger.warning(f"Camera resolution set to {actual_width}x{actual_height} "
                                  f"(requested {self.width}x{self.height})")
                self.width = actual_width
                self.height = actual_height

            # Start frame capture thread
            self.is_running = True
            self.frame_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.frame_thread.start()

            self.logger.info(f"Camera initialized successfully at {self.width}x{self.height}")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing camera: {str(e)}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False

    def select_camera(self, camera_name: str) -> bool:
        """Select a camera by name."""
        try:
            for camera in self.available_cameras:
                if camera['name'] == camera_name:
                    self.selected_camera_index = camera['id']
                    return self.initialize()
            return False
        except Exception as e:
            self.logger.error(f"Error selecting camera: {str(e)}")
            return False

    def get_camera_list(self) -> List[str]:
        """Get list of available camera names."""
        return [cam['name'] for cam in self.available_cameras]

    def get_camera_resolutions(self, camera_name: str) -> List[str]:
        """Get available resolutions for the specified camera."""
        try:
            for camera in self.available_cameras:
                if camera['name'] == camera_name:
                    return camera['resolutions']
            return []
        except Exception as e:
            self.logger.error(f"Error getting camera resolutions: {str(e)}")
            return []

    def set_resolution(self, resolution: str) -> bool:
        """Set camera resolution from string format (e.g., '1280x720')."""
        try:
            width, height = map(int, resolution.split('x'))
            return self._set_resolution(width, height)
        except Exception as e:
            self.logger.error(f"Error parsing resolution string: {str(e)}")
            return False

    def _set_resolution(self, width: int, height: int) -> bool:
        """Internal method to set camera resolution."""
        try:
            if self.cap is None:
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Verify the change
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if actual_width == width and actual_height == height:
                self.width = width
                self.height = height
                self.logger.info(f"Resolution set to {width}x{height}")
                return True
            else:
                self.logger.warning(f"Failed to set resolution to {width}x{height}, "
                                  f"got {actual_width}x{actual_height}")
                return False

        except Exception as e:
            self.logger.error(f"Error setting resolution: {str(e)}")
            return False

    def _capture_frames(self):
        """Continuously capture frames in a separate thread."""
        last_frame_time = 0
        frame_interval = 1.0 / 60.0  # Increased to 60 FPS target

        while self.is_running and self.cap is not None:
            try:
                current_time = time.time()
                if current_time - last_frame_time >= frame_interval:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        # Store the latest frame with minimal copying
                        with self.frame_lock:
                            self.current_frame = frame  # Store direct reference
                            # Only update queue if someone is waiting for frames
                            if not self.frame_queue.full():
                                self.frame_queue.put(frame)  # No need to copy here
                        last_frame_time = current_time
                    else:
                        time.sleep(0.001)  # Minimal sleep on failure
                else:
                    time.sleep(0.0001)  # Reduced sleep time for higher responsiveness
            except Exception as e:
                self.logger.error(f"Error capturing frame: {str(e)}")
                time.sleep(0.001)  # Minimal sleep on error

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the latest frame from the camera."""
        try:
            if not self.is_running or self.cap is None:
                return False, None

            with self.frame_lock:
                if self.current_frame is not None:
                    # Only copy if the frame will be modified
                    return True, self.current_frame
                return False, None

        except Exception as e:
            self.logger.error(f"Error getting frame: {str(e)}")
            return False, None

    def get_current_resolution(self) -> str:
        """Get current camera resolution."""
        if self.cap is None:
            return "Not available"
        return f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"

    def cleanup(self):
        """Clean up camera resources."""
        self.is_running = False
        if self.frame_thread is not None:
            self.frame_thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.logger.info("Camera resources cleaned up") 