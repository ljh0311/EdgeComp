"""
Camera Manager Module
===================
Manages multiple camera instances for the Baby Monitor System.
"""

import logging
import threading
from typing import Dict, List, Optional, Union
import cv2
import json
from .camera import Camera

class CameraManager:
    def __init__(self):
        """Initialize the camera manager."""
        self.logger = logging.getLogger(__name__)
        self.cameras: Dict[str, Camera] = {}
        self.lock = threading.Lock()
        self._discover_cameras()

    def _discover_cameras(self) -> None:
        """Discover available cameras and initialize default camera if found."""
        temp_camera = Camera()
        available = temp_camera.get_available_cameras()
        if available:
            # Initialize default camera
            self.add_camera("Default Camera", source=0)

    def add_camera(self, name: str, source: Union[int, str], width: int = 640, height: int = 480) -> bool:
        """
        Add a new camera to the manager.
        
        Args:
            name: Unique name for the camera
            source: Camera source (index for USB cameras, URL for IP cameras)
            width: Camera resolution width
            height: Camera resolution height
            
        Returns:
            bool: True if camera was added successfully
        """
        with self.lock:
            if name in self.cameras:
                self.logger.error(f"Camera with name '{name}' already exists")
                return False

            try:
                camera = Camera(width=width, height=height)
                if isinstance(source, str):  # IP camera
                    camera.cap = cv2.VideoCapture(source)
                else:  # USB camera
                    camera.select_camera(source)
                    camera.initialize()

                if not camera.cap or not camera.cap.isOpened():
                    raise Exception("Failed to initialize camera")

                self.cameras[name] = camera
                self.logger.info(f"Added camera '{name}' successfully")
                return True

            except Exception as e:
                self.logger.error(f"Error adding camera '{name}': {str(e)}")
                return False

    def remove_camera(self, name: str) -> bool:
        """
        Remove a camera from the manager.
        
        Args:
            name: Name of the camera to remove
            
        Returns:
            bool: True if camera was removed successfully
        """
        with self.lock:
            if name not in self.cameras:
                self.logger.error(f"Camera '{name}' not found")
                return False

            try:
                camera = self.cameras[name]
                camera.release()
                del self.cameras[name]
                self.logger.info(f"Removed camera '{name}' successfully")
                return True

            except Exception as e:
                self.logger.error(f"Error removing camera '{name}': {str(e)}")
                return False

    def restart_camera(self, name: str) -> bool:
        """
        Restart a camera.
        
        Args:
            name: Name of the camera to restart
            
        Returns:
            bool: True if camera was restarted successfully
        """
        with self.lock:
            if name not in self.cameras:
                self.logger.error(f"Camera '{name}' not found")
                return False

            try:
                camera = self.cameras[name]
                camera.release()
                success = camera.initialize()
                if success:
                    self.logger.info(f"Restarted camera '{name}' successfully")
                else:
                    self.logger.error(f"Failed to restart camera '{name}'")
                return success

            except Exception as e:
                self.logger.error(f"Error restarting camera '{name}': {str(e)}")
                return False

    def get_camera(self, name: str) -> Optional[Camera]:
        """Get a camera instance by name."""
        return self.cameras.get(name)

    def get_camera_list(self) -> List[Dict]:
        """
        Get list of all managed cameras.
        
        Returns:
            List of dictionaries containing camera information
        """
        camera_list = []
        for name, camera in self.cameras.items():
            try:
                status = "Active" if camera.cap and camera.cap.isOpened() else "Error"
                camera_list.append({
                    "name": name,
                    "status": status,
                    "resolution": f"{camera.width}x{camera.height}"
                })
            except Exception as e:
                self.logger.error(f"Error getting info for camera '{name}': {str(e)}")
                camera_list.append({
                    "name": name,
                    "status": "Error",
                    "resolution": "Unknown"
                })
        return camera_list

    def cleanup(self) -> None:
        """Release all cameras and cleanup resources."""
        with self.lock:
            for name, camera in self.cameras.items():
                try:
                    camera.release()
                except Exception as e:
                    self.logger.error(f"Error releasing camera '{name}': {str(e)}")
            self.cameras.clear() 