"""
Camera Module
============
Handles camera initialization and frame capture for different platforms.
"""

import cv2
import platform
import logging

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
    
    def initialize(self):
        """Initialize the camera based on platform."""
        try:
            if self.is_raspberry_pi:
                from picamera2 import Picamera2
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (self.width, self.height)},
                    lores={"size": (640, 480)},
                    display="lores"
                )
                self.camera.configure(config)
                self.camera.start()
            else:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                # Try to enable camera properties dialog
                try:
                    self.camera.set(cv2.CAP_PROP_SETTINGS, 1)
                except:
                    self.logger.warning("Camera settings dialog not available")
            
            self.logger.info("Camera initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {str(e)}")
            return False
    
    def get_frame(self):
        """
        Get a frame from the camera.
        
        Returns:
            tuple: (success, frame)
        """
        if self.camera is None:
            return False, None
            
        try:
            if self.is_raspberry_pi:
                return True, self.camera.capture_array()
            else:
                return self.camera.read()
        except Exception as e:
            self.logger.error(f"Error capturing frame: {str(e)}")
            return False, None
    
    def release(self):
        """Release the camera resources."""
        if self.camera is not None:
            try:
                if self.is_raspberry_pi:
                    self.camera.stop()
                else:
                    self.camera.release()
                self.logger.info("Camera released successfully")
            except Exception as e:
                self.logger.error(f"Error releasing camera: {str(e)}")
        self.camera = None 