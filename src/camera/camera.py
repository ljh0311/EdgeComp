"""
Camera module for video capture and processing.
"""

import cv2
import logging
import numpy as np

class Camera:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.cap = None
        self.logger = logging.getLogger(__name__)

    def initialize(self):
        """Initialize the camera."""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return self.cap.isOpened()
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {str(e)}")
            return False

    def get_frame(self):
        """Get a frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None 