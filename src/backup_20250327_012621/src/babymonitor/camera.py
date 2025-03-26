"""
Camera module for Baby Monitor System
"""

import cv2
import logging

logger = logging.getLogger('Camera')

class Camera:
    """
    Camera class for capturing video frames
    """
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        """
        Initialize the camera
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        if self.is_opened():
            logger.info(f"Camera initialized (ID: {camera_id}, Resolution: {width}x{height}, FPS: {fps})")
        else:
            logger.error(f"Failed to open camera (ID: {camera_id})")
    
    def is_opened(self):
        """
        Check if the camera is opened
        
        Returns:
            bool: True if the camera is opened, False otherwise
        """
        return self.cap.isOpened()
    
    def read(self):
        """
        Read a frame from the camera
        
        Returns:
            tuple: (ret, frame) where ret is True if the frame was read successfully
        """
        if not self.is_opened():
            return False, None
        
        return self.cap.read()
    
    def release(self):
        """
        Release the camera
        """
        if self.is_opened():
            self.cap.release()
            logger.info("Camera released")
    
    def get_resolution(self):
        """
        Get the camera resolution
        
        Returns:
            tuple: (width, height)
        """
        return (self.width, self.height)
    
    def get_fps(self):
        """
        Get the camera FPS
        
        Returns:
            float: Frames per second
        """
        return self.fps 