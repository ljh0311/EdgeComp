"""
Camera Wrapper Module
====================
Provides a unified interface for camera access, wrapping the actual Camera implementation.
"""

import logging
import cv2
import time

logger = logging.getLogger('CameraWrapper')

class Camera:
    """
    Camera wrapper class that provides a unified interface for camera access.
    This wrapper ensures compatibility with different camera implementations.
    """
    def __init__(self, camera_id=0, width=640, height=480, fps=30, max_retries=3):
        """
        Initialize the camera wrapper
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            fps: Frames per second
            max_retries: Maximum number of retries for initialization
        """
        logger.info(f"Initializing camera with ID {camera_id}")
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.last_frame = None
        self.last_frame_time = 0
        self.frame_timeout = 1.0  # 1 second timeout for frames
        
        # Try to initialize the camera with retries
        for attempt in range(max_retries):
            try:
                # Initialize camera
                self.cap = cv2.VideoCapture(camera_id)
                
                if not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open camera (ID: {camera_id})")
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                
                # Verify settings were applied
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                
                logger.info(f"Camera initialized (ID: {camera_id}, Resolution: {actual_width}x{actual_height}, FPS: {actual_fps})")
                
                # Read test frame to verify camera is working
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise RuntimeError("Failed to capture test frame")
                
                self.last_frame = frame
                self.last_frame_time = time.time()
                return
                
            except Exception as e:
                logger.warning(f"Camera initialization attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retrying
        
        logger.error(f"Failed to initialize camera after {max_retries} attempts")
        raise RuntimeError(f"Failed to initialize camera (ID: {camera_id})")
    
    def read(self):
        """
        Read a frame from the camera
        
        Returns:
            tuple: (ret, frame) where ret is True if the frame was read successfully
        """
        if not self.is_opened():
            return False, self.last_frame
        
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.last_frame = frame
                self.last_frame_time = time.time()
                return True, frame
            else:
                # If frame capture fails but we have a recent frame, return it
                if self.last_frame is not None and time.time() - self.last_frame_time < self.frame_timeout:
                    logger.warning("Failed to capture frame, using last known good frame")
                    return True, self.last_frame
                logger.warning("Failed to capture frame")
                return False, self.last_frame
        except Exception as e:
            logger.error(f"Error capturing frame: {str(e)}")
            return False, self.last_frame
    
    def release(self):
        """
        Release the camera
        """
        if self.cap is not None and self.is_opened():
            try:
                self.cap.release()
                logger.info("Camera released")
            except Exception as e:
                logger.error(f"Error releasing camera: {str(e)}")
            self.cap = None
    
    def is_opened(self):
        """
        Check if the camera is opened
        
        Returns:
            bool: True if the camera is opened, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()
    
    def get_resolution(self):
        """
        Get the camera resolution
        
        Returns:
            tuple: (width, height)
        """
        if self.is_opened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (self.width, self.height)
    
    def get_fps(self):
        """
        Get the camera FPS
        
        Returns:
            float: Frames per second
        """
        if self.is_opened():
            return self.cap.get(cv2.CAP_PROP_FPS)
        return self.fps 