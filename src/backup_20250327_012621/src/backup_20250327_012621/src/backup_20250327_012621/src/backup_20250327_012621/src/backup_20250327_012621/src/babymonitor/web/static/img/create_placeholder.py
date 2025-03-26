"""
Create a simple placeholder image for when the camera is not available
"""

import cv2
import numpy as np
import os

# Create a blank image (640x480, black background)
img = np.zeros((480, 640, 3), dtype=np.uint8)

# Draw a border rectangle
cv2.rectangle(img, (20, 20), (620, 460), (50, 50, 50), 2)

# Add text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'NO CAMERA SIGNAL', (180, 200), font, 1, (255, 255, 255), 2)
cv2.putText(img, 'Camera disconnected or unavailable', (150, 250), font, 0.7, (200, 200, 200), 1)

# Save the image
output_path = os.path.join(os.path.dirname(__file__), 'no_signal.jpg')
cv2.imwrite(output_path, img)

print(f"Placeholder image created at: {output_path}") 