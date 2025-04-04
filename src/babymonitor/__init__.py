"""
Baby Monitor System
===================

A comprehensive baby monitoring system with video streaming, emotion detection, 
and alert capabilities using both local and web interfaces.

Features:
- Real-time video monitoring
- Audio analysis
- Emotion detection
- MQTT communication protocol with HTTP/Socket.IO fallback
- Local and web-based UI options
"""

# Version information
__version__ = "2.1.0"  # Added MQTT support
__author__ = "Edge Computing Team"
__license__ = "MIT"

# Define what should be imported with "from babymonitor import *"
__all__ = [
    "BabyMonitorSystem",
    "Camera",
    "AudioProcessor",
    "EmotionRecognizer",
    "PersonDetector",
    "BabyMonitorWeb",
    "MQTTServer",  # Added MQTT server class
]

# Note: We're not importing modules directly here to avoid circular imports
# Users should import specific modules as needed
