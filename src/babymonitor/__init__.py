"""
Baby Monitor Package
==================
Main package for the Baby Monitor application.
"""

# Version information
__version__ = "1.0.0"

# Define what should be imported with "from babymonitor import *"
__all__ = [
    "BabyMonitorSystem",
    "Camera",
    "AudioProcessor",
    "EmotionRecognizer",
    "PersonDetector",
    "BabyMonitorWeb",
]

# Note: We're not importing modules directly here to avoid circular imports
# Users should import specific modules as needed
