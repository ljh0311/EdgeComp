"""
Main Program
-----------
Entry point for the Baby Monitor System.
"""

import logging
import os
from babymonitor.web.app import BabyMonitorWeb
from babymonitor.detectors.person_detector import PersonDetector
from babymonitor.detectors.detector_factory import DetectorFactory, DetectorType
from babymonitor.audio.sound_detector import SoundDetector
from babymonitor.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the Baby Monitor System."""
    try:
        # Initialize components
        logger.info("Initializing Baby Monitor System...")

        # Initialize web interface
        web_interface = BabyMonitorWeb(host=Config.WEB_HOST, port=Config.WEB_PORT)

        # Initialize detectors based on configuration
        if Config.DETECTOR_TYPE.lower() == "lightweight":
            logger.info("Using lightweight detector")
            person_detector = DetectorFactory.create_detector(
                DetectorType.LIGHTWEIGHT.value, config=Config.LIGHTWEIGHT_DETECTION
            )
        else:
            logger.info("Using Haar Cascade detector")
            person_detector = PersonDetector(
                threshold=Config.PERSON_DETECTION.get("threshold", 0.7),
                target_size=(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT),
            )

        sound_detector = SoundDetector()

        # Start web interface
        logger.info("Starting web interface...")
        web_interface.start()

    except Exception as e:
        logger.error(f"Error in main program: {e}")
        raise


if __name__ == "__main__":
    main()
