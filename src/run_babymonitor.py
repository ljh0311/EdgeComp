"""
Run Baby Monitor
==============
Script to set up models and start the Baby Monitor System.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Set up models and start the Baby Monitor System."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Baby Monitor System")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only set up models, do not start the system",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["yolov8", "lightweight", "tracker", "motion"],
        help="Detector type to use (overrides config)",
    )
    parser.add_argument(
        "--force-cpu", action="store_true", help="Force CPU usage for models"
    )
    parser.add_argument(
        "--disable-audio", action="store_true", help="Disable audio processing"
    )
    args = parser.parse_args()

    # Add src directory to path if needed
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Set environment variables based on arguments
    if args.force_cpu:
        os.environ["FORCE_CPU"] = "1"

    if args.disable_audio:
        os.environ["DISABLE_AUDIO"] = "1"

    if args.detector:
        os.environ["DETECTOR_TYPE"] = args.detector

    # Import after setting environment variables
    try:
        from babymonitor.utils.setup_models import setup_models

        # Set up models
        logger.info("Setting up models...")
        setup_models()

        if not args.setup_only:
            # Start the system
            logger.info("Starting Baby Monitor System...")
            from babymonitor.main import main as start_system

            start_system()
        else:
            logger.info(
                "Model setup complete. Use 'python -m babymonitor.main' to start the system."
            )

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(
            "Make sure you are running this script from the project root directory."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
