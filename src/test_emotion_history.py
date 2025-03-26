#!/usr/bin/env python
"""
Test Emotion History Module
==========================
This script tests the emotion history logging functionality of the EmotionDetector.
"""

import os
import sys
import time
import random
import json
import logging
from pathlib import Path

# Add the project root to system path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the emotion history test."""
    try:
        # Import the emotion detector
        from src.babymonitor.detectors.emotion_detector import EmotionDetector
        
        logger.info("Testing Emotion History Functionality")
        
        # Create an emotion detector instance
        emotion_detector = EmotionDetector()
        
        # Display current model info
        logger.info(f"Current model: {emotion_detector.model_id} - {emotion_detector.model_info['name']}")
        logger.info(f"Supported emotions: {emotion_detector.emotions}")
        
        # Test initial state
        history_data = emotion_detector.get_emotion_history('1h')
        logger.info(f"Initial history data: {json.dumps(history_data, indent=2)}")
        
        # Simulate processing audio for some time
        logger.info("Simulating audio processing...")
        
        # Create a dummy numpy array for testing
        import numpy as np
        dummy_audio = np.random.rand(4000).astype(np.float32)
        
        # Process audio multiple times to generate history
        for i in range(20):
            result = emotion_detector.process_audio(dummy_audio)
            if result['emotion'] != 'buffering':
                logger.info(f"Processed audio: {result['emotion']} ({result['confidence']:.2f})")
            time.sleep(0.5)
        
        # Get updated history
        logger.info("Getting updated history data...")
        history_data = emotion_detector.get_emotion_history('1h')
        logger.info(f"Updated history data: {json.dumps(history_data, indent=2)}")
        
        # Switch to a different model
        logger.info("Switching to 'emotion2' model...")
        switch_result = emotion_detector.switch_model('emotion2')
        logger.info(f"Switch result: {json.dumps(switch_result, indent=2)}")
        logger.info(f"New supported emotions: {emotion_detector.emotions}")
        
        # Process audio with new model
        logger.info("Processing audio with new model...")
        for i in range(20):
            result = emotion_detector.process_audio(dummy_audio)
            if result['emotion'] != 'buffering':
                logger.info(f"Processed audio: {result['emotion']} ({result['confidence']:.2f})")
            time.sleep(0.5)
        
        # Get history for new model
        logger.info("Getting history data for new model...")
        history_data = emotion_detector.get_emotion_history('1h')
        logger.info(f"New model history data: {json.dumps(history_data, indent=2)}")
        
        # Test history saving
        logger.info("Testing history saving...")
        emotion_detector._save_history()
        
        # Get file path where history was saved
        history_file = emotion_detector.history_file
        logger.info(f"History saved to: {history_file}")
        
        # Display file contents if it exists
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                saved_data = json.load(f)
            logger.info(f"Saved history data: {json.dumps(saved_data, indent=2)}")
        
        # Clean up
        emotion_detector.cleanup()
        logger.info("Test completed successfully")
        
        return 0
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 