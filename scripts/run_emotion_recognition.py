"""
Run Emotion Recognition
====================
Script to run sound-based emotion recognition for baby monitoring.
"""

import os
import sys
import signal
import argparse
import logging
import time
from pathlib import Path

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print('\nStopping emotion recognition...')
    if 'audio_stream' in globals():
        audio_stream.stop()
    sys.exit(0)

def main():
    """Run the emotion recognition system."""
    parser = argparse.ArgumentParser(description='Sound-based emotion recognition for baby monitoring')
    parser.add_argument('--model', type=str,
                      help='Path to emotion recognition model')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda'],
                      help='Device to run inference on')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Detection confidence threshold')
    parser.add_argument('--input-device', type=int,
                      help='Audio input device index')
    args = parser.parse_args()

    try:
        # Import after path setup
        import sounddevice as sd
        import numpy as np
        from babymonitor.detectors import EmotionDetector
        
        # Create emotion detector
        detector = EmotionDetector(
            model_path=args.model,
            threshold=args.threshold,
            device=args.device
        )
        
        # Audio callback
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            
            # Process audio chunk
            result = detector.process_audio(indata.copy())
            
            # Print results
            if result['emotion'] not in ['buffering', 'unknown', 'error']:
                emotion = result['emotion']
                confidence = result['confidence']
                print(f"\rEmotion: {emotion} ({confidence:.2f})", end='')
                
                # Print alert for crying
                if emotion == 'crying' and confidence > args.threshold:
                    print("\n[ALERT] Baby is crying!")
        
        # Set up audio stream
        with sd.InputStream(
            channels=1,
            samplerate=detector.SAMPLE_RATE,
            blocksize=detector.CHUNK_SIZE,
            device=args.input_device,
            callback=audio_callback
        ) as audio_stream:
            print("\nStarting emotion recognition...")
            print("Press Ctrl+C to stop")
            
            # Keep the main thread alive
            signal.signal(signal.SIGINT, signal_handler)
            signal.pause()
            
    except KeyboardInterrupt:
        print("\nStopping emotion recognition...")
    except Exception as e:
        logger.error(f"Error: {e}")
        if 'audio_stream' in locals():
            audio_stream.stop()
        sys.exit(1)
    finally:
        if 'detector' in locals():
            detector.cleanup()

if __name__ == '__main__':
    main() 