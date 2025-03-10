"""
Demo script for real-time emotion recognition
------------------------------------------
This script demonstrates the real-time emotion recognition system
optimized for Raspberry Pi 400.
"""

import os
import sys
import signal
import argparse
from pathlib import Path
from realtime_emotion import create_recognizer

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\nStopping emotion recognition...')
    if 'recognizer' in globals():
        recognizer.stop()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Real-time emotion recognition from audio')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the emotion recognition model')
    parser.add_argument('--device', type=int, default=None,
                      help='Audio device index (default: system default)')
    args = parser.parse_args()

    # Verify model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Create and start the recognizer
    try:
        print("Initializing emotion recognition system...")
        recognizer = create_recognizer(str(model_path))
        
        print("\nStarting real-time emotion recognition...")
        print("Press Ctrl+C to stop")
        recognizer.start()
        
        # Keep the main thread alive
        signal.signal(signal.SIGINT, signal_handler)
        signal.pause()
        
    except Exception as e:
        print(f"Error: {e}")
        if 'recognizer' in locals():
            recognizer.stop()
        sys.exit(1)

if __name__ == '__main__':
    main() 