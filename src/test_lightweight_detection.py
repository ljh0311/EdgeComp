#!/usr/bin/env python3
"""
Test script for lightweight person detection.
This script demonstrates the lightweight TensorFlow Lite-based detection system.
"""

import cv2
import argparse
import time
import numpy as np
import os
import sys
import platform
import logging
from pathlib import Path
from flask import Flask, Response, render_template_string

# Add the src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

# Import the lightweight detector
from babymonitor.detectors.lightweight_detector import VideoStream, LightweightDetector

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app for web interface
app = Flask(__name__)

# Global variables for sharing data between threads
frame_buffer = None
detection_results = None
debug_info = {
    'fps': 0,
    'detections': 0,
    'resolution': '',
    'model': ''
}

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Lightweight Person Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .video-container {
            margin: 20px 0;
            border: 1px solid #ddd;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            background-color: #fff;
            padding: 10px;
            border-radius: 5px;
        }
        .debug-container {
            width: 640px;
            margin: 20px 0;
            padding: 15px;
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .debug-item {
            margin: 10px 0;
            padding: 5px;
            border-bottom: 1px solid #eee;
        }
        .debug-label {
            font-weight: bold;
            display: inline-block;
            width: 150px;
        }
    </style>
    <script>
        // Auto-refresh debug info every second
        function refreshDebugInfo() {
            fetch('/debug_info')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(2);
                    document.getElementById('detections').textContent = data.detections;
                    document.getElementById('resolution').textContent = data.resolution;
                    document.getElementById('model').textContent = data.model;
                });
        }
        
        // Set up auto-refresh when page loads
        window.onload = function() {
            setInterval(refreshDebugInfo, 1000);
        };
    </script>
</head>
<body>
    <h1>Lightweight Person Detection</h1>
    <div class="container">
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </div>
        <div class="debug-container">
            <h2>Debug Information</h2>
            <div class="debug-item">
                <span class="debug-label">FPS:</span>
                <span id="fps">{{ debug_info.fps }}</span>
            </div>
            <div class="debug-item">
                <span class="debug-label">Detections:</span>
                <span id="detections">{{ debug_info.detections }}</span>
            </div>
            <div class="debug-item">
                <span class="debug-label">Resolution:</span>
                <span id="resolution">{{ debug_info.resolution }}</span>
            </div>
            <div class="debug-item">
                <span class="debug-label">Model:</span>
                <span id="model">{{ debug_info.model }}</span>
            </div>
        </div>
    </div>
</body>
</html>
"""

def process_frames(video_stream, detector, debug=False):
    """Process frames from the video stream using the detector."""
    global frame_buffer, detection_results, debug_info
    
    while True:
        # Get frame from video stream
        frame = video_stream.read()
        if frame is None:
            logger.error("Failed to get frame from camera")
            time.sleep(0.1)
            continue
            
        # Process the frame with the detector
        results = detector.process_frame(frame)
        
        # Update global variables
        frame_buffer = results['frame']
        detection_results = results['detections']
        
        # Update debug info
        debug_info['fps'] = results['fps']
        debug_info['detections'] = len(detection_results)
        debug_info['resolution'] = f"{frame.shape[1]}x{frame.shape[0]}"
        debug_info['model'] = os.path.basename(detector.model_path)
        
        # If not in web mode, display the frame
        if not debug:
            cv2.imshow('Lightweight Person Detection', frame_buffer)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    # Clean up
    cv2.destroyAllWindows()

@app.route('/')
def index():
    """Render the web interface."""
    return render_template_string(HTML_TEMPLATE, debug_info=debug_info)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    def generate():
        global frame_buffer
        while True:
            if frame_buffer is not None:
                # Encode the frame as JPEG
                _, jpeg = cv2.imencode('.jpg', frame_buffer)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                time.sleep(0.1)
                
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/debug_info')
def debug_info_route():
    """Return debug information as JSON."""
    global debug_info
    return debug_info

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lightweight Person Detection')
    parser.add_argument('--model', type=str, default='models/person_detection_model.tflite',
                        help='Path to TFLite model file')
    parser.add_argument('--labels', type=str, default='models/person_labels.txt',
                        help='Path to label file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--resolution', type=str, default='640x480',
                        help='Camera resolution (WxH)')
    parser.add_argument('--web', action='store_true',
                        help='Enable web interface')
    parser.add_argument('--display', action='store_true',
                        help='Display the detection results')
    parser.add_argument('--port', type=int, default=5000,
                        help='Web server port')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    resolution = (width, height)
    
    # Check if model file exists
    if not os.path.exists(args.model):
        # Try to find the model in the models directory
        model_dir = os.path.join(os.path.dirname(src_dir), 'models')
        model_path = os.path.join(model_dir, os.path.basename(args.model))
        if os.path.exists(model_path):
            args.model = model_path
            logger.info(f"Using model from models directory: {args.model}")
        else:
            logger.warning(f"Model file not found: {args.model}")
            logger.warning("Using the pet detection model from BirdRepeller as fallback")
            args.model = os.path.join(os.path.dirname(src_dir), 'BirdRepeller', 'models', 'pet_detection_model.tflite')
            args.labels = os.path.join(os.path.dirname(src_dir), 'BirdRepeller', 'models', 'pet_labels.txt')
    
    # Initialize video stream
    logger.info(f"Starting video stream with resolution {resolution}")
    video_stream = VideoStream(resolution=resolution).start()
    time.sleep(1.0)  # Allow camera to warm up
    
    # Initialize detector
    logger.info(f"Initializing detector with model {args.model}")
    detector = LightweightDetector(
        model_path=args.model,
        label_path=args.labels,
        threshold=args.threshold,
        resolution=resolution
    )
    
    # Start processing frames in a separate thread
    import threading
    processing_thread = threading.Thread(
        target=process_frames,
        args=(video_stream, detector, args.web or not args.display)
    )
    processing_thread.daemon = True
    processing_thread.start()
    
    try:
        if args.web:
            # Start web server
            logger.info(f"Starting web server on port {args.port}")
            app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
        elif args.display:
            # Display the detection results directly
            logger.info("Displaying detection results")
            while True:
                if frame_buffer is not None:
                    cv2.imshow('Lightweight Person Detection', frame_buffer)
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.1)
        else:
            # Wait for processing thread to finish
            processing_thread.join()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Clean up
        logger.info("Cleaning up")
        video_stream.stop()
        detector.cleanup()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    main() 