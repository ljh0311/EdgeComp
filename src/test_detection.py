#!/usr/bin/env python3
"""
Test script for person detection with debug interface.
Run with --dev flag to enable debug information display.
"""

import cv2
import argparse
import time
import numpy as np
from pathlib import Path
import sys
import os

# Add the src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from flask import Flask, Response, render_template_string
import threading
from queue import Queue
import logging
from babymonitor.detectors.person_tracker import PersonTracker

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
frame_queue = Queue(maxsize=2)
debug_info = {}
stop_signal = False

# HTML template for debug interface
DEBUG_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Detection Debug Interface</title>
    <style>
        body { 
            background: #1e1e1e; 
            color: #fff; 
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        .container {
            display: flex;
            gap: 20px;
        }
        .video-feed {
            flex: 1;
        }
        .debug-panel {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            min-width: 300px;
        }
        .debug-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .debug-table th, .debug-table td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #444;
        }
        .debug-table th {
            color: #00ff00;
        }
        h2 { color: #00ff00; }
        .metric-value { font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <div class="video-feed">
            <h2>Detection Feed</h2>
            <img src="{{ url_for('video_feed') }}" width="100%">
        </div>
        {% if debug %}
        <div class="debug-panel">
            <h2>Debug Information</h2>
            <table class="debug-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>FPS</td>
                    <td class="metric-value" id="fps">-</td>
                </tr>
                <tr>
                    <td>Frame Time (ms)</td>
                    <td class="metric-value" id="frame_time">-</td>
                </tr>
                <tr>
                    <td>Detection Confidence</td>
                    <td class="metric-value" id="confidence">-</td>
                </tr>
                <tr>
                    <td>Device</td>
                    <td class="metric-value" id="device">-</td>
                </tr>
                <tr>
                    <td>Resolution</td>
                    <td class="metric-value" id="resolution">-</td>
                </tr>
            </table>
        </div>
        {% endif %}
    </div>
    {% if debug %}
    <script>
        function updateDebugInfo() {
            fetch('/debug_info')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('frame_time').textContent = data.frame_time.toFixed(1);
                    document.getElementById('confidence').textContent = data.confidence.toFixed(3);
                    document.getElementById('device').textContent = data.device;
                    document.getElementById('resolution').textContent = data.resolution;
                });
        }
        setInterval(updateDebugInfo, 1000);
    </script>
    {% endif %}
</body>
</html>
"""

def process_frames(cap, person_tracker, debug=False):
    """Process frames from camera and update debug information."""
    global stop_signal, debug_info
    
    while not stop_signal:
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            # Process frame with person tracking (includes motion and fall detection)
            result = person_tracker.process_frame(frame)
            processed_frame = result['frame']
            
            # Update debug information
            if debug:
                debug_info.update({
                    'fps': person_tracker.detector.get_fps(),
                    'frame_time': person_tracker.detector.get_processing_time(),
                    'confidence': person_tracker.detector.model.conf if person_tracker.detector.model else 0,
                    'device': person_tracker.detector.device,
                    'resolution': f"{frame.shape[1]}x{frame.shape[0]}"
                })
            
            # Put processed frame in queue
            if not frame_queue.full():
                frame_queue.put(processed_frame)
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            if not frame_queue.full():
                frame_queue.put(frame)  # Show original frame on error

@app.route('/')
def index():
    """Render the main page."""
    return render_template_string(DEBUG_TEMPLATE, debug=app.config['DEBUG_MODE'])

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    def generate():
        while not stop_signal:
            if not frame_queue.empty():
                frame = frame_queue.get()
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/debug_info')
def debug_info_route():
    """Return debug information as JSON."""
    return debug_info

def main():
    parser = argparse.ArgumentParser(description='Person Detection Debug Interface')
    parser.add_argument('--dev', action='store_true', help='Enable debug interface')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    args = parser.parse_args()
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Error: Could not open camera")
        return
    
    # Initialize person tracker
    person_tracker = PersonTracker()
    
    # Set debug mode in Flask app
    app.config['DEBUG_MODE'] = args.dev
    
    # Start frame processing thread
    process_thread = threading.Thread(target=process_frames, 
                                    args=(cap, person_tracker, args.dev))
    process_thread.daemon = True
    process_thread.start()
    
    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Cleanup
        global stop_signal
        stop_signal = True
        process_thread.join()
        cap.release()

if __name__ == '__main__':
    main() 