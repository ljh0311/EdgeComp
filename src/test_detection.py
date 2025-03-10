#!/usr/bin/env python3
"""
Test script for person and motion detection with debug interface.
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
from babymonitor.detectors.person_detector import PersonDetector
from babymonitor.detectors.motion_detector import MotionDetector

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

def process_frames(cap, person_detector, motion_detector, debug=False):
    """Process frames from camera and update debug information."""
    global stop_signal, debug_info
    
    while not stop_signal:
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            # Store original frame
            original_frame = frame.copy()
            
            # Get person detections first
            detections = person_detector.detect(frame)
            
            # Process frame with person detection
            processed_frame = person_detector.process_frame(frame)
            
            # Add motion detection
            if motion_detector:
                # Process motion detection
                motion_frame, motion_detected, fall_detected = motion_detector.detect(original_frame, detections)
                
                # Combine motion detection with person detection frame
                if motion_detected or fall_detected:
                    # Add semi-transparent motion overlay
                    alpha = 0.3
                    processed_frame = cv2.addWeighted(processed_frame, 1 - alpha, motion_frame, alpha, 0)
                    
                    # Add motion and fall indicators
                    if motion_detected:
                        cv2.putText(processed_frame, "Motion Detected", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if fall_detected:
                        cv2.putText(processed_frame, "Fall Detected!", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw person detections with improved visibility
            if detections and len(detections) > 0:
                for r in detections:
                    boxes = r.boxes
                    for box in boxes:
                        if box.cls == 0:  # Person class
                            # Get box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            conf = float(box.conf[0].cpu().numpy())
                            
                            # Draw thick black outline
                            cv2.rectangle(processed_frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 0), 4)
                            # Draw green box
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add confidence label with better visibility
                            label = f"Person {conf:.2f}"
                            font_scale = 0.7
                            thickness = 2
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            
                            # Get text size for background
                            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                            
                            # Draw black background for text
                            cv2.rectangle(processed_frame, 
                                        (x1-2, y1-text_h-10),
                                        (x1+text_w+4, y1),
                                        (0, 0, 0), -1)
                            
                            # Draw white text
                            cv2.putText(processed_frame, label,
                                      (x1, y1-5),
                                      font,
                                      font_scale,
                                      (255, 255, 255),
                                      thickness)
            
            # Update debug information
            if debug:
                debug_info.update({
                    'fps': person_detector.get_fps(),
                    'frame_time': person_detector.get_processing_time(),
                    'confidence': person_detector.model.conf if person_detector.model else 0,
                    'device': person_detector.device,
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
def get_debug_info():
    """Return current debug information."""
    return debug_info

def main():
    parser = argparse.ArgumentParser(description='Test person and motion detection')
    parser.add_argument('--dev', action='store_true', help='Enable debug interface')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--port', type=int, default=5000, help='Web interface port')
    parser.add_argument('--motion-threshold', type=int, default=25, help='Motion detection threshold (10-50)')
    parser.add_argument('--fall-threshold', type=float, default=1.5, help='Fall detection aspect ratio threshold')
    args = parser.parse_args()
    
    # Initialize detectors
    try:
        person_detector = PersonDetector()
        motion_detector = MotionDetector({
            'MOTION_THRESHOLD': args.motion_threshold,
            'HISTORY': 300,  # Shorter history for faster response
            'VAR_THRESHOLD': 20,  # Slightly higher variance threshold
            'FALL_ASPECT_RATIO': args.fall_threshold,
            'MIN_AREA': 500,  # Minimum area for motion detection
            'MAX_AREA': 50000,  # Maximum area for motion detection
            'BLUR_SIZE': 5,  # Blur size for noise reduction
            'DILATE_ITERATIONS': 2  # Number of dilations for motion mask
        })
    except Exception as e:
        logger.error(f"Error initializing detectors: {e}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Error: Could not open camera")
        return
    
    # Set camera resolution and parameters
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Try to get 30 FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    
    logger.info(f"Camera initialized at {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    logger.info(f"Running with motion threshold: {args.motion_threshold}, fall threshold: {args.fall_threshold}")
    
    # Start frame processing thread
    app.config['DEBUG_MODE'] = args.dev
    processing_thread = threading.Thread(
        target=process_frames,
        args=(cap, person_detector, motion_detector, args.dev)
    )
    processing_thread.start()
    
    try:
        # Start Flask app
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Stopping application...")
    finally:
        # Cleanup
        global stop_signal
        stop_signal = True
        processing_thread.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 