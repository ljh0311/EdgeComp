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
import platform
from threading import Thread, Lock
from queue import Queue
import logging
from flask import Flask, Response, render_template_string
from babymonitor.detectors.person_tracker import PersonTracker

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

class VideoStream:
    """Threaded camera stream optimized for both Raspberry Pi and Windows"""
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        self.resolution = resolution
        self.framerate = framerate
        
        # Platform specific optimizations
        if platform.system() == "Linux":
            # Raspberry Pi optimizations
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, framerate)
        
        self.frame = None
        self.grabbed = False
        self.stopped = False
        self.lock = Lock()
        
        # Motion detection parameters
        self.prev_frame = None
        self.motion_threshold = 25
        self.min_motion_area = 500
        
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
        
    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
                
            grabbed, frame = self.stream.read()
            if grabbed:
                with self.lock:
                    self.frame = frame
                    self.grabbed = grabbed
                    
                    # Update motion detection
                    if self.prev_frame is None:
                        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        continue
                        
                    # Motion detection
                    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_delta = cv2.absdiff(self.prev_frame, current_frame)
                    thresh = cv2.threshold(frame_delta, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    motion_detected = False
                    for contour in contours:
                        if cv2.contourArea(contour) > self.min_motion_area:
                            motion_detected = True
                            break
                            
                    self.motion_detected = motion_detected
                    self.prev_frame = current_frame
                    
    def read(self):
        with self.lock:
            return self.grabbed, self.frame, self.motion_detected
            
    def stop(self):
        self.stopped = True

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

def process_frames(video_stream, person_tracker, debug=False):
    """Process frames from the video stream"""
    global debug_info, stop_signal
    fps_start_time = time.time()
    fps = 0
    frame_count = 0
    
    while not stop_signal:
        grabbed, frame, motion_detected = video_stream.read()
        if not grabbed:
            continue
            
        frame_start_time = time.time()
        
        # Only run person detection if motion is detected (optimization)
        if motion_detected:
            # Run person detection
            detections = person_tracker.detect(frame)
            
            # Draw detections
            for detection in detections:
                bbox = detection['bbox']
                conf = detection['confidence']
                
                # Draw bounding box
                cv2.rectangle(frame, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            (0, 255, 0), 2)
                            
                # Draw confidence
                label = f"Person: {conf:.2f}"
                cv2.putText(frame, label, 
                          (int(bbox[0]), int(bbox[1] - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate FPS
        frame_count += 1
        if frame_count >= 30:  # Update FPS every 30 frames
            fps = frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            frame_count = 0
            
        # Update debug info
        if debug:
            frame_time = (time.time() - frame_start_time) * 1000
            debug_info.update({
                'fps': f"{fps:.1f}",
                'frame_time': f"{frame_time:.1f}",
                'motion_detected': str(motion_detected),
                'resolution': f"{frame.shape[1]}x{frame.shape[0]}"
            })
            
            # Draw debug info on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                      
        # Put frame in queue for streaming
        try:
            frame_queue.put_nowait(frame)
        except:
            pass
            
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true',
                      help='Enable debug interface')
    parser.add_argument('--resolution', default='640x480',
                      help='Camera resolution (WxH)')
    parser.add_argument('--fps', type=int, default=30,
                      help='Target FPS')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Initialize video stream
    video_stream = VideoStream(resolution=(width, height),
                            framerate=args.fps).start()
    
    # Initialize person tracker
    person_tracker = PersonTracker()
    
    try:
        # Start frame processing thread
        logger.info("Starting frame processing thread...")
        process_thread = Thread(target=process_frames,
                             args=(video_stream, person_tracker, args.dev))
        process_thread.daemon = True
        process_thread.start()
        
        # Start Flask server
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=8000, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("Stopping application...")
        
    finally:
        # Cleanup
        stop_signal = True
        video_stream.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 