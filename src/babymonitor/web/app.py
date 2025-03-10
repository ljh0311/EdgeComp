from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ..detectors.person_detector import PersonDetector
import logging
import time
import threading
import queue

app = Flask(__name__)
socketio = SocketIO(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize PersonDetector
detector = None
frame_lock = threading.Lock()
metrics_queue = queue.Queue()

def initialize_detector():
    global detector
    try:
        detector = PersonDetector()
        logger.info("PersonDetector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PersonDetector: {e}")
        raise

# Initialize detector
initialize_detector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection and send initial device info."""
    try:
        if detector is not None:
            device_info = detector.get_device_info()
            emit('device_info', device_info)
            logger.info(f"Sent initial device info: {device_info}")
    except Exception as e:
        logger.error(f"Error sending device info: {e}")
        emit('device_info', {
            'current_device': 'cpu',
            'gpu_available': False,
            'gpu_name': 'N/A',
            'gpu_memory_used': 0,
            'gpu_memory_total': 0
        })

@socketio.on('switch_device')
def handle_switch_device(data):
    """Handle device switching request."""
    try:
        if detector is None:
            raise RuntimeError("Detector not initialized")
            
        force_cpu = data.get('force_cpu', False)
        logger.info(f"Attempting to switch device (force_cpu={force_cpu})")
        
        with frame_lock:  # Ensure no frame is being processed during switch
            # Try to switch device
            success = detector.switch_device(force_cpu)
            
            if success:
                # Get updated device info after successful switch
                device_info = detector.get_device_info()
                emit('switch_result', {
                    'success': True,
                    'device_info': device_info
                })
                logger.info(f"Device switched successfully to {device_info['current_device']}")
                
                # Broadcast updated metrics immediately after switch
                current_metrics = get_current_metrics()
                if current_metrics:
                    socketio.emit('metrics_update', {
                        'current': current_metrics['current'],
                        'device_info': device_info
                    })
            else:
                raise RuntimeError(f"Failed to switch to {'CPU' if force_cpu else 'GPU'}")
                
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error switching device: {error_msg}")
        emit('switch_result', {
            'success': False,
            'error': error_msg
        })
        
        # Try to emit current device info after error
        try:
            if detector is not None:
                emit('device_info', detector.get_device_info())
        except:
            pass

def get_current_metrics():
    """Get current performance metrics."""
    try:
        if detector is None:
            return None
            
        device_info = detector.get_device_info()
        metrics = {
            'current': {
                'fps': detector.get_fps(),
                'frame_time': detector.get_processing_time(),
                'cpu_usage': detector.get_cpu_usage(),  # Fixed method name
                'memory_usage': detector.get_memory_usage()
            },
            'device_info': device_info
        }
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return None

def emit_metrics():
    """Emit metrics periodically."""
    metrics_history = {
        'fps': [],
        'frame_time': [],
        'cpu_usage': [],
        'memory_usage': [],
        'timestamps': []  # Add timestamps array
    }
    max_history = 60  # Keep 60 seconds of history
    start_time = time.time()  # Record start time
    
    while True:
        try:
            current_metrics = get_current_metrics()
            if current_metrics:
                current_time = time.time()
                # Update history
                for key in metrics_history:
                    if key != 'timestamps':  # Skip timestamps in this loop
                        metrics_history[key].append(current_metrics['current'][key])
                        if len(metrics_history[key]) > max_history:
                            metrics_history[key] = metrics_history[key][-max_history:]
                
                # Add timestamp
                timestamp = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(current_time))
                metrics_history['timestamps'].append(timestamp)
                if len(metrics_history['timestamps']) > max_history:
                    metrics_history['timestamps'] = metrics_history['timestamps'][-max_history:]
                
                # Emit both current values and history
                socketio.emit('metrics_update', {
                    'current': current_metrics['current'],
                    'history': metrics_history,
                    'start_time': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(start_time))
                })
                
                # Emit device info separately
                socketio.emit('device_info', current_metrics['device_info'])
                
        except Exception as e:
            logger.error(f"Error in metrics emission: {e}")
            
        time.sleep(1)  # Update every second

# Start metrics emission in a background thread
metrics_thread = threading.Thread(target=emit_metrics, daemon=True)
metrics_thread.start()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 