#!/usr/bin/env python3
"""
Baby Monitor System - Main Entry Point

This script serves as the main entry point for the Baby Monitor System, supporting three launch modes:
1. Normal Mode: Default for standard users, showing the main page with camera feed and metrics, but no access to dev tools.
2. Dev Mode: Displays metrics page with access to all development tools.
3. Local Mode: Shows the local GUI version of the baby monitor.

Usage:
    python main.py --mode [normal|dev|local] [options]

Options:
    --mode MODE             Launch mode (normal, dev, local) [default: normal]
    --device DEVICE         Target device (pc, pi) [default: pc]
    --threshold THRESHOLD   Detection threshold [default: 0.5]
    --camera_id CAMERA_ID   Camera ID [default: 0]
    --input_device INPUT    Audio input device ID [default: None]
    --host HOST             Host for web interface [default: 0.0.0.0]
    --port PORT             Port for web interface [default: 5000]
    --mqtt-host HOST        MQTT broker host (defaults to --host)
    --mqtt-port PORT        Port for MQTT broker [default: 1883]
    --mqtt-disable          Disable MQTT broker (use only HTTP/WebSocket)
    --debug                 Enable debug mode
    --person-model          Path to the person detection model (YOLOv8n.pt)
"""

# Apply eventlet monkey patching at the very beginning
import eventlet
eventlet.monkey_patch(os=True, select=True, socket=True, thread=True, time=True)

import os
import sys
import time
import json
import signal
import logging
import argparse
import threading
import io
from threading import Event, Lock
from pathlib import Path
from datetime import datetime

# Try to import MQTT broker library
try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.publish import multiple as mqtt_publish_multiple
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("Warning: paho-mqtt module not available. MQTT functionality will be disabled.")
    print("To enable MQTT, install: pip install paho-mqtt")

# Device-specific configurations
DEVICE_CONFIGS = {
    'pc': {
        'video': {
            'width': 640,
            'height': 480,
            'fps': 30,
            'format': 'MJPG'
        },
        'audio': {
            'rate': 44100,
            'channels': 2,
            'chunk_size': 1024
        },
        'processing': {
            'scale_factor': 1.0,
            'use_gpu': True,
            'buffer_size': 10
        }
    },
    'pi': {
        'video': {
            'width': 320,
            'height': 240,
            'fps': 15,
            'format': 'RGB'
        },
        'audio': {
            'rate': 22050,
            'channels': 1,
            'chunk_size': 512
        },
        'processing': {
            'scale_factor': 0.5,
            'use_gpu': False,
            'buffer_size': 5
        }
    }
}

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import Baby Monitor components
from babymonitor.camera_wrapper import Camera
from babymonitor.audio import AudioProcessor
from babymonitor.detectors.person_detector import PersonDetector
from babymonitor.detectors.emotion_detector import EmotionDetector
from babymonitor.web.simple_server import SimpleBabyMonitorWeb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'baby_monitor.log'))
    ]
)
logger = logging.getLogger('baby_monitor')

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Global variables to track running state and components
stop_event = Event()
running_components = []

# Define a flexible MQTT publisher
class MQTTPublisher:
    """MQTT Publisher for Baby Monitor System"""
    
    def __init__(self, host='localhost', port=1883):
        self.host = host
        self.port = port
        self.running = False
        self.client = None
        self.client_connected = False
        self.lock = Lock()
        
        # Statistics
        self.messages_sent = 0
        self.start_time = None
        
        # Create topics
        self.topics = {
            'video': 'babymonitor/video',
            'emotion': 'babymonitor/emotion',
            'system': 'babymonitor/system',
            'alert': 'babymonitor/alert',
            'crying': 'babymonitor/crying'
        }
        
        # Video frame handling
        self.latest_frame = None
        self.last_frame_time = 0
        self.frame_rate = 10  # Target frame rate for MQTT video
        self.min_frame_interval = 1.0 / self.frame_rate
    
    def start(self):
        """Start the MQTT publisher"""
        if not MQTT_AVAILABLE:
            logger.warning("MQTT functionality is disabled. Missing paho-mqtt module.")
            return False
            
        if self.running:
            logger.warning("MQTT publisher is already running")
            return True
            
        logger.info(f"Starting MQTT publisher connecting to {self.host}:{self.port}")
        try:
            # Try to detect Mosquitto running locally if host is localhost
            if self.host in ('localhost', '127.0.0.1', '::1'):
                try:
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('127.0.0.1', self.port))
                    sock.close()
                    if result != 0:
                        logger.warning(f"No MQTT broker detected on {self.host}:{self.port}.")
                        logger.warning("Make sure Mosquitto or another MQTT broker is running.")
                        logger.warning("Windows users: Download from https://mosquitto.org/download/")
                        logger.warning("Linux users: sudo apt-get install mosquitto")
                except:
                    pass
            
            # Connect our publisher client
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"babymonitor-server-{id(self)}")
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            
            # Set reconnection parameters
            self.client.reconnect_delay_set(min_delay=1, max_delay=30)
            
            # Connect to broker with a short timeout
            self.client.connect_async(self.host, self.port)
            self.client.loop_start()
            
            # Set statistics
            self.start_time = datetime.now()
            self.running = True
            
            # Publish a test message to ensure connection is working
            self.publish_alert('info', 'Baby Monitor server started with MQTT support')
            
            return True
        except Exception as e:
            logger.error(f"Failed to start MQTT publisher: {e}")
            self.running = False
            return False
    
    def stop(self):
        """Stop the MQTT publisher"""
        if not self.running:
            return
            
        logger.info("Stopping MQTT publisher")
        try:
            if self.client:
                # Publish a shutdown message
                try:
                    self.publish_alert('info', 'Baby Monitor server shutting down')
                except:
                    pass
                    
                self.client.loop_stop()
                self.client.disconnect()
                
            self.running = False
            logger.info("MQTT publisher stopped")
        except Exception as e:
            logger.error(f"Error stopping MQTT publisher: {e}")
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Handle client connection"""
        if rc == 0:
            logger.info("MQTT client connected to broker")
            self.client_connected = True
        else:
            logger.error(f"MQTT client connection failed with code {rc}")
            self.client_connected = False
    
    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Handle client disconnection"""
        logger.info("MQTT client disconnected from broker")
        self.client_connected = False
    
    def publish_video_frame(self, frame_data):
        """Publish a video frame if enough time has passed since the last frame"""
        if not self.running or not self.client_connected:
            return
            
        current_time = time.time()
        if current_time - self.last_frame_time >= self.min_frame_interval:
            try:
                with self.lock:
                    # Convert frame to JPEG binary data
                    import cv2
                    import numpy as np
                    if isinstance(frame_data, np.ndarray):
                        # If frame is a numpy array (OpenCV frame), encode it
                        _, buffer = cv2.imencode('.jpg', frame_data, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        frame_bytes = buffer.tobytes()
                    else:
                        # If frame is already binary data, use it directly
                        frame_bytes = frame_data
                        
                    self.client.publish(self.topics['video'], frame_bytes)
                    self.messages_sent += 1
                    self.last_frame_time = current_time
            except Exception as e:
                logger.error(f"Error publishing video frame: {e}")
    
    def publish_emotion_update(self, emotion_data):
        """Publish emotion detection update"""
        if not self.running:
            return
            
        try:
            with self.lock:
                payload = json.dumps(emotion_data)
                self.client.publish(self.topics['emotion'], payload)
                
                # If crying detected with high confidence, also publish to crying topic
                if emotion_data.get('emotion') == 'crying' and emotion_data.get('confidence', 0) > 0.7:
                    self.client.publish(self.topics['crying'], payload)
                    
                self.messages_sent += 1
        except Exception as e:
            logger.error(f"Error publishing emotion update: {e}")
    
    def publish_system_info(self, system_data):
        """Publish system information update"""
        if not self.running:
            return
            
        try:
            with self.lock:
                self.client.publish(self.topics['system'], json.dumps(system_data))
                self.messages_sent += 1
        except Exception as e:
            logger.error(f"Error publishing system info: {e}")
    
    def publish_alert(self, level, message):
        """Publish an alert message"""
        if not self.running:
            return
            
        try:
            with self.lock:
                alert_data = {
                    'level': level,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                }
                self.client.publish(self.topics['alert'], json.dumps(alert_data))
                self.messages_sent += 1
        except Exception as e:
            logger.error(f"Error publishing alert: {e}")
    
    def get_stats(self):
        """Get publisher statistics"""
        if not self.running:
            return {
                'running': False,
                'uptime': '00:00:00',
                'messages_sent': 0,
                'broker_connected': False
            }
            
        uptime = (datetime.now() - self.start_time) if self.start_time else datetime.timedelta(0)
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        return {
            'running': True,
            'uptime': uptime_str,
            'messages_sent': self.messages_sent,
            'broker_connected': self.client_connected
        }

def signal_handler(sig, frame):
    """
    Improved signal handler for graceful shutdown.
    
    This will set the stop_event and attempt to stop all running components
    that have been registered.
    """
    logger.info("Shutdown signal received (Ctrl+C). Stopping Baby Monitor System...")
    
    # Set the global stop event
    stop_event.set()

    # Try to stop web interface which might be blocking in the main thread
    for component in running_components:
        if hasattr(component, 'stop') and callable(component.stop):
            logger.info(f"Stopping component: {component.__class__.__name__}")
            try:
                component.stop()
            except Exception as e:
                logger.error(f"Error stopping {component.__class__.__name__}: {e}")
    
    # Signal to eventlet that we want to stop
    try:
        import eventlet
        eventlet.kill(eventlet.getcurrent())
    except Exception as e:
        logger.error(f"Error killing eventlet greenlet: {e}")
    
    # If all else fails, exit more forcefully after a short delay
    def force_exit():
        logger.warning("Forcing exit...")
        os._exit(0)
        
    # Schedule a force exit if graceful shutdown doesn't work
    from threading import Timer
    t = Timer(5.0, force_exit)
    t.daemon = True
    t.start()

# Register the signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def register_component(component):
    """Register a component to be stopped on shutdown."""
    global running_components
    running_components.append(component)
    return component

def apply_device_config(args):
    """Apply device-specific configurations"""
    device = args.device.lower()
    if device not in DEVICE_CONFIGS:
        logger.warning(f"Unknown device '{device}', using PC configuration")
        device = 'pc'
    
    config = DEVICE_CONFIGS[device]
    
    # Set environment variables for configuration
    os.environ['VIDEO_WIDTH'] = str(config['video']['width'])
    os.environ['VIDEO_HEIGHT'] = str(config['video']['height'])
    os.environ['VIDEO_FPS'] = str(config['video']['fps'])
    os.environ['VIDEO_FORMAT'] = config['video']['format']
    
    os.environ['AUDIO_RATE'] = str(config['audio']['rate'])
    os.environ['AUDIO_CHANNELS'] = str(config['audio']['channels'])
    os.environ['AUDIO_CHUNK_SIZE'] = str(config['audio']['chunk_size'])
    
    os.environ['PROCESSING_SCALE'] = str(config['processing']['scale_factor'])
    os.environ['USE_GPU'] = str(config['processing']['use_gpu']).lower()
    os.environ['BUFFER_SIZE'] = str(config['processing']['buffer_size'])
    
    logger.info(f"Applied {device.upper()} configuration")
    if device == 'pi':
        logger.info("Running in Raspberry Pi mode with optimized settings")
        # Additional Pi-specific setup
        try:
            import picamera
            logger.info("PiCamera module available")
        except ImportError:
            logger.warning("PiCamera module not found. Please install it for Raspberry Pi camera support")

def check_mqtt_broker(host, port):
    """Check if an MQTT broker is available at the specified host and port"""
    if not MQTT_AVAILABLE:
        return False
        
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def main(args):
    # Apply device-specific configuration
    apply_device_config(args)
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Also set debug level for audio processor and emotion detector
        logging.getLogger('AudioProcessor').setLevel(logging.INFO)
        logging.getLogger('babymonitor.detectors.emotion_detector').setLevel(logging.INFO)
    
    # Check if MQTT broker is running
    mqtt_broker_available = False
    if not args.mqtt_disable and MQTT_AVAILABLE:
        mqtt_broker_available = check_mqtt_broker(args.mqtt_host or 'localhost', args.mqtt_port)
        if not mqtt_broker_available:
            logger.warning(f"No MQTT broker detected at {args.mqtt_host or 'localhost'}:{args.mqtt_port}")
            if args.mqtt_host in ('localhost', '127.0.0.1', '::1'):
                logger.warning("For MQTT support, please install and run Mosquitto or another MQTT broker:")
                logger.warning("  - Windows: https://mosquitto.org/download/")
                logger.warning("  - Linux: sudo apt-get install mosquitto")
                logger.warning("  - Mac: brew install mosquitto")
            logger.warning("Continuing without MQTT support (HTTP/WebSocket only)")
    
    # Initialize camera first
    logger.info(f"Initializing camera for {args.device.upper()}...")
    camera = register_component(Camera(args.camera_id))
    if not camera or not camera.is_opened():
        logger.error("Failed to initialize camera")
        return 1

    # Initialize detectors
    logger.info("Initializing detectors...")
    person_detector = register_component(PersonDetector(
        model_path=args.person_model,
        threshold=args.threshold
    ))
    emotion_detector = register_component(EmotionDetector(threshold=args.threshold))

    # Start MQTT publisher if enabled
    mqtt_publisher = None
    if not args.mqtt_disable and MQTT_AVAILABLE and mqtt_broker_available:
        logger.info(f"Starting MQTT publisher connecting to {args.mqtt_host or 'localhost'}:{args.mqtt_port}...")
        mqtt_publisher = register_component(MQTTPublisher(host=args.mqtt_host or 'localhost', port=args.mqtt_port))
        if not mqtt_publisher.start():
            logger.warning("Failed to start MQTT publisher, continuing with HTTP/WebSocket only")
            mqtt_publisher = None
    else:
        logger.info("MQTT disabled or not available, using HTTP/WebSocket only")
        
    # Initialize web interface
    logger.info("Initializing web interface...")
    web = register_component(SimpleBabyMonitorWeb(
        camera=camera,  # Pass the initialized camera
        person_detector=person_detector,
        emotion_detector=emotion_detector,
        host=args.host,
        port=args.port,
        mode=args.mode,
        debug=args.debug,
        stop_event=stop_event
    ))
    
    # Initialize audio processor with emotion detector
    logger.info("Initializing audio processor...")
    audio_processor = register_component(AudioProcessor(
        device=args.input_device,
        emotion_detector=emotion_detector
    ))
    
    # Set up emotion callback to handle both MQTT and Socket.IO
    def emotion_callback(data):
        if not stop_event.is_set():
            # Send via Socket.IO
            web.emit_emotion_update(data)
            
            # Send via MQTT if available
            if mqtt_publisher and mqtt_publisher.running:
                mqtt_publisher.publish_emotion_update(data)
    
    # Register the emotion callback
    audio_processor.set_emotion_callback(emotion_callback)
    
    # Connect camera frame callbacks to MQTT if available
    if mqtt_publisher and mqtt_publisher.running:
        # Override web's frame processing to also send to MQTT
        original_process_frame = web._process_frames
        
        def process_frame_with_mqtt(frame):
            # Process frame with original method for HTTP/WebSocket
            result = original_process_frame(frame)
            
            # Also send to MQTT
            if mqtt_publisher and mqtt_publisher.running:
                # Don't send every frame, limit by frame rate
                mqtt_publisher.publish_video_frame(frame)
                
            return result
        
        # Replace the method
        web.process_frame = process_frame_with_mqtt
        
        # Set up system info publishing for MQTT
        def publish_system_info():
            if mqtt_publisher and mqtt_publisher.running:
                try:
                    system_info = {
                        'uptime': web.get_uptime_str(),
                        'cpu_usage': web.system_monitor.get_cpu_percent(),
                        'memory_usage': web.system_monitor.get_memory_percent(),
                        'camera_status': 'running' if camera and camera.is_opened() else 'error',
                        'person_detector_status': 'running' if person_detector and person_detector.is_loaded() else 'error',
                        'emotion_detector_status': 'running' if emotion_detector and emotion_detector.is_loaded() else 'error',
                        'timestamp': datetime.now().isoformat()
                    }
                    mqtt_publisher.publish_system_info(system_info)
                except Exception as e:
                    logger.error(f"Error publishing system info: {e}")
        
        # Set up periodic system info publishing
        system_info_thread = threading.Thread(
            target=lambda: web.run_periodic(publish_system_info, interval=5.0),
            daemon=True
        )
        system_info_thread.start()
    
    # Import the global message queue
    from babymonitor.audio import global_message_queue
    
    # Create a greenlet to process messages from the global queue
    def process_message_queue():
        # Add error handling wrapper around entire function
        try:
            logger.info("Starting message queue processor")
            message_count = 0
            last_stats_time = time.time()
            processed_batch_ids = set()  # Track already processed batch IDs
            
            while not stop_event.is_set():
                try:
                    # Check if we should stop
                    if stop_event.is_set() or not web.running:
                        logger.info("Stop event set or web interface stopped, stopping message queue processor")
                        break
                    
                    # Use a non-blocking get with short timeout to allow for cooperative multitasking
                    try:
                        message_type, callback, data = global_message_queue.get(timeout=0.1)
                        message_count += 1
                        
                        # Log statistics periodically
                        current_time = time.time()
                        if current_time - last_stats_time > 30:
                            logger.debug(f"Message queue stats: processed {message_count} messages in the last 30 seconds")
                            message_count = 0
                            last_stats_time = current_time
                            # Also clean up old batch IDs to prevent set from growing too large
                            processed_batch_ids = set()
                        
                        # Process message based on type
                        if message_type == 'emotion':
                            try:
                                if data:
                                    # Check for duplicate batch_id to avoid processing the same audio chunk multiple times
                                    batch_id = data.get('batch_id', 'unknown')
                                    
                                    # Only process if we haven't seen this batch_id before
                                    if batch_id not in processed_batch_ids:
                                        processed_batch_ids.add(batch_id)
                                        
                                        # Log message details at debug level
                                        emotion = data.get('emotion', 'unknown')
                                        confidence = data.get('confidence', 0.0)
                                        
                                        logger.debug(f"Processing emotion message: {emotion} ({confidence:.4f}) [batch: {batch_id}]")
                                        
                                        # Use emotion_callback for both MQTT and Socket.IO
                                        if not stop_event.is_set():
                                            emotion_callback(data)
                                        
                                    else:
                                        logger.debug(f"Skipping duplicate emotion batch: {batch_id}")
                                else:
                                    logger.warning("Received empty emotion data")
                            except Exception as e:
                                logger.error(f"Error processing emotion update: {str(e)}")
                                import traceback
                                logger.error(traceback.format_exc())
                    except eventlet.queue.Empty:
                        # Queue is empty, just yield to other greenlets
                        eventlet.sleep(0.05)
                        continue
                    
                    # Always yield to other greenlets after processing a message
                    # This is crucial for eventlet cooperative multitasking
                    eventlet.sleep(0)
                    
                except Exception as e:
                    logger.error(f"Error processing message queue: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    eventlet.sleep(0.5)  # Sleep longer after an error
                
                # Check stop_event more frequently
                if stop_event.is_set():
                    logger.info("Stop event detected in message queue processor")
                    break
        except Exception as e:
            logger.error(f"Message queue processor crashed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("Starting message queue processor...")
    queue_processor = eventlet.spawn(process_message_queue)
    register_component(queue_processor)  # Register for proper cleanup
    
    # Print access information
    print("\n" + "="*80)
    print(f"Baby Monitor Server is running!")
    print(f"Web Interface: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    if mqtt_publisher and mqtt_publisher.running:
        print(f"MQTT Support:  Connected to broker at {args.mqtt_host or 'localhost'}:{args.mqtt_port}")
    print("="*80 + "\n")
    
    try:
        # Start audio processing explicitly
        logger.info("Starting audio processing...")
        audio_processor.start()
        
        # Start web interface (this will block)
        web.run(stop_event=stop_event)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt caught in main function")
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        logger.info("Cleaning up from main function...")
        stop_event.set()  # Ensure stop event is set
        
        # Explicitly stop components in reverse order
        for component in reversed(running_components):
            if hasattr(component, 'stop') and callable(component.stop):
                logger.info(f"Stopping {component.__class__.__name__}")
                try:
                    component.stop()
                except Exception as e:
                    logger.error(f"Error stopping {component.__class__.__name__}: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Baby Monitor System')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'dev', 'local'],
                        help='Launch mode (normal, dev, local)')
    
    # Device selection
    parser.add_argument('--device', type=str, default='pc',
                        choices=['pc', 'pi'],
                        help='Target device (pc or pi)')
    
    # Common options
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera ID')
    parser.add_argument('--input_device', type=int, default=None,
                        help='Audio input device ID')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host for web interface')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for web interface')
    parser.add_argument('--mqtt-host', type=str, default=None,
                        help='MQTT broker host (defaults to --host)')
    parser.add_argument('--mqtt-port', type=int, default=1883,
                        help='Port for MQTT broker')
    parser.add_argument('--mqtt-disable', action='store_true',
                        help='Disable MQTT broker (use only HTTP/WebSocket)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--person-model', type=str, default=None,
                        help='Path to the person detection model (YOLOv8n.pt)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("Main program interrupted via keyboard")
        signal_handler(signal.SIGINT, None)
    finally:
        logger.info("Baby Monitor System shutdown complete")
