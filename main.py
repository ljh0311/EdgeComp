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
    --threshold THRESHOLD   Detection threshold [default: 0.5]
    --camera_id CAMERA_ID   Camera ID [default: 0]
    --input_device INPUT    Audio input device ID [default: None]
    --host HOST             Host for web interface [default: 0.0.0.0]
    --port PORT             Port for web interface [default: 5000]
    --debug                 Enable debug mode
"""

# Apply eventlet monkey patching at the very beginning
import eventlet
eventlet.monkey_patch(os=True, select=True, socket=True, thread=True, time=True)

import os
import sys
import time
import signal
import logging
import argparse
from threading import Event
from pathlib import Path

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

def run_normal_mode(args):
    """
    Start the Baby Monitor in normal mode.
    
    This mode is intended for standard users and provides access to the main
    dashboard and metrics, but not to development tools.
    """
    logger.info("Starting Baby Monitor System in NORMAL mode")
    
    try:
        # Initialize camera
        logger.info("Initializing camera...")
        camera = register_component(Camera(args.camera_id))
        
        # Initialize person detector
        logger.info("Initializing person detector...")
        person_detector = register_component(PersonDetector(
            threshold=args.threshold
        ))
        
        # Initialize emotion detector
        logger.info("Initializing emotion detector...")
        emotion_detector = register_component(EmotionDetector(
            threshold=args.threshold
        ))
        
        # Initialize audio processor with emotion detector
        logger.info("Initializing audio processor...")
        audio_processor = register_component(AudioProcessor(
            device=args.input_device,
            emotion_detector=emotion_detector
        ))
        
        # Set up emotion callback to send updates to web interface
        def emotion_callback(result):
            if 'web_interface' in locals() and not stop_event.is_set():
                web_interface.emit_emotion_update(result)
        
        audio_processor.set_emotion_callback(emotion_callback)
        
        # Start web interface
        logger.info("Starting web interface...")
        web_interface = register_component(SimpleBabyMonitorWeb(
            camera=camera,
            person_detector=person_detector,
            emotion_detector=emotion_detector,
            host=args.host,
            port=args.port,
            mode="normal",
            debug=args.debug,
            stop_event=stop_event  # Pass stop_event to web interface
        ))
        
        # Print access information
        print("\n" + "="*80)
        print(f"Baby Monitor Web Interface is running!")
        print(f"Access the web interface at: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
        print("Press Ctrl+C to stop the server")
        print("="*80 + "\n")
        
        # Start the web interface in the main thread, but check stop_event
        try:
            web_interface.run(stop_event=stop_event)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected in web interface")
            signal_handler(signal.SIGINT, None)
        
        # This code will only be reached when the web interface is stopped
        logger.info("Web interface stopped")
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Error in normal mode: {e}")
        raise
    finally:
        logger.info("Shutting down Baby Monitor System...")
        # Shutdown is handled by signal_handler and running_components list
    
    return 0

def run_dev_mode(args):
    """
    Start the Baby Monitor in developer mode.
    
    This mode provides access to all development tools and metrics,
    and is intended for developers and testers.
    """
    logger.info("Starting Baby Monitor System in DEVELOPER mode")
    
    # Set logging level to DEBUG in dev mode
    logger.setLevel(logging.DEBUG)
    
    try:
        # Initialize camera
        logger.info("Initializing camera...")
        camera = register_component(Camera(args.camera_id))
        
        # Initialize person detector
        logger.info("Initializing person detector...")
        person_detector = register_component(PersonDetector(
            threshold=args.threshold
        ))
        
        # Initialize emotion detector
        logger.info("Initializing emotion detector...")
        emotion_detector = register_component(EmotionDetector(
            threshold=args.threshold
        ))
        
        # Initialize audio processor with emotion detector
        logger.info("Initializing audio processor...")
        audio_processor = register_component(AudioProcessor(
            device=args.input_device,
            emotion_detector=emotion_detector
        ))
        
        # Set up emotion callback to send updates to web interface
        def emotion_callback(result):
            if 'web_interface' in locals() and not stop_event.is_set():
                web_interface.emit_emotion_update(result)
        
        audio_processor.set_emotion_callback(emotion_callback)
        
        # Start web interface in dev mode
        logger.info("Starting web interface in developer mode...")
        web_interface = register_component(SimpleBabyMonitorWeb(
            camera=camera,
            person_detector=person_detector,
            emotion_detector=emotion_detector,
            host=args.host,
            port=args.port,
            mode="dev",
            debug=True,  # Always enable debug in dev mode
            stop_event=stop_event  # Pass stop_event to web interface
        ))
        
        # Print access information
        print("\n" + "="*80)
        print(f"Baby Monitor Web Interface is running in DEVELOPER mode!")
        print(f"Access the web interface at: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
        print(f"Developer tools are available at: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/dev/tools")
        print("Press Ctrl+C to stop the server")
        print("="*80 + "\n")
        
        # Start the web interface in the main thread, but check stop_event
        try:
            web_interface.run(stop_event=stop_event)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected in web interface")
            signal_handler(signal.SIGINT, None)
        
        # This code will only be reached when the web interface is stopped
        logger.info("Web interface stopped")
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Error in developer mode: {e}")
        raise
    finally:
        logger.info("Shutting down Baby Monitor System...")
        # Shutdown is handled by signal_handler and running_components list
    
    return 0

def run_local_mode(args):
    """
    Start the Baby Monitor in local GUI mode.
    
    This mode runs the local GUI version of the baby monitor using PyQt5.
    """
    logger.info("Starting Baby Monitor System in LOCAL mode")
    
    try:
        # Import GUI components
        try:
            from PyQt5.QtWidgets import QApplication
            from babymonitor.gui.main_gui import launch_main_gui
        except ImportError as e:
            logger.error(f"Failed to import GUI components: {e}")
            logger.error("Make sure PyQt5 is installed: pip install PyQt5")
            return 1
        
        # Launch the main GUI
        logger.info("Starting local GUI...")
        return launch_main_gui(stop_event=stop_event)
        
    except Exception as e:
        logger.error(f"Error in local mode: {e}")
        raise

def main(args):
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Also set debug level for audio processor and emotion detector
        logging.getLogger('AudioProcessor').setLevel(logging.INFO)
        logging.getLogger('babymonitor.detectors.emotion_detector').setLevel(logging.INFO)
        
    # Initialize camera first
    logger.info("Initializing camera...")
    camera = register_component(Camera(args.camera_id))
    if not camera or not camera.is_opened():
        logger.error("Failed to initialize camera")
        return 1

    # Initialize detectors
    logger.info("Initializing detectors...")
    person_detector = register_component(PersonDetector(threshold=args.threshold))
    emotion_detector = register_component(EmotionDetector(threshold=args.threshold))
    
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
    
    # Set up emotion callback
    audio_processor.set_emotion_callback(web.emit_emotion_update)
    
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
                                        
                                        # Send to web interface - we're in the same eventlet context as the web server
                                        if not stop_event.is_set():
                                            web.emit_emotion_update(data)
                                        
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
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
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
