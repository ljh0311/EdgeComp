#!/usr/bin/env python3
"""
MQTT Test Script for Baby Monitor
=================================

This script tests the MQTT functionality of the Baby Monitor system.
It connects to the MQTT broker and subscribes to all Baby Monitor topics
to verify data is being published correctly.
"""

import argparse
import json
import time
import threading
import logging
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Error: Paho MQTT client not installed. Please install it using 'pip install paho-mqtt'")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('MQTT_Tester')

# Global statistics
stats = {
    "video_frames": 0,
    "emotion_updates": 0,
    "system_updates": 0,
    "alerts": 0,
    "crying_events": 0,
    "start_time": None,
    "last_video_time": None,
    "last_emotion_time": None,
    "last_system_time": None,
    "last_alert_time": None,
    "last_crying_time": None,
}

# Topics
TOPICS = {
    "video": "babymonitor/video",
    "emotion": "babymonitor/emotion",
    "system": "babymonitor/system",
    "alert": "babymonitor/alert",
    "crying": "babymonitor/crying",
}

# Global lock for thread-safe updates
stats_lock = threading.Lock()

def on_connect(client, userdata, flags, rc, properties=None):
    """Callback when connected to MQTT broker"""
    if rc == 0:
        logger.info("Connected to MQTT broker")
        
        # Subscribe to all topics
        for topic in TOPICS.values():
            client.subscribe(topic)
            logger.info(f"Subscribed to {topic}")
        
        # Initialize start time
        with stats_lock:
            stats["start_time"] = time.time()
            
    else:
        logger.error(f"Failed to connect to MQTT broker with code {rc}")
        exit(1)

def on_message(client, userdata, msg):
    """Callback when message is received"""
    topic = msg.topic
    now = time.time()

    with stats_lock:
        if topic == TOPICS["video"]:
            stats["video_frames"] += 1
            stats["last_video_time"] = now
            # Don't log video frames as they're too frequent
        
        elif topic == TOPICS["emotion"]:
            stats["emotion_updates"] += 1
            stats["last_emotion_time"] = now
            try:
                data = json.loads(msg.payload.decode())
                logger.info(f"Emotion update: {data}")
            except Exception as e:
                logger.error(f"Error parsing emotion data: {e}")
        
        elif topic == TOPICS["system"]:
            stats["system_updates"] += 1
            stats["last_system_time"] = now
            try:
                data = json.loads(msg.payload.decode())
                logger.info(f"System update: {data}")
            except Exception as e:
                logger.error(f"Error parsing system data: {e}")
        
        elif topic == TOPICS["alert"]:
            stats["alerts"] += 1
            stats["last_alert_time"] = now
            try:
                data = json.loads(msg.payload.decode())
                logger.info(f"Alert: [{data.get('level', 'info')}] {data.get('message', '')}")
            except Exception as e:
                logger.error(f"Error parsing alert data: {e}")
        
        elif topic == TOPICS["crying"]:
            stats["crying_events"] += 1
            stats["last_crying_time"] = now
            try:
                data = json.loads(msg.payload.decode())
                logger.info(f"Crying detected: confidence={data.get('confidence', 0):.2f}")
            except Exception as e:
                logger.error(f"Error parsing crying data: {e}")

def print_stats():
    """Print statistics about received messages"""
    with stats_lock:
        if stats["start_time"] is None:
            logger.info("No messages received yet")
            return
            
        elapsed = time.time() - stats["start_time"]
        minutes = int(elapsed / 60)
        seconds = elapsed % 60
        
        logger.info(f"\n=== MQTT Statistics (after {minutes}m {seconds:.1f}s) ===")
        logger.info(f"Video frames:    {stats['video_frames']} ({stats['video_frames']/elapsed:.1f} fps)")
        logger.info(f"Emotion updates: {stats['emotion_updates']} ({stats['emotion_updates']/elapsed:.1f} per second)")
        logger.info(f"System updates:  {stats['system_updates']} ({stats['system_updates']/elapsed:.1f} per second)")
        logger.info(f"Alerts:          {stats['alerts']}")
        logger.info(f"Crying events:   {stats['crying_events']}")
        
        # Check for stalled streams
        now = time.time()
        video_stalled = stats["last_video_time"] is not None and (now - stats["last_video_time"]) > 5
        logger.info(f"\nStream Status:")
        logger.info(f"Video:   {'STALLED' if video_stalled else 'OK'}")
        
        if stats["last_emotion_time"] is not None:
            emotion_age = now - stats["last_emotion_time"]
            logger.info(f"Emotion: Last update {emotion_age:.1f}s ago")
            
        if stats["last_system_time"] is not None:
            system_age = now - stats["last_system_time"]
            logger.info(f"System:  Last update {system_age:.1f}s ago")
            
        if stats["last_alert_time"] is not None:
            alert_age = now - stats["last_alert_time"]
            logger.info(f"Alerts:  Last alert {alert_age:.1f}s ago")

def stats_thread_func(interval=10):
    """Thread function to periodically print statistics"""
    while True:
        time.sleep(interval)
        print_stats()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MQTT Test Script for Baby Monitor")
    parser.add_argument("--host", default="localhost", help="MQTT broker hostname")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--stats-interval", type=int, default=10, 
                        help="Interval in seconds for printing statistics")
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    logger.info(f"Connecting to MQTT broker at {args.host}:{args.port}")
    
    # Create MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Start statistics thread
    stats_thread = threading.Thread(target=stats_thread_func, args=(args.stats_interval,))
    stats_thread.daemon = True
    stats_thread.start()
    
    try:
        # Connect to broker
        client.connect(args.host, args.port)
        
        # Start MQTT loop
        client.loop_start()
        
        logger.info("Press Ctrl+C to exit")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        
    finally:
        # Print final statistics
        print_stats()
        
        # Clean up
        client.loop_stop()
        client.disconnect()
        logger.info("Disconnected from MQTT broker")

if __name__ == "__main__":
    main()