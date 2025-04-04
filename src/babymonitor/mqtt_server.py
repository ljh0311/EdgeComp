"""
MQTT Server Module for Baby Monitor
==================================

This module provides MQTT server functionality for the Baby Monitor application.
It allows clients to connect using MQTT and receive real-time updates.
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None
    logging.warning("Paho MQTT client not installed. MQTT functionality will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)

class MQTTServer:
    """MQTT server for the Baby Monitor."""

    def __init__(self, host: str = "localhost", port: int = 1883):
        """Initialize the MQTT server.

        Args:
            host: MQTT broker host
            port: MQTT broker port
        """
        self.host = host
        self.port = port
        self.client_id = f"babymonitor-server-{int(time.time())}"
        self.connected = False
        self.client = None
        self.connection_lock = threading.Lock()
        self.on_client_connected_callback = None

        # Define MQTT topics
        self.topics = {
            "video": "babymonitor/video",
            "emotion": "babymonitor/emotion",
            "system": "babymonitor/system",
            "alert": "babymonitor/alert",
            "crying": "babymonitor/crying",
        }

        # Check if MQTT is available
        if mqtt is None:
            logger.error("Paho MQTT client not installed. MQTT functionality will be disabled.")
            return

        # Initialize MQTT client
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback when the client connects to the broker."""
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {self.host}:{self.port}")
            self.connected = True
            # Subscribe to topics if needed
            # Currently we only publish, but we could subscribe to commands in the future
            if self.on_client_connected_callback:
                self.on_client_connected_callback()
        else:
            logger.error(f"Failed to connect to MQTT broker with error code {rc}")
            self.connected = False

    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Callback when the client disconnects from the broker."""
        logger.info(f"Disconnected from MQTT broker with code {rc}")
        self.connected = False

    def _on_message(self, client, userdata, message):
        """Callback when a message is received."""
        logger.debug(f"Received message on topic {message.topic}: {message.payload}")
        # Handle incoming messages if needed

    def set_on_client_connected_callback(self, callback: Callable):
        """Set callback to be called when a client connects.

        Args:
            callback: Function to call when a client connects
        """
        self.on_client_connected_callback = callback

    def start(self) -> bool:
        """Start the MQTT server.

        Returns:
            bool: True if started successfully, False otherwise
        """
        if mqtt is None:
            logger.error("MQTT functionality is disabled because Paho MQTT client is not installed.")
            return False

        try:
            with self.connection_lock:
                if not self.connected:
                    self.client.connect_async(self.host, self.port)
                    self.client.loop_start()
                    logger.info(f"Started MQTT server on {self.host}:{self.port}")
                    return True
                else:
                    logger.warning("MQTT server is already running")
                    return True
        except Exception as e:
            logger.error(f"Failed to start MQTT server: {str(e)}")
            return False

    def stop(self):
        """Stop the MQTT server."""
        if mqtt is None or self.client is None:
            return

        try:
            with self.connection_lock:
                if self.connected:
                    self.client.loop_stop()
                    self.client.disconnect()
                    logger.info("Stopped MQTT server")
                    self.connected = False
        except Exception as e:
            logger.error(f"Error stopping MQTT server: {str(e)}")

    def publish_video_frame(self, frame_data: bytes):
        """Publish a video frame.

        Args:
            frame_data: Binary JPEG frame data
        """
        if not self.connected or self.client is None:
            return

        try:
            self.client.publish(self.topics["video"], frame_data, qos=0)
        except Exception as e:
            logger.error(f"Error publishing video frame: {str(e)}")

    def publish_emotion_state(self, emotion_data: Dict[str, Any]):
        """Publish emotion state.

        Args:
            emotion_data: Dictionary containing emotion data
        """
        if not self.connected or self.client is None:
            return

        try:
            payload = json.dumps(emotion_data)
            self.client.publish(self.topics["emotion"], payload, qos=1)
        except Exception as e:
            logger.error(f"Error publishing emotion state: {str(e)}")

    def publish_system_status(self, status_data: Dict[str, Any]):
        """Publish system status.

        Args:
            status_data: Dictionary containing system status data
        """
        if not self.connected or self.client is None:
            return

        try:
            payload = json.dumps(status_data)
            self.client.publish(self.topics["system"], payload, qos=1)
        except Exception as e:
            logger.error(f"Error publishing system status: {str(e)}")

    def publish_alert(self, message: str, level: str = "info"):
        """Publish an alert.

        Args:
            message: Alert message
            level: Alert level (info, warning, danger)
        """
        if not self.connected or self.client is None:
            return

        try:
            payload = json.dumps({"message": message, "level": level})
            self.client.publish(self.topics["alert"], payload, qos=1)
        except Exception as e:
            logger.error(f"Error publishing alert: {str(e)}")

    def publish_crying_detection(self, confidence: float):
        """Publish crying detection result.

        Args:
            confidence: Confidence level (0-1)
        """
        if not self.connected or self.client is None:
            return

        try:
            payload = json.dumps({"confidence": confidence})
            self.client.publish(self.topics["crying"], payload, qos=1)
        except Exception as e:
            logger.error(f"Error publishing crying detection: {str(e)}")

    def is_connected(self) -> bool:
        """Check if connected to MQTT broker.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected 