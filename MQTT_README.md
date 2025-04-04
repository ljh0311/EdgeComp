# MQTT Support for Baby Monitor

This document provides information on how to set up and use the MQTT functionality in the Baby Monitor system.

## Overview

The Baby Monitor system now supports MQTT (Message Queuing Telemetry Transport) as the primary communication protocol between the server and client. MQTT is a lightweight publish/subscribe messaging protocol designed for constrained devices and low-bandwidth, high-latency or unreliable networks.

Benefits of using MQTT:
- Lower bandwidth usage compared to HTTP/Socket.IO
- Better performance on unstable networks
- Decoupled communication between server and clients
- Supports multiple clients simultaneously

If MQTT is not available or fails to connect, the system automatically falls back to HTTP/Socket.IO to ensure connectivity.

## Connection Architecture

The Baby Monitor communication architecture works as follows:

1. The Baby Monitor Server (main.py) acts as an MQTT client that publishes data to topics on an MQTT broker
2. The Baby Monitor Client (baby_client.py) connects to the same MQTT broker to subscribe to these topics
3. If MQTT connection fails, both server and client fallback to HTTP/Socket.IO for communication

## Installation Requirements

### Server-side Requirements

To enable MQTT support, you need to:

1. Install the Paho MQTT client library:
   ```
   pip install paho-mqtt>=2.0.0
   ```

2. Install and run an MQTT broker (such as Mosquitto):
   - **Windows**: Use the setup script included in `scripts/setup_mqtt.bat`
   - **Linux**: `sudo apt install mosquitto mosquitto-clients`
   - **macOS**: `brew install mosquitto`

### Client-side Requirements

The Baby Monitor client requires the Paho MQTT client library:
```
pip install paho-mqtt>=2.0.0
```

## Running with MQTT Support

### Starting the Server

To start the Baby Monitor server with MQTT support:

```bash
python main.py --mqtt-host localhost --mqtt-port 1883
```

Parameters:
- `--mqtt-host`: The MQTT broker hostname or IP address
- `--mqtt-port`: The MQTT broker port (default: 1883)

### Starting the Client

To start the Baby Monitor client with MQTT support:

```bash
python baby_client.py --host localhost --port 5000 --mqtt-host localhost --mqtt-port 1883
```

Parameters:
- `--host`: The Baby Monitor server hostname/IP (for HTTP/Socket.IO fallback)
- `--port`: The Baby Monitor server port (for HTTP/Socket.IO fallback)
- `--mqtt-host`: The MQTT broker hostname or IP address
- `--mqtt-port`: The MQTT broker port (default: 1883)

## MQTT Topics

The Baby Monitor uses the following MQTT topics:

| Topic                 | Description                      | Format                     |
|-----------------------|----------------------------------|----------------------------|
| `babymonitor/video`   | Video frames from camera         | Binary JPEG data           |
| `babymonitor/emotion` | Emotion detection results        | JSON with emotions and confidence values |
| `babymonitor/system`  | System status information        | JSON with system metrics   |
| `babymonitor/alert`   | Alerts and notifications         | JSON with message and level |
| `babymonitor/crying`  | Crying detection events          | JSON with confidence value |

## Testing MQTT Connection

To test if your MQTT broker is working correctly:

1. Install the Mosquitto command-line clients
2. Subscribe to all Baby Monitor topics:
   ```bash
   mosquitto_sub -h localhost -p 1883 -t "babymonitor/#" -v
   ```
3. Publish a test message:
   ```bash
   mosquitto_pub -h localhost -p 1883 -t "babymonitor/alert" -m '{"message":"MQTT Test","level":"info"}'
   ```

## Troubleshooting

- **Connection Issues**: If the client cannot connect to the MQTT broker, it will automatically fall back to HTTP/Socket.IO.
- **Missing Data**: If you're not receiving data on certain topics, make sure the broker is running and there are no firewall issues.
- **Performance Issues**: Try adjusting MQTT QoS levels in the code if you experience performance problems.

## Advanced Configuration

For advanced users, the following MQTT parameters can be modified in the code:

- Quality of Service (QoS) levels
- Retain flags
- Client IDs
- Username/password authentication
- TLS/SSL encryption

Refer to the Paho MQTT documentation for more details.

## Compatibility

This MQTT implementation is compatible with any standard MQTT broker, including:

- Mosquitto
- HiveMQ
- EMQ X
- AWS IoT Core
- Azure IoT Hub 