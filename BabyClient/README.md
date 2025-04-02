# Baby Monitor Client

A client application for the Baby Monitor System that connects to the server using MQTT as the primary protocol, with HTTP/WebSocket as a fallback for enhanced reliability.

## Features

- Real-time video streaming from the baby monitor camera
- Emotion detection and alerts
- System status monitoring
- Automatic server discovery
- Primary MQTT connection with HTTP/Socket.IO fallback
- Dark mode UI theme

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Auto-Connect Mode (Recommended)

Run the auto-connect script to automatically discover and connect to the baby monitor server:

```bash
python auto_connect.py
```

or use the batch file:

```bash
auto_connect.bat
```

The auto-connect tool will:
1. Scan your local network for MQTT brokers
2. Verify both MQTT and HTTP services are available
3. Automatically start the client with the correct settings
4. Fall back to HTTP/Socket.IO if MQTT is unavailable

### Manual Connection

You can start the client manually with specific connection parameters:

```bash
python baby_client.py --host <SERVER_IP> --port <HTTP_PORT> --mqtt-host <MQTT_IP> --mqtt-port <MQTT_PORT>
```

or use the batch file with default settings:

```bash
start_client.bat
```

Default connection settings:
- MQTT Broker: 192.168.1.100:1883 (Primary)
- HTTP Server: 192.168.1.100:5000 (Fallback)

### Command Line Arguments

- `--mqtt-host`: MQTT broker IP address (default: same as host)
- `--mqtt-port`: MQTT broker port (default: 1883)
- `--host`: HTTP server IP address (default: 192.168.1.100)
- `--port`: HTTP server port (default: 5000)

## Connection Architecture

The client implements a dual-protocol connection strategy:

1. Primary MQTT Connection:
   - Used for all real-time updates and events
   - Efficient publish-subscribe pattern
   - Automatic reconnection handling

2. Fallback HTTP/WebSocket Connection:
   - Activated if MQTT connection fails
   - Compatible with legacy server configurations
   - Seamless fallback handling

This dual-connection architecture ensures:
- Maximum reliability through protocol redundancy
- Efficient real-time communication via MQTT
- Backward compatibility with HTTP-only servers
- Automatic protocol switching on connection issues

## Troubleshooting

If the auto-connect fails:
1. Ensure the MQTT broker is running and accessible (port 1883)
2. Check that HTTP services are available (port 5000)
3. Verify your network allows both MQTT and HTTP connections
4. Try manual connection with specific IP/port settings

For more detailed logs, run the client with debug output:
```bash
python baby_client.py --debug
```

## Contributing

This client is part of the Baby Monitor System developed by:
- JunHong: Backend & Client Architecture
- Darrel: Dashboard Frontend
- Ashraf: Datasets & Model Architecture
- Xuan Yu: Dataset Collection & Model Training
- Javin: Camera Detection System

## License

Copyright (c) 2024 Baby Monitor System Team
All rights reserved.
