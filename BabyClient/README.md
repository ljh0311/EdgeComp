# Baby Monitor Client

A desktop application that connects to a Baby Monitor system running on a Raspberry Pi.

## Features

- Live video feed from the baby monitor
- Real-time emotion detection display
- System status monitoring
- Alert notifications and history
- Auto-connect tool to find the baby monitor server on your network

## Requirements

- Python 3.6 or higher
- PyQt5 and dependencies (see requirements.txt)

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

### Easy Start (Recommended)

For Windows users, simply double-click the `auto_connect.bat` file. The tool will:
1. Scan your network for the baby monitor server
2. Automatically connect if found
3. Allow you to enter server details manually if not found

### Manual Start

Run the client application directly with:

```
python baby_client.py --host YOUR_RASPBERRY_PI_IP --port 5000
```

Replace `YOUR_RASPBERRY_PI_IP` with the IP address of your Raspberry Pi running the baby monitor server.

### Command Line Options

- `--host`: The IP address of the baby monitor server (default: 192.168.1.100)
- `--port`: The port of the baby monitor server (default: 5000)

### Using the Client

1. When the application starts, it will attempt to connect to the baby monitor server
2. If the connection fails, you can click the "Connect" button to enter a different server address
3. The main window shows:
   - Live video feed from the camera
   - Current emotion detection state
   - System status (uptime, CPU usage, etc.)
   - Alert history

## Troubleshooting

If you encounter issues:

- Ensure your Raspberry Pi is running the baby monitor server
- Check that the IP address and port are correct
- Verify that your network allows connections to the specified port
- Check that all dependencies are properly installed
- Try the auto-connect tool which can automatically find the server on your network 