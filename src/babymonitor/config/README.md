# Baby Monitor System Configuration

This package contains the configuration management system for the Baby Monitor System. It provides a flexible and extensible way to manage configuration settings through various sources including environment variables, JSON files, and default values.

## Package Structure

- `__init__.py`: Main configuration class and initialization
- `defaults.py`: Default configuration values
- `env.py`: Environment-specific configuration handling
- `logging.py`: Logging configuration setup
- `utils.py`: Utility functions for configuration management

## Configuration Sources

The configuration system uses a hierarchical approach to load settings:

1. Default values (from `defaults.py`)
2. Environment variables (from `.env` file)
3. User-specific configuration files (JSON)
4. Runtime modifications

## Usage

### Basic Usage

```python
from babymonitor.config import Config

# Initialize configuration
config = Config()

# Access configuration values
camera_config = config.get('camera')
web_port = config.get('web', {}).get('port', 5000)

# Update configuration
config.update({'web': {'port': 8080}})

# Save configuration
config.save()
```

### Environment Variables

Copy the `.env.example` file to `.env` and modify the values as needed:

```bash
cp .env.example .env
```

### Configuration Files

Configuration files can be stored in the following locations:
- `./config/`
- `~/.babymonitor/`
- `src/babymonitor/config/`

## Available Settings

### Camera Configuration
- `width`: Camera frame width (default: 640)
- `height`: Camera frame height (default: 480)
- `fps`: Frames per second (default: 30)
- `default_device`: Default camera device ID (default: 0)

### Audio Configuration
- `sample_rate`: Audio sample rate (default: 16000)
- `chunk_size`: Audio chunk size (default: 8000)
- `channels`: Number of audio channels (default: 1)
- `format`: Audio format (default: 'float32')
- `device`: Audio device name (default: system default)
- `gain`: Audio gain (default: 1.0)

### Detection Configuration
- `person.threshold`: Person detection threshold (default: 0.7)
- `person.device`: Computing device for person detection (default: 'cpu')
- `emotion.threshold`: Emotion detection threshold (default: 0.7)
- `emotion.device`: Computing device for emotion detection (default: 'cpu')

### Web Interface Configuration
- `host`: Web server host (default: '0.0.0.0')
- `port`: Web server port (default: 5000)
- `debug`: Debug mode (default: False)
- `cors_origins`: CORS allowed origins (default: '*')

### Logging Configuration
- `level`: Logging level (default: 'INFO')
- `format`: Log message format
- `file`: Log file path
- `max_size`: Maximum log file size (default: 10MB)
- `backup_count`: Number of backup log files (default: 5)
- `console_output`: Enable console output (default: True)

### System Configuration
- `alert_threshold`: Alert triggering threshold (default: 0.7)
- `alert_cooldown`: Alert cooldown period in seconds (default: 10)
- `detection_interval`: Detection interval in seconds (default: 0.1)
- `save_detections`: Save detection results (default: True)
- `save_emotions`: Save emotion results (default: True)
- `history_length`: Number of events to keep in memory (default: 1000)
- `backup_interval`: Backup interval in seconds (default: 3600)
- `cleanup_interval`: Cleanup interval in seconds (default: 86400) 