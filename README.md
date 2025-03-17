# Baby Monitor System

A comprehensive baby monitoring system with video, audio, and emotion detection capabilities.

## Project Structure

```
src/
├── babymonitor/
│   ├── core/           # Core system components
│   ├── detectors/      # Person and motion detection
│   ├── audio/         # Audio processing and analysis
│   ├── web/           # Web interface
│   └── utils/         # Utility functions
├── tests/             # Test scripts
└── scripts/           # Run scripts
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/babymonitor.git
cd babymonitor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Run the setup script to download required models:
```bash
python setup.py
```

## Usage

### Starting the System

1. Basic usage:
```bash
python scripts/run.py
```

2. With specific options:
```bash
python scripts/run_babymonitor.py --detector yolov8 --force-cpu
```

3. Testing detection:
```bash
python tests/test_detection.py --dev
```

4. Testing lightweight detection:
```bash
python tests/test_lightweight_detection.py --web
```

### Web Interface

Once started, access the web interface at:
- Main dashboard: http://localhost:5000
- Direct camera feed: http://localhost:5000/direct-feed

## Configuration

The system can be configured through:
1. Environment variables
2. Command line arguments
3. Configuration file at `config/system_config.json`

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

The project uses:
- Black for code formatting
- Flake8 for linting

Run formatting:
```bash
black src/ tests/ scripts/
```

Run linting:
```bash
flake8 src/ tests/ scripts/
```

## License

MIT License - see LICENSE file for details.
