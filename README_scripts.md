# Baby Monitor System - Utility Scripts Guide

This document provides information about the utility scripts available in the Baby Monitor System project and how to use them.

## Scripts Overview

Below is a list of the utility scripts that have been organized into their respective directories:

### Model Management Scripts

- **`scripts/fix_emotion_models.bat`** - Windows batch file to fix emotion model installation issues
- **`src/babymonitor/utils/verify_emotion_models.py`** - Script to verify emotion model installation
- **`src/babymonitor/utils/setup_models.py`** - Script to set up and copy all models

### Testing Scripts

- **`tests/test_installer_ui.py`** - Test script for the installer UI
- **`tests/test_setup.py`** - Test script to verify setup.py functionality
- **`scripts/test_metrics.bat`** - Windows batch file to test metrics functionality

## How to Use These Scripts

### Model Management

#### Fix Emotion Models (Windows)

To fix issues with emotion models on Windows, run the following command from the project root:

```
scripts\fix_emotion_models.bat
```

This will:

1. Verify your Python installation
2. Run the setup_models.py script to copy all emotion models
3. Verify the emotion model installation

#### Verify Emotion Models

To verify that all emotion models are properly installed, run:

```
python -m src.babymonitor.utils.verify_emotion_models
```

This will scan both the global models directory and the application models directory and show which models are present or missing.

### Testing Scripts

#### Test Installer UI

To test the installer UI, run:

```
python tests\test_installer_ui.py
```

This script will verify that the setup.py file launches correctly and that platform-specific scripts are properly filtered.

#### Test Setup

To test that both setup.py files can be run directly, run:

```
python tests\test_setup.py
```

This will attempt to run both the root setup.py file and the scripts/install/setup.py file with the --help option.

#### Test Metrics (Windows)

To test the metrics functionality on Windows, run:

```
scripts\test_metrics.bat
```

This will start a test server that simulates metrics data for testing the metrics page.

## Directory Structure

The scripts have been organized into the following directory structure:

```
baby_monitor_system/
├── scripts/
│   ├── fix_emotion_models.bat     # Windows batch file to fix emotion models
│   ├── test_metrics.bat           # Windows batch file to test metrics
│   └── install/
│       └── setup.py               # Installation setup script
├── src/
│   └── babymonitor/
│       └── utils/
│           ├── setup_models.py    # Script to set up models
│           └── verify_emotion_models.py # Script to verify emotion models
└── tests/
    ├── test_installer_ui.py       # Test script for installer UI
    └── test_setup.py              # Test script for setup.py functionality
```

## Notes

- Always run these scripts from the project root directory
- Windows batch files (.bat) are Windows-specific and won't work on other operating systems
- Use the Python module notation (`python -m src.babymonitor.utils.script_name`) to ensure proper path resolution
