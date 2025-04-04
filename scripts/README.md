# Baby Monitor System - Scripts Directory

This directory contains utility scripts for the Baby Monitor System.

## Directory Structure

- `/install/` - Installation and setup scripts
- `*.bat` - Windows batch files
- `*.sh` - Linux/Mac shell scripts
- `*.py` - Python utility scripts

## Installation

To install the Baby Monitor System:

Windows:

## Starting the Application

The Baby Monitor System should be started using the main.py script in the root directory:

```bash
python main.py --mode normal  # For regular users
python main.py --mode dev     # For developers
```

**Note about start scripts:** The repository includes several start scripts (`start_monitor.bat`, `start.bat`, `start.sh`, `start_pi.sh`) which provide platform-specific ways to launch the application. While these scripts work, it's recommended to use `main.py` directly for consistency across platforms.

## Utility Scripts

- `run.py` - Utility script for running components of the Baby Monitor System
- `scripts_manager.bat` / `scripts_manager.sh` - Script management utilities for different platforms
- `scripts_manager_gui.py` - Graphical interface for script management
- `scripts_manager_pi.sh` - Raspberry Pi optimized script management
- `restart_baby_monitor.bat` - Utility to restart the Baby Monitor System on Windows

## Fix Scripts

- `fix_windows.bat` - Fixes common issues on Windows systems
- `fix_raspberry.sh` - Fixes common issues on Raspberry Pi
- `fix_metrics.bat` - Fixes metrics-related issues

## Demo Scripts

- `metrics_demo.py` / `metrics_demo.bat` - Demonstrates and tests the metrics system

## Usage

Most scripts can be run directly:

### Windows
```
scripts\script_name.bat
```

### Linux/Mac
```
bash scripts/script_name.sh
```

### Python Scripts
```
python scripts/script_name.py
```

## Adding New Scripts

When adding new scripts to this directory:

1. Follow the naming conventions:
   - Use `.bat` extension for Windows scripts
   - Use `.sh` extension for Linux/Mac scripts
   - Use `.py` extension for Python scripts
2. Add appropriate comments at the top of the script
3. Update this README.md if adding a new category of scripts 