#!/usr/bin/env python3
"""
Baby Monitor System - Legacy Script

This script serves as a compatibility layer for users who are still using the old command structure.
It redirects to the new main.py script with appropriate arguments.

Usage:
    python scripts/run.py --mode [full|person|emotion|web] [options]

This script is deprecated. Please use the new main.py script instead:
    python main.py --mode [normal|dev|local] [options]
"""

import os
import sys
import argparse
import subprocess
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('baby_monitor_legacy')

def main():
    """Parse command line arguments and redirect to the new main.py script."""
    # Show deprecation warning
    warnings.warn(
        "This script is deprecated and will be removed in a future version. "
        "Please use 'python main.py' instead.",
        DeprecationWarning, stacklevel=2
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Baby Monitor System (Legacy)')
    
    # Legacy arguments
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'person', 'emotion', 'web'],
                        help='Operation mode (full, person, emotion, web)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Detection threshold')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model directory')
    parser.add_argument('--audio_device', type=int, default=None,
                        help='Audio input device ID')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Web interface host')
    parser.add_argument('--port', type=int, default=5000,
                        help='Web interface port')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Map legacy mode to new mode
    mode_mapping = {
        'full': 'normal',
        'web': 'normal',
        'person': 'normal',
        'emotion': 'normal'
    }
    
    new_mode = mode_mapping.get(args.mode, 'normal')
    
    # Construct command for main.py
    main_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'main.py')
    
    cmd = [
        sys.executable,
        main_script,
        '--mode', new_mode,
        '--camera_id', str(args.camera),
        '--threshold', str(args.threshold),
        '--host', args.host,
        '--port', str(args.port)
    ]
    
    # Add optional arguments
    if args.audio_device is not None:
        cmd.extend(['--input_device', str(args.audio_device)])
    
    if args.debug:
        cmd.append('--debug')
    
    # Log the redirection
    logger.info(f"Redirecting to new script: {' '.join(cmd)}")
    logger.info("This script is deprecated. Please use 'python main.py' instead.")
    
    # Execute the new script
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error executing main.py: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 