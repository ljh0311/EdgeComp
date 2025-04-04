#!/bin/bash

# Check for Python installation
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    ACTION="${1:-gui}"
    echo "This script requires administrative privileges"
    sudo "$0" "$ACTION" "$2" "$3" "$4" "$5"
    exit $?
fi

# Parse arguments
ACTION="${1:-gui}"

# Run the system manager
python3 system_manager.py "$ACTION" "$2" "$3" "$4" "$5"
exit $? 