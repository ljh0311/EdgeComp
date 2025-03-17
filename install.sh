#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display banner
echo -e "${BLUE}"
echo "==============================================="
echo "    Baby Monitor System - Unix Installer"
echo "==============================================="
echo -e "${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed or not in PATH.${NC}"
    echo "Please install Python 3.8 or newer."
    echo "On Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
    echo "On macOS: brew install python3"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "Detected Python version: ${GREEN}${PYTHON_VERSION}${NC}"

# Run the installer
echo
echo -e "${YELLOW}Starting the Baby Monitor System installer...${NC}"
echo

python3 install.py

if [ $? -ne 0 ]; then
    echo
    echo -e "${RED}Installation failed. Please check the error messages above.${NC}"
    echo "For more information, see INSTALL.md"
    exit 1
fi

echo
echo -e "${GREEN}Installation completed successfully!${NC}"
echo
echo "You can now start the Baby Monitor System using:"
echo "1. The desktop shortcut (if created during installation)"
echo "2. Command line: python3 main.py"
echo

# Make the script executable
chmod +x main.py 