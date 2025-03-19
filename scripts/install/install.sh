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

# Function to check Python version
check_python_version() {
    local python_cmd=$1
    local version=$($python_cmd -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)
    local patch=$(echo $version | cut -d. -f3)
    
    if [ $major -eq 3 ] && [ $minor -eq 11 ]; then
        if [ $patch -le 5 ]; then
            return 0
        fi
    elif [ $major -eq 3 ] && [ $minor -lt 11 ] && [ $minor -ge 8 ]; then
        return 0
    fi
    return 1
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed or not in PATH.${NC}"
    echo "Please install Python 3.8 or newer."
    echo "On Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
    echo "On macOS: brew install python3"
    exit 1
fi

# Check Python version and try to find a compatible version
PYTHON_CMD=""
for cmd in "python3.11" "python3.10" "python3.9" "python3.8" "python3"; do
    if command -v $cmd &> /dev/null; then
        if check_python_version $cmd; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}No compatible Python version found.${NC}"
    echo "Please install Python 3.11.5 or an earlier compatible version (3.8-3.11)."
    echo "Current Python versions found:"
    for cmd in "python3.11" "python3.10" "python3.9" "python3.8" "python3"; do
        if command -v $cmd &> /dev/null; then
            echo "$cmd: $($cmd --version 2>&1)"
        fi
    done
    exit 1
fi

# Show selected Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "Using Python: ${GREEN}${PYTHON_VERSION}${NC}"

# Check for required system dependencies
echo -e "\n${YELLOW}Checking system dependencies...${NC}"
MISSING_DEPS=()

check_dependency() {
    if ! dpkg -l "$1" &> /dev/null; then
        MISSING_DEPS+=("$1")
    fi
}

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Check for Linux dependencies
    check_dependency "python3-dev"
    check_dependency "python3-pip"
    check_dependency "python3-venv"
    check_dependency "libportaudio2"
    check_dependency "portaudio19-dev"
    check_dependency "libsndfile1"
    
    if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
        echo -e "${YELLOW}Missing dependencies: ${MISSING_DEPS[*]}${NC}"
        echo "Install them using:"
        echo "sudo apt-get install ${MISSING_DEPS[*]}"
        read -p "Would you like to install them now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo apt-get update
            sudo apt-get install -y "${MISSING_DEPS[@]}"
        else
            echo "Please install the dependencies and run this script again."
            exit 1
        fi
    fi
fi

# Create virtual environment
echo -e "\n${YELLOW}Setting up Python virtual environment...${NC}"
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install required packages
echo -e "\n${YELLOW}Installing required packages...${NC}"
pip install eventlet==0.33.3 flask-socketio==5.3.6 werkzeug==2.3.7 python-engineio==4.5.1 python-socketio==5.8.0

# Run the installer
echo -e "\n${YELLOW}Starting the Baby Monitor System installer...${NC}"
python install.py "$@"

if [ $? -ne 0 ]; then
    echo -e "\n${RED}Installation failed. Please check the error messages above.${NC}"
    echo "For more information, see INSTALL.md"
    exit 1
fi

# Create start script
echo -e "\n${YELLOW}Creating start script...${NC}"
cat > start_monitor.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
export PYTHONPATH=.
export EVENTLET_NO_GREENDNS=yes
python run_monitor.py --mode dev --camera_id 0 --debug
EOF

chmod +x start_monitor.sh
chmod +x run_monitor.py

echo -e "\n${GREEN}Installation completed successfully!${NC}"
echo
echo "You can now start the Baby Monitor System using:"
echo "1. Run: ./start_monitor.sh"
echo "2. Open http://localhost:5000 in your web browser"
echo
echo "For more information, see README.md" 