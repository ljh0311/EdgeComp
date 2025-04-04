#!/bin/bash

# Baby Monitor System - Raspberry Pi Optimization Script
# This script optimizes Raspberry Pi performance settings for running the baby monitor system
# Run with sudo privileges for full optimization

echo "======================================"
echo "  Baby Monitor RPi Optimization Tool"
echo "======================================"
echo

# Check if running with root privileges
if [ "$EUID" -ne 0 ]; then
  echo "Warning: Some optimizations require root privileges."
  echo "Consider running with 'sudo' for full optimization."
  echo
fi

# Check if we're running on a Raspberry Pi
if [ ! -f /proc/device-tree/model ] || ! grep -q "Raspberry Pi" /proc/device-tree/model; then
  echo "Error: This script is intended for Raspberry Pi systems only."
  exit 1
fi

echo "Detected: $(cat /proc/device-tree/model)"
echo

# Function to apply optimizations
apply_optimization() {
  if [ "$EUID" -eq 0 ]; then
    return 0  # True, can apply
  else
    echo "Skipping (requires root): $1"
    return 1  # False, cannot apply
  fi
}

echo "Applying performance optimizations..."
echo

# CPU Governor Optimization
if apply_optimization "CPU Governor"; then
  echo "Setting CPU Governor to performance mode..."
  
  # Check if cpufreq directory exists
  if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
    echo "✓ CPU Governor set to: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  else
    echo "× CPU frequency scaling not available on this system"
  fi
fi

# Memory Split Optimization
if apply_optimization "GPU Memory"; then
  echo "Optimizing GPU memory allocation..."
  
  # Set GPU memory based on total memory
  total_mem=$(grep MemTotal /proc/meminfo | awk '{print $2}')
  
  if [ $total_mem -lt 1000000 ]; then
    # Less than 1GB RAM - allocate 128MB to GPU
    gpu_mem=128
  elif [ $total_mem -lt 2000000 ]; then
    # 1GB-2GB RAM - allocate 256MB to GPU
    gpu_mem=256
  else
    # More than 2GB RAM - allocate 384MB to GPU
    gpu_mem=384
  fi
  
  if [ -f /boot/config.txt ]; then
    # Remove any existing gpu_mem line
    sed -i '/^gpu_mem=/d' /boot/config.txt
    
    # Add our gpu_mem line
    echo "gpu_mem=$gpu_mem" >> /boot/config.txt
    echo "✓ GPU memory set to ${gpu_mem}MB (requires reboot)"
  else
    echo "× Could not find /boot/config.txt"
  fi
fi

# Swap Optimization
if apply_optimization "Swap Settings"; then
  echo "Optimizing swap settings..."
  
  # Reduce swappiness
  sysctl -w vm.swappiness=10 > /dev/null
  
  # Make it permanent
  if ! grep -q "vm.swappiness" /etc/sysctl.conf; then
    echo "vm.swappiness=10" >> /etc/sysctl.conf
  else
    sed -i 's/^vm.swappiness=.*/vm.swappiness=10/' /etc/sysctl.conf
  fi
  
  echo "✓ Swap preference reduced (vm.swappiness=10)"
fi

# System Service Optimization
if apply_optimization "System Services"; then
  echo "Checking for unnecessary services..."
  
  # Services that can be safely disabled on a dedicated baby monitor
  services_to_disable=(
    "bluetooth.service"
    "avahi-daemon.service"
    "triggerhappy.service"
    "cups.service"
  )
  
  for service in "${services_to_disable[@]}"; do
    if systemctl is-active --quiet "$service"; then
      systemctl stop "$service"
      systemctl disable "$service"
      echo "✓ Disabled $service"
    fi
  done
fi

# Network optimization
if apply_optimization "Network Settings"; then
  echo "Optimizing network settings..."
  
  # Optimize network for lower latency
  sysctl -w net.ipv4.tcp_fastopen=3 > /dev/null
  
  # Make it permanent
  if ! grep -q "net.ipv4.tcp_fastopen" /etc/sysctl.conf; then
    echo "net.ipv4.tcp_fastopen=3" >> /etc/sysctl.conf
  fi
  
  echo "✓ Network settings optimized"
fi

echo
echo "Optimization complete!"
if [ "$EUID" -eq 0 ]; then
  echo "All optimizations have been applied."
  echo "Some changes require a reboot to take effect."
  echo
  read -p "Would you like to reboot now? (y/n): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Rebooting..."
    reboot
  fi
else
  echo "Some optimizations were skipped (requires root)."
  echo "Run with 'sudo' for full optimization."
fi 