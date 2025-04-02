#!/usr/bin/env python3
"""
Baby Monitor Client Auto-Connect Tool

This script helps to automatically detect and connect to a baby monitor server
on the local network. It scans common IP addresses and ports to find the server
and MQTT broker.

Usage:
    python auto_connect.py
"""

import sys
import os
import socket
import threading
import time
import subprocess
import ipaddress
import platform
from queue import Queue
import paho.mqtt.client as mqtt

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Create a socket that connects to a well-known external server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

def is_port_open(ip, port, timeout=0.5):
    """Check if a port is open on a given IP address"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

def verify_mqtt_broker(ip, port=1883, timeout=1):
    """Verify if the IP/port combination is an MQTT broker"""
    try:
        client = mqtt.Client()
        client.connect_async(ip, port)
        client.loop_start()
        time.sleep(timeout)
        if client.is_connected():
            client.loop_stop()
            client.disconnect()
            return True
        client.loop_stop()
        return False
    except:
        return False

def verify_baby_monitor(ip, port=5000, timeout=1):
    """Verify if the IP/port combination is a baby monitor server"""
    try:
        # Check MQTT first as it's the primary protocol
        if verify_mqtt_broker(ip):
            # If MQTT is available, check HTTP as backup
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            
            # Return True if both MQTT and HTTP are available
            return result == 0
        return False
    except:
        return False

def scan_network(base_ip, port=5000, result_queue=None):
    """Scan the network for baby monitor servers"""
    # Parse the base IP to get the network part
    ip_parts = base_ip.split('.')
    network_prefix = '.'.join(ip_parts[0:3])
    
    # Scan common IP addresses first (common for Raspberry Pi)
    common_ips = [f"{network_prefix}.1", f"{network_prefix}.100", f"{network_prefix}.101", base_ip]
    
    for ip in common_ips:
        if verify_mqtt_broker(ip):  # Check MQTT first
            if verify_baby_monitor(ip, port):  # Then verify HTTP backup
                if result_queue:
                    result_queue.put((ip, port, 1883))
                return ip, port, 1883
    
    # Scan the entire subnet (1-254)
    for i in range(1, 255):
        ip = f"{network_prefix}.{i}"
        if ip in common_ips:
            continue  # Already checked
            
        if verify_mqtt_broker(ip):  # Check MQTT first
            if verify_baby_monitor(ip, port):  # Then verify HTTP backup
                if result_queue:
                    result_queue.put((ip, port, 1883))
                return ip, port, 1883
    
    return None, None, None

def start_client(ip, port, mqtt_port):
    """Start the baby monitor client with the given IP and ports"""
    print(f"Starting Baby Monitor Client with server at {ip}:{port} (MQTT: {mqtt_port})")
    
    if platform.system() == "Windows":
        # Use pythonw.exe to avoid keeping the console window open
        subprocess.Popen(f"python baby_client.py --host {ip} --port {port} --mqtt-host {ip} --mqtt-port {mqtt_port}")
    else:
        # For non-Windows platforms
        subprocess.Popen(["python", "baby_client.py", "--host", ip, "--port", str(port), 
                         "--mqtt-host", ip, "--mqtt-port", str(mqtt_port)])

def main():
    print("Baby Monitor Auto-Connect Tool")
    print("==============================")
    print("Scanning the network for baby monitor servers...")
    print("Note: MQTT is the primary protocol, HTTP/Socket.IO will be used as fallback")
    
    # Get the local IP
    local_ip = get_local_ip()
    print(f"Local IP: {local_ip}")
    
    # Start a thread to scan the network
    result_queue = Queue()
    scan_thread = threading.Thread(target=scan_network, args=(local_ip, 5000, result_queue))
    scan_thread.daemon = True
    scan_thread.start()
    
    # Wait for the scan to complete or timeout
    timeout = 30  # seconds
    start_time = time.time()
    server_found = False
    
    try:
        while (time.time() - start_time) < timeout:
            print(f"Scanning... ({int(time.time() - start_time)}s)", end='\r')
            
            if not result_queue.empty():
                ip, port, mqtt_port = result_queue.get()
                print(f"\nFound baby monitor server at {ip}")
                print(f"MQTT broker port: {mqtt_port}")
                print(f"HTTP fallback port: {port}")
                server_found = True
                break
                
            time.sleep(0.5)
        
        if not server_found:
            print("\nNo baby monitor servers found automatically.")
            print("Would you like to:")
            print("1) Enter server details manually")
            print("2) Start the client with default settings")
            print("3) Exit")
            
            choice = input("Enter your choice (1-3): ")
            
            if choice == "1":
                ip = input("Enter server IP: ")
                mqtt_port_str = input("Enter MQTT port (default 1883): ")
                http_port_str = input("Enter HTTP port (default 5000): ")
                mqtt_port = int(mqtt_port_str) if mqtt_port_str else 1883
                http_port = int(http_port_str) if http_port_str else 5000
                
                # Verify MQTT connection
                if verify_mqtt_broker(ip, mqtt_port):
                    print("MQTT connection verified")
                else:
                    print("Warning: MQTT connection failed, will try HTTP/Socket.IO")
                
                start_client(ip, http_port, mqtt_port)
            elif choice == "2":
                start_client("192.168.1.100", 5000, 1883)
            else:
                print("Exiting...")
                return
        else:
            # Start the client with the discovered server
            start_client(ip, port, mqtt_port)
            
    except KeyboardInterrupt:
        print("\nScan cancelled. Exiting...")
        return

if __name__ == "__main__":
    main() 