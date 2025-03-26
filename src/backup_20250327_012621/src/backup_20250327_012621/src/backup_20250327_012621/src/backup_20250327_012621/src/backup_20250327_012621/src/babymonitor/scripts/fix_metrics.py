#!/usr/bin/env python
"""
Metrics Dashboard Test/Fix Script

This script runs a standalone Flask server with Socket.IO that
generates demo data for the metrics dashboard. It helps test and
troubleshoot the metrics dashboard functionality.

Usage:
    python fix_metrics.py
"""

import os
import sys
import time
import random
import threading
import logging
from pathlib import Path

# Add the parent directory to sys.path to import babymonitor modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

from babymonitor.core.metrics import MetricsCollector, demo_metrics_generation

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricsDemoServer:
    """A demo server for testing the metrics dashboard."""
    
    def __init__(self, host='127.0.0.1', port=5000):
        """Initialize the demo server."""
        self.host = host
        self.port = port
        
        # Create Flask app and SocketIO instance
        self.app = Flask(__name__, 
                         template_folder=os.path.join(parent_dir, 'src/babymonitor/web/templates'),
                         static_folder=os.path.join(parent_dir, 'src/babymonitor/web/static'))
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Create metrics collector
        self.metrics_collector = MetricsCollector()
        
        # Setup routes and socket events
        self._setup_routes()
        self._setup_socketio()
        
        # Start metrics emission thread
        self.running = False
        self.metrics_thread = None
        
        # Print startup info
        logger.info(f"Metrics demo server initialized on http://{self.host}:{self.port}")
        logger.info(f"Template folder: {self.app.template_folder}")
        logger.info(f"Static folder: {self.app.static_folder}")
        
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/metrics')
        def metrics():
            return render_template('metrics.html')
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API endpoint for metrics."""
            return jsonify(self.metrics_collector.get_metrics())
        
        @self.app.route('/api/system_info')
        def api_system_info():
            """API endpoint for system info."""
            return jsonify(self.metrics_collector.system_info)
    
    def _setup_socketio(self):
        """Setup Socket.IO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            
            # Send initial metrics data
            metrics_data = self.metrics_collector.get_metrics()
            self.socketio.emit('metrics_update', metrics_data)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_metrics')
        def handle_request_metrics(data=None):
            """Handle client request for metrics data."""
            time_range = '1h'
            if data and 'timeRange' in data:
                time_range = data['timeRange']
            
            logger.info(f"Metrics requested with time range: {time_range}")
            
            # Get current metrics
            metrics_data = self.metrics_collector.get_metrics()
            
            # Emit metrics update
            self.socketio.emit('metrics_update', metrics_data)
    
    def _emit_metrics(self):
        """Periodically emit metrics updates."""
        while self.running:
            try:
                # Get current metrics
                metrics_data = self.metrics_collector.get_metrics()
                
                # Emit metrics update
                self.socketio.emit('metrics_update', metrics_data)
                
                # Randomly emit crying events (5% chance)
                if random.random() < 0.05:
                    self.socketio.emit('crying_detected', {
                        'timestamp': time.time() * 1000  # Use milliseconds
                    })
                    
                    # Also emit a general alert
                    self.socketio.emit('alert', {
                        'message': 'Crying detected!',
                        'level': 'warning',
                        'timestamp': time.time() * 1000
                    })
                
                # Sleep for 2 seconds
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error emitting metrics: {e}")
    
    def start(self):
        """Start the demo server."""
        # Start demo metrics generation
        self.demo_thread = demo_metrics_generation(self.metrics_collector)
        
        # Start metrics emission thread
        self.running = True
        self.metrics_thread = threading.Thread(target=self._emit_metrics)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
        
        # Start the Flask server
        logger.info(f"Starting Metrics Demo Server on http://{self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=True, use_reloader=False)

def main():
    """Main function to run the metrics demo server."""
    try:
        # Create and start the demo server
        server = MetricsDemoServer()
        server.start()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")

if __name__ == "__main__":
    main() 