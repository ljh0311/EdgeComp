"""
System Monitor
=============
Module for monitoring system resources.
"""

import psutil
import logging

class SystemMonitor:
    def __init__(self):
        """Initialize the system monitor."""
        self.logger = logging.getLogger(__name__)

    def get_system_stats(self):
        """Get current system statistics."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
        except Exception as e:
            self.logger.error(f"Error getting system stats: {str(e)}")
            return None 