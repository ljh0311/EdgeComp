"""
Alert Management System
=====================
Handles alert generation, management, and notification for the baby monitor system.
"""

from datetime import datetime
import logging
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import os
from pathlib import Path

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertType(Enum):
    FALL = "fall"
    RAPID_MOTION = "rapid_motion"
    EMOTION = "emotion"
    SYSTEM = "system"
    AUDIO = "audio"
    PERSON = "person"

@dataclass
class Alert:
    type: AlertType
    level: AlertLevel
    message: str
    timestamp: datetime
    details: Optional[Dict] = None
    should_notify: bool = True
    
    def to_dict(self):
        return {
            "type": self.type.value,
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "details": self.details,
            "should_notify": self.should_notify
        }

class AlertManager:
    def __init__(self, max_history: int = 100, storage_path: Optional[str] = None):
        """
        Initialize the alert manager.
        
        Args:
            max_history (int): Maximum number of alerts to keep in history
            storage_path (str, optional): Path to store alert history
        """
        self.logger = logging.getLogger(__name__)
        self.max_history = max_history
        self.alerts: List[Alert] = []
        self.alert_handlers = []
        self.storage_path = storage_path or str(Path.home() / ".babymonitor" / "alerts")
        
        # Create storage directory if it doesn't exist
        if self.storage_path:
            os.makedirs(self.storage_path, exist_ok=True)
        
        # Load previous alerts
        self._load_alerts()
    
    def add_alert_handler(self, handler):
        """Add a handler function to be called when new alerts are generated."""
        self.alert_handlers.append(handler)
    
    def create_alert(self, 
                    alert_type: AlertType,
                    level: AlertLevel,
                    message: str,
                    details: Optional[Dict] = None,
                    should_notify: bool = True) -> Alert:
        """
        Create and process a new alert.
        
        Args:
            alert_type (AlertType): Type of the alert
            level (AlertLevel): Severity level of the alert
            message (str): Alert message
            details (dict, optional): Additional alert details
            should_notify (bool): Whether to notify users
            
        Returns:
            Alert: The created alert object
        """
        alert = Alert(
            type=alert_type,
            level=level,
            message=message,
            timestamp=datetime.now(),
            details=details,
            should_notify=should_notify
        )
        
        # Add to history
        self.alerts.append(alert)
        
        # Trim history if needed
        if len(self.alerts) > self.max_history:
            self.alerts = self.alerts[-self.max_history:]
        
        # Save to storage
        self._save_alerts()
        
        # Notify handlers
        self._notify_handlers(alert)
        
        return alert
    
    def get_recent_alerts(self, limit: int = 10, 
                         alert_type: Optional[AlertType] = None,
                         level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get recent alerts with optional filtering."""
        filtered = self.alerts
        
        if alert_type:
            filtered = [a for a in filtered if a.type == alert_type]
        if level:
            filtered = [a for a in filtered if a.level == level]
            
        return filtered[-limit:]
    
    def clear_alerts(self):
        """Clear all alerts from history."""
        self.alerts = []
        self._save_alerts()
    
    def _notify_handlers(self, alert: Alert):
        """Notify all registered handlers of new alert."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {str(e)}")
    
    def _save_alerts(self):
        """Save alerts to persistent storage."""
        if not self.storage_path:
            return
            
        try:
            alerts_file = os.path.join(self.storage_path, "alert_history.json")
            with open(alerts_file, 'w') as f:
                json.dump([a.to_dict() for a in self.alerts], f)
        except Exception as e:
            self.logger.error(f"Error saving alerts: {str(e)}")
    
    def _load_alerts(self):
        """Load alerts from persistent storage."""
        if not self.storage_path:
            return
            
        try:
            alerts_file = os.path.join(self.storage_path, "alert_history.json")
            if os.path.exists(alerts_file):
                with open(alerts_file, 'r') as f:
                    data = json.load(f)
                    self.alerts = []
                    for alert_dict in data:
                        self.alerts.append(Alert(
                            type=AlertType(alert_dict["type"]),
                            level=AlertLevel(alert_dict["level"]),
                            message=alert_dict["message"],
                            timestamp=datetime.strptime(alert_dict["timestamp"], "%Y-%m-%d %H:%M:%S"),
                            details=alert_dict.get("details"),
                            should_notify=alert_dict.get("should_notify", True)
                        ))
        except Exception as e:
            self.logger.error(f"Error loading alerts: {str(e)}")
    
    def create_fall_alert(self, details: Optional[Dict] = None):
        """Create a fall detection alert."""
        return self.create_alert(
            AlertType.FALL,
            AlertLevel.CRITICAL,
            "Possible fall detected in monitored area",
            details=details
        )
    
    def create_rapid_motion_alert(self, details: Optional[Dict] = None):
        """Create a rapid motion alert."""
        return self.create_alert(
            AlertType.RAPID_MOTION,
            AlertLevel.WARNING,
            "Unusual rapid movement detected",
            details=details
        )
    
    def create_emotion_alert(self, emotion: str, confidence: float, details: Optional[Dict] = None):
        """Create an emotion-related alert."""
        if emotion.lower() in ['anger', 'fear', 'sadness'] and confidence > 0.7:
            return self.create_alert(
                AlertType.EMOTION,
                AlertLevel.WARNING,
                f"High confidence {emotion.lower()} emotion detected",
                details=details
            )
        elif emotion.lower() == 'worried' and confidence > 0.8:
            return self.create_alert(
                AlertType.EMOTION,
                AlertLevel.WARNING,
                "High distress detected",
                details=details
            )
    
    def create_system_alert(self, message: str, level: AlertLevel = AlertLevel.WARNING):
        """Create a system-related alert."""
        return self.create_alert(
            AlertType.SYSTEM,
            level,
            message,
            should_notify=level == AlertLevel.CRITICAL
        ) 