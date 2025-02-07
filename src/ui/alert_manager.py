"""
Alert Manager Module
==================
Handles system alerts, notifications, and alert history.
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime
import platform
import logging

class AlertManager:
    def __init__(self, root, config):
        """
        Initialize the alert manager.
        
        Args:
            root: Tkinter root window
            config: Alert configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.root = root
        self.config = config
        self.current_alert_timer = None
        self.alert_history = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup alert UI components."""
        # Alert container
        self.alert_container = ttk.Frame(self.root, style='AlertContainer.TFrame')
        
        # Current alert frame
        self.alert_frame = ttk.Frame(self.alert_container, style='Alert.TFrame')
        self.alert_label = ttk.Label(
            self.alert_frame,
            text="",
            font=('Arial', 16, 'bold'),
            foreground='white'
        )
        self.alert_label.grid(row=0, column=0, pady=10, padx=10, sticky="ew")
        
        # Dismiss button
        self.dismiss_button = ttk.Button(
            self.alert_frame,
            text="Dismiss Alert",
            command=self.dismiss_alert
        )
        self.dismiss_button.grid(row=1, column=0, pady=(0, 5))
        
        # History section
        self.setup_history_section()
    
    def setup_history_section(self):
        """Setup the alert history section."""
        self.history_container = ttk.Frame(self.alert_container, style='AlertHistory.TFrame')
        
        # History title
        self.history_title = ttk.Label(
            self.history_container,
            text="Alert History",
            font=('Arial', 14, 'bold'),
            foreground='white',
            background=self.config['dark_theme']['background']
        )
        self.history_title.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 0))
        
        # Create scrollable canvas for history
        self.history_canvas = tk.Canvas(
            self.history_container,
            background=self.config['dark_theme']['background'],
            highlightthickness=0
        )
        self.history_canvas.grid(row=1, column=0, sticky="nsew", padx=(10, 0))
        
        # Add scrollbar
        self.history_scrollbar = ttk.Scrollbar(
            self.history_container,
            orient=tk.VERTICAL,
            command=self.history_canvas.yview
        )
        self.history_scrollbar.grid(row=1, column=1, sticky="ns")
        
        # Configure canvas
        self.history_canvas.configure(yscrollcommand=self.history_scrollbar.set)
        
        # Create frame for history items
        self.history_frame = ttk.Frame(self.history_canvas, style='AlertHistory.TFrame')
        self.history_canvas.create_window(
            (0, 0),
            window=self.history_frame,
            anchor='nw',
            width=self.history_canvas.winfo_width()
        )
        
        # Bind events
        self.history_frame.bind('<Configure>', self._on_frame_configure)
        self.history_canvas.bind('<Configure>', self._on_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
    
    def show_alert(self, message, level="warning"):
        """
        Show an alert message.
        
        Args:
            message (str): Alert message to display
            level (str): Alert level ('warning' or 'critical')
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"{timestamp} - {message}"
        
        # Cancel existing timer
        if self.current_alert_timer:
            self.root.after_cancel(self.current_alert_timer)
        
        # Configure alert appearance
        if level == "critical":
            bg_color = self.config['critical_color']
            font_size = 18
            self.play_alert_sound("critical")
            self.flash_alert()
        else:
            bg_color = self.config['warning_color']
            font_size = 16
            self.play_alert_sound("warning")
        
        # Update alert
        self.alert_label.configure(
            text=full_message,
            background=bg_color,
            font=('Arial', font_size, 'bold')
        )
        self.alert_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        # Add to history
        self.add_to_history(full_message, level)
        
        # Set auto-hide timer
        self.current_alert_timer = self.root.after(
            self.config['duration'],
            self.hide_alert
        )
    
    def add_to_history(self, message, level):
        """Add an alert to the history."""
        # Create new history label
        history_label = ttk.Label(
            self.history_frame,
            text=message,
            font=('Arial', 12),
            foreground='white' if level == "warning" else '#ff9999',
            background=self.config['dark_theme']['background'],
            wraplength=self.history_canvas.winfo_width() - 20
        )
        
        # Insert at top
        history_label.pack(
            fill=tk.X,
            padx=10,
            pady=2,
            before=self.history_frame.winfo_children()[0] if self.history_frame.winfo_children() else None
        )
        
        # Add to history list
        self.alert_history.append(history_label)
        
        # Remove oldest if exceeding maximum
        if len(self.alert_history) > self.config['max_history']:
            oldest = self.alert_history.pop(0)
            oldest.destroy()
        
        # Update scroll region
        self._on_frame_configure()
        self.history_canvas.yview_moveto(0)
    
    def hide_alert(self):
        """Hide the current alert."""
        self.alert_frame.grid_remove()
        self.current_alert_timer = None
    
    def dismiss_alert(self):
        """Manually dismiss the current alert."""
        self.hide_alert()
        if self.current_alert_timer:
            self.root.after_cancel(self.current_alert_timer)
    
    def flash_alert(self):
        """Flash the alert for critical notifications."""
        if not self.alert_frame.winfo_viewable():
            return
        
        current_bg = self.alert_label.cget('background')
        new_bg = '#ffffff' if current_bg == self.config['critical_color'] else self.config['critical_color']
        self.alert_label.configure(background=new_bg)
        
        self.root.after(self.config['flash_interval'], self.flash_alert)
    
    def play_alert_sound(self, level="warning"):
        """Play alert sound based on severity level."""
        if platform.system() == "Windows":
            import winsound
            frequency = self.config['critical_frequency'] if level == "critical" else self.config['warning_frequency']
            duration = self.config['critical_duration'] if level == "critical" else self.config['warning_duration']
            winsound.Beep(frequency, duration)
    
    def _on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame."""
        self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """When canvas is resized, resize the inner frame to match."""
        self.history_canvas.itemconfig('window', width=event.width)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.history_canvas.yview_scroll(int(-1*(event.delta/120)), "units") 