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
    def __init__(self, root, config, alert_container):
        """
        Initialize the alert manager.
        
        Args:
            root: Tkinter root window
            config: Alert configuration dictionary
            alert_container: Frame to contain alerts
        """
        self.logger = logging.getLogger(__name__)
        self.root = root
        self.config = config
        self.current_alert_timer = None
        self.alert_history = []
        
        # Use the provided alert container
        self.alert_container = alert_container
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup alert UI components with improved styling."""
        # Style configuration
        style = ttk.Style()
        style.configure('Alert.TFrame', 
            background=self.config['dark_theme']['background']
        )
        style.configure('Critical.TFrame',
            background=self.config['critical_color']
        )
        style.configure('Warning.TFrame',
            background=self.config['warning_color']
        )
        style.configure('Alert.TButton',
            padding=5,
            font=('Segoe UI', 11, 'bold')
        )
        
        # Configure alert container grid
        self.alert_container.grid_rowconfigure(1, weight=1)  # History section expands
        self.alert_container.grid_columnconfigure(0, weight=1)
        
        # Current alert section
        self.current_alert_frame = ttk.Frame(self.alert_container, style='Alert.TFrame')
        self.current_alert_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.current_alert_frame.grid_remove()  # Hidden by default
        
        # Configure current alert frame grid
        self.current_alert_frame.grid_columnconfigure(0, weight=1)
        
        self.alert_label = ttk.Label(
            self.current_alert_frame,
            text="",
            font=('Segoe UI', 12, 'bold'),
            wraplength=300,
            justify=tk.LEFT,
            padding=(15, 10)
        )
        self.alert_label.grid(row=0, column=0, sticky="ew")
        
        self.dismiss_button = ttk.Button(
            self.current_alert_frame,
            text="✖ Dismiss",
            command=self.dismiss_alert,
            style='Alert.TButton'
        )
        self.dismiss_button.grid(row=1, column=0, pady=(0, 5))
        
        # History section container
        self.history_section = ttk.Frame(self.alert_container)
        self.history_section.grid(row=1, column=0, sticky="nsew")
        self.history_section.grid_columnconfigure(0, weight=1)
        self.history_section.grid_rowconfigure(1, weight=1)
        
        # History header
        header_frame = ttk.Frame(self.history_section)
        header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        header_frame.grid_columnconfigure(0, weight=1)
        
        # History title with icon
        self.history_title = ttk.Label(
            header_frame,
            text="🔔 Alert History",
            font=('Segoe UI', 14, 'bold'),
            foreground=self.config['dark_theme']['accent_text'],
            background=self.config['dark_theme']['background']
        )
        self.history_title.grid(row=0, column=0, sticky="w")
        
        # Clear history button
        self.clear_history_button = ttk.Button(
            header_frame,
            text="Clear All",
            command=self.clear_history,
            style='Alert.TButton'
        )
        self.clear_history_button.grid(row=0, column=1, padx=5)
        
        # Create scrollable frame for history items
        self.history_canvas = tk.Canvas(
            self.history_section,
            background=self.config['dark_theme']['background'],
            highlightthickness=0
        )
        self.history_canvas.grid(row=1, column=0, sticky="nsew", padx=5)
        
        # Add scrollbar
        self.scrollbar = ttk.Scrollbar(
            self.history_section,
            orient=tk.VERTICAL,
            command=self.history_canvas.yview
        )
        self.scrollbar.grid(row=1, column=1, sticky="ns")
        
        # Configure canvas
        self.history_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Create frame for history items
        self.history_items_frame = ttk.Frame(
            self.history_canvas,
            style='Alert.TFrame'
        )
        
        # Create window in canvas
        self.history_canvas.create_window(
            (0, 0),
            window=self.history_items_frame,
            anchor='nw',
            width=self.history_canvas.winfo_width()
        )
        
        # Configure history items frame
        self.history_items_frame.grid_columnconfigure(0, weight=1)
        
        # Bind events
        self.history_items_frame.bind('<Configure>', self._on_frame_configure)
        self.history_canvas.bind('<Configure>', self._on_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
    
    def show_alert(self, message, level="warning"):
        """Show an alert with improved visibility."""
        # Cancel existing timer
        if self.current_alert_timer:
            self.root.after_cancel(self.current_alert_timer)
        
        # Configure alert appearance
        style = 'Critical.TFrame' if level == "critical" else 'Warning.TFrame'
        self.current_alert_frame.configure(style=style)
        
        # Format message with timestamp and icon
        timestamp = datetime.now().strftime("%H:%M:%S")
        icon = "🔴" if level == "critical" else "⚠️"
        full_message = f"{timestamp} {icon} {message}"
        
        self.alert_label.configure(
            text=full_message,
            foreground='white',
            background=self.config['critical_color'] if level == "critical" else self.config['warning_color']
        )
        
        # Show alert
        self.current_alert_frame.grid()
        
        # Play sound and flash if critical
        if level == "critical":
            self.play_alert_sound("critical")
            self.flash_alert()
        else:
            self.play_alert_sound("warning")
        
        # Add to history
        self.add_to_history(message, level)
        
        # Set auto-hide timer
        self.current_alert_timer = self.root.after(
            self.config['duration'],
            self.hide_alert
        )
    
    def add_to_history(self, message, level="warning"):
        """Add an alert to history with improved styling."""
        try:
            # Create container for history item
            item_frame = ttk.Frame(
                self.history_items_frame,
                style='Critical.TFrame' if level == "critical" else 'Warning.TFrame'
            )
            item_frame.grid(row=len(self.alert_history), column=0, sticky="ew", padx=5, pady=2)
            item_frame.grid_columnconfigure(0, weight=1)
            
            # Create label with timestamp and criticality indicator
            timestamp = datetime.now().strftime("%H:%M:%S")
            icon = "🔴" if level == "critical" else "⚠️"
            label = ttk.Label(
                item_frame,
                text=f"{timestamp} {icon} {message}",
                font=('Segoe UI', 11),
                foreground='white',
                background=self.config['critical_color'] if level == "critical" else self.config['warning_color'],
                wraplength=self.history_canvas.winfo_width() - 20,
                justify=tk.LEFT,
                padding=(10, 5)
            )
            label.grid(row=0, column=0, sticky="ew")
            
            # Add separator
            ttk.Separator(self.history_items_frame).grid(
                row=len(self.alert_history) + 1,
                column=0,
                sticky="ew",
                padx=5,
                pady=(2, 0)
            )
            
            # Keep track of history items
            self.alert_history.append((item_frame, label))
            
            # Remove old alerts if exceeding maximum
            while len(self.alert_history) > self.config['max_history']:
                old_frame, _ = self.alert_history.pop(0)
                old_frame.destroy()
            
            # Ensure newest alert is visible
            self.root.update_idletasks()
            self.history_canvas.yview_moveto(1.0)
            
            # Update scroll region
            self._on_frame_configure()
            
        except Exception as e:
            self.logger.error(f"Error adding alert to history: {str(e)}")
    
    def clear_history(self):
        """Clear all alerts from history."""
        for frame, _ in self.alert_history:
            frame.destroy()
        self.alert_history.clear()
    
    def hide_alert(self):
        """Hide the current alert."""
        self.current_alert_frame.grid_remove()  # Use grid_remove instead of pack_forget
        self.current_alert_timer = None
    
    def dismiss_alert(self):
        """Manually dismiss the current alert."""
        self.hide_alert()
        if self.current_alert_timer:
            self.root.after_cancel(self.current_alert_timer)
    
    def flash_alert(self):
        """Flash the alert for critical notifications."""
        if not self.current_alert_frame.winfo_viewable():
            return
        
        current_style = self.current_alert_frame.cget('style')
        new_style = 'Alert.TFrame' if current_style == 'Critical.TFrame' else 'Critical.TFrame'
        self.current_alert_frame.configure(style=new_style)
        
        # Schedule next flash
        self.root.after(self.config['flash_interval'], self.flash_alert)
    
    def play_alert_sound(self, level="warning"):
        """Play alert sound based on severity level."""
        if platform.system() == "Windows":
            import winsound
            try:
                frequency = self.config['critical_frequency'] if level == "critical" else self.config['warning_frequency']
                duration = self.config['critical_duration'] if level == "critical" else self.config['warning_duration']
                winsound.Beep(frequency, duration)
            except:
                self.logger.warning("Could not play alert sound")
    
    def _on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame."""
        self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))
        # Update the width of the history frame to match the canvas
        self.history_canvas.itemconfig("window", width=self.history_canvas.winfo_width())
        
    def _on_canvas_configure(self, event):
        """When canvas is resized, resize the inner frame to match."""
        # Update the width of the history frame and all labels
        canvas_width = event.width
        self.history_canvas.itemconfig("window", width=canvas_width)
        
        # Update wraplength for all history items
        for _, label in self.alert_history:
            label.configure(wraplength=canvas_width - 20)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        if self.history_canvas.winfo_height() < self.history_section.winfo_height():
            self.history_canvas.yview_scroll(int(-1*(event.delta/120)), "units") 