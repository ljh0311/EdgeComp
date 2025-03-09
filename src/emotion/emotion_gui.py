"""
Simple GUI for testing emotion recognition.
"""

import tkinter as tk
from tkinter import ttk
import logging
from emotion_recognizer import EmotionRecognizer

class EmotionRecognizerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Recognizer")
        self.root.geometry("400x300")
        
        # Initialize emotion recognizer
        self.recognizer = EmotionRecognizer(web_app=self)
        
        # Create GUI elements
        self.setup_gui()
        
        # Initialize state
        self.is_running = False
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Emotion Recognition", font=('Helvetica', 16))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Current emotion display
        ttk.Label(main_frame, text="Current Emotion:").grid(row=1, column=0, pady=5)
        self.emotion_label = ttk.Label(main_frame, text="Not detecting", font=('Helvetica', 12))
        self.emotion_label.grid(row=1, column=1, pady=5)
        
        # Confidence display
        ttk.Label(main_frame, text="Confidence:").grid(row=2, column=0, pady=5)
        self.confidence_label = ttk.Label(main_frame, text="0%", font=('Helvetica', 12))
        self.confidence_label.grid(row=2, column=1, pady=5)
        
        # Start/Stop button
        self.control_button = ttk.Button(main_frame, text="Start", command=self.toggle_recognition)
        self.control_button.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Status: Stopped", font=('Helvetica', 10))
        self.status_label.grid(row=4, column=0, columnspan=2, pady=5)
        
    def toggle_recognition(self):
        if not self.is_running:
            try:
                self.recognizer.start()
                self.is_running = True
                self.control_button.config(text="Stop")
                self.status_label.config(text="Status: Running")
            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}")
        else:
            self.recognizer.stop()
            self.is_running = False
            self.control_button.config(text="Start")
            self.status_label.config(text="Status: Stopped")
            self.emotion_label.config(text="Not detecting")
            self.confidence_label.config(text="0%")
    
    def emit_emotion(self, emotion, confidence):
        """Called by the emotion recognizer when a new emotion is detected"""
        self.emotion_label.config(text=emotion.capitalize())
        self.confidence_label.config(text=f"{confidence*100:.1f}%")
        
    def run(self):
        try:
            self.root.mainloop()
        finally:
            if self.is_running:
                self.recognizer.stop()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run GUI
    app = EmotionRecognizerGUI()
    app.run() 