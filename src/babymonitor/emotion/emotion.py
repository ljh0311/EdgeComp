"""
Emotion Recognition Module
========================
A comprehensive module for real-time emotion recognition using HuBERT model.
Includes both GUI and console interfaces for testing.

Usage:
    from babymonitor.emotion.emotion import EmotionRecognizer, EmotionGUI
    # Or run directly:
    python -m babymonitor.emotion.emotion [--console]
"""

import os
import queue
import threading
import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionRecognizer:
    """Real-time emotion recognition system using HuBERT"""

    # Map our model's emotions to UI emotions
    EMOTION_MAPPING = {
        "natural": "neutral",
        "anger": "angry",
        "worried": "fear",
        "happy": "happy",
        "fear": "fear",
        "sadness": "sad"
    }

    # Default emotions for UI
    UI_EMOTIONS = ['happy', 'neutral', 'sad']

    def __init__(self, model_path=None):
        """Initialize the emotion recognition system"""
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.web_app = None
        self.monitor_system = None
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.block_size = 4000
        
        # Initialize state
        self.running = False
        self.audio_queue = queue.Queue()
        self.stream = None
        
        try:
            # Initialize model
            if model_path is None:
                model_paths = [
                    Path(__file__).resolve().parent.parent.parent / "models" / "hubert-base-ls960_emotion.pt",
                    Path(__file__).resolve().parent.parent.parent / "models" / "best_emotion_model.pt"
                ]
                for path in model_paths:
                    if path.exists():
                        model_path = str(path)
                        break
                else:
                    self.logger.warning("No model found in standard locations, using base model")
                    model_path = None
            
            self._load_model(model_path)
            self.logger.info("Emotion recognizer initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing emotion recognizer: {str(e)}")
            raise

    def _load_model(self, model_path):
        """Load the HuBERT-based emotion recognition model"""
        try:
            # Load HuBERT model and processor
            model_name = "facebook/hubert-base-ls960"
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = HubertModel.from_pretrained(model_name)
            
            # Add classification head
            self.classifier = torch.nn.Linear(768, len(self.EMOTION_MAPPING))
            
            # Load trained weights if available
            if model_path and os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict['hubert'])
                self.classifier.load_state_dict(state_dict['classifier'])
                self.logger.info(f"Loaded HuBERT model from {model_path}")
            
            self.model = self.model.to(self.device)
            self.classifier = self.classifier.to(self.device)
            self.model.eval()
            self.classifier.eval()
            
        except Exception as e:
            self.logger.error(f"Error loading emotion model: {str(e)}")
            raise

    def set_monitor_system(self, system):
        """Set reference to the main monitor system"""
        self.monitor_system = system

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream processing"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        # Normalize and convert to float32
        audio_data = indata.copy()
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        self.audio_queue.put(audio_data)

    def process_audio(self):
        """Process audio data and detect emotions"""
        while self.running:
            try:
                # Get audio data
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Process audio for HuBERT
                features = audio_data.flatten()
                inputs = self.processor(
                    features,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True
                )
                input_values = inputs.input_values.to(self.device)
                
                with torch.no_grad():
                    # Get HuBERT features
                    outputs = self.model(input_values=input_values)
                    # Use mean pooling over time
                    pooled = torch.mean(outputs.last_hidden_state, dim=1)
                    # Get emotion logits
                    logits = self.classifier(pooled)
                    probs = F.softmax(logits, dim=1)[0]
                
                # Create emotion dictionary for UI
                emotions = {emotion: 0.0 for emotion in self.UI_EMOTIONS}
                
                # Map model emotions to UI emotions
                for model_emotion, ui_emotion in self.EMOTION_MAPPING.items():
                    if ui_emotion in self.UI_EMOTIONS:  # Only process emotions we show in UI
                        emotion_idx = list(self.EMOTION_MAPPING.keys()).index(model_emotion)
                        confidence = probs[emotion_idx].item()
                        emotions[ui_emotion] = max(emotions.get(ui_emotion, 0), confidence)
                
                # Get the most confident emotion
                max_emotion = max(emotions.items(), key=lambda x: x[1])
                
                # Update UI if available
                if self.monitor_system is not None:
                    self.monitor_system.update_emotion_ui(emotions)
                
                # Emit result if web app is available
                if self.web_app is not None:
                    try:
                        self.web_app.emit_emotion(max_emotion[0], max_emotion[1])
                    except Exception as e:
                        self.logger.error(f"Error emitting emotion: {str(e)}")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing audio: {str(e)}")
                continue

    def start(self):
        """Start emotion recognition"""
        if self.running:
            return

        try:
            self.running = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.block_size,
                callback=self.audio_callback
            )
            self.stream.start()
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self.process_audio)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            self.logger.info("Emotion recognition started")
            
        except Exception as e:
            self.logger.error(f"Error starting emotion recognition: {str(e)}")
            self.running = False
            raise

    def stop(self):
        """Stop emotion recognition"""
        if not self.running:
            return

        self.running = False
        
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1.0)
            
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        self.logger.info("Emotion recognition stopped")


class EmotionGUI:
    """GUI for testing emotion recognition"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Recognizer")
        self.root.geometry("500x400")
        
        # Define emotion colors
        self.emotion_colors = {
            "natural": "#90EE90",  # Light green
            "anger": "#FF6B6B",    # Red
            "worried": "#FFD93D",  # Yellow
            "happy": "#98FB98",    # Pale green
            "fear": "#FF69B4",     # Pink
            "sadness": "#87CEEB"   # Sky blue
        }
        
        # Initialize emotion recognizer
        self.recognizer = EmotionRecognizer()
        self.recognizer.web_app = self
        
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
        
        # Create emotion display frame
        emotion_frame = ttk.Frame(main_frame)
        emotion_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Current emotion display
        ttk.Label(emotion_frame, text="Current Emotion:").grid(row=0, column=0, pady=5)
        self.emotion_label = ttk.Label(emotion_frame, text="Not detecting", font=('Helvetica', 12))
        self.emotion_label.grid(row=0, column=1, pady=5)
        
        # Confidence display
        ttk.Label(emotion_frame, text="Confidence:").grid(row=1, column=0, pady=5)
        self.confidence_label = ttk.Label(emotion_frame, text="0%", font=('Helvetica', 12))
        self.confidence_label.grid(row=1, column=1, pady=5)
        
        # Create emotion indicators
        self.emotion_indicators = {}
        indicator_frame = ttk.Frame(main_frame)
        indicator_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Create indicator for each emotion
        for i, emotion in enumerate(self.emotion_colors.keys()):
            row = i // 2
            col = i % 2
            
            # Create frame for this emotion
            emotion_box = ttk.Frame(indicator_frame, style=f"{emotion}.TFrame")
            emotion_box.grid(row=row, column=col, padx=10, pady=5)
            
            # Create colored indicator
            indicator = tk.Label(emotion_box, width=2, height=1)
            indicator.pack(side=tk.LEFT, padx=5)
            indicator.configure(bg=self.emotion_colors[emotion])
            
            # Create label
            ttk.Label(emotion_box, text=emotion.capitalize()).pack(side=tk.LEFT, padx=5)
            
            self.emotion_indicators[emotion] = indicator
        
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
            
            # Reset all indicators
            for indicator in self.emotion_indicators.values():
                indicator.configure(bg='lightgray')
    
    def emit_emotion(self, emotion, confidence):
        """Called by the emotion recognizer when a new emotion is detected"""
        self.emotion_label.config(text=emotion.capitalize())
        self.confidence_label.config(text=f"{confidence*100:.1f}%")
        
        # Update indicators
        for e, indicator in self.emotion_indicators.items():
            if e == emotion:
                indicator.configure(bg=self.emotion_colors[e])
            else:
                indicator.configure(bg='lightgray')
        
    def run(self):
        try:
            self.root.mainloop()
        finally:
            if self.is_running:
                self.recognizer.stop()


class ConsoleInterface:
    """Console interface for testing emotion recognition"""
    
    def __init__(self):
        self.recognizer = EmotionRecognizer()
        self.recognizer.web_app = self
        self.running = False
        
    def emit_emotion(self, emotion, confidence):
        """Called by the emotion recognizer when a new emotion is detected"""
        # Clear the current line
        sys.stdout.write('\r' + ' ' * 50 + '\r')
        # Print the emotion and confidence
        sys.stdout.write(f"Detected: {emotion.capitalize()} ({confidence*100:.1f}%)")
        sys.stdout.flush()
    
    def run(self):
        """Run the console interface"""
        try:
            print("Starting emotion recognition...")
            print("Press Ctrl+C to stop")
            self.recognizer.start()
            self.running = True
            
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping emotion recognition...")
        finally:
            if self.running:
                self.recognizer.stop()
                self.running = False


def main():
    """Main entry point for testing emotion recognition"""
    parser = argparse.ArgumentParser(description="Test emotion recognition")
    parser.add_argument(
        "--console",
        action="store_true",
        help="Use console interface instead of GUI"
    )
    
    args = parser.parse_args()
    
    try:
        if args.console:
            interface = ConsoleInterface()
        else:
            interface = EmotionGUI()
        interface.run()
    except Exception as e:
        logger.error(f"Error running emotion recognition: {str(e)}")
        raise

if __name__ == "__main__":
    main() 