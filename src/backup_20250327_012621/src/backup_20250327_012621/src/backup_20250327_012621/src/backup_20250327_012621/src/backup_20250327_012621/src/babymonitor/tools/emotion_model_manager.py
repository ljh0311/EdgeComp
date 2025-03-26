#!/usr/bin/env python
"""
Emotion Model Manager GUI
A tool for managing and testing emotion recognition models.
"""

import os
import sys
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
import soundfile as sf
import numpy as np
from speechbrain.pretrained import EncoderClassifier

class EmotionModelManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Model Manager")
        self.root.geometry("800x600")
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model paths
        self.models_dir = Path("models")
        self.emotion_dir = self.models_dir / "emotion"
        self.speechbrain_dir = self.emotion_dir / "speechbrain"
        self.cry_detection_dir = self.emotion_dir / "cry_detection"
        
        # Create directories if they don't exist
        self.emotion_dir.mkdir(parents=True, exist_ok=True)
        self.speechbrain_dir.mkdir(parents=True, exist_ok=True)
        self.cry_detection_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main container
        self.main_container = ttk.Frame(root, padding="10")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tabs
        self.create_model_list_tab()
        self.create_test_tab()
        self.create_training_tab()
        
        # Load model configurations
        self.load_model_configs()
        
        # Import existing models
        self.import_existing_models()
        
    def create_model_list_tab(self):
        """Create the model list tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Models")
        
        # Model list
        self.model_list = ttk.Treeview(frame, columns=("Type", "Status", "Last Used"), show="headings")
        self.model_list.heading("Type", text="Model Type")
        self.model_list.heading("Status", text="Status")
        self.model_list.heading("Last Used", text="Last Used")
        self.model_list.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.model_list.yview)
        scrollbar.grid(row=0, column=2, sticky=(tk.N, tk.S))
        self.model_list.configure(yscrollcommand=scrollbar.set)
        
        # Buttons
        ttk.Button(frame, text="Import Model", command=self.import_model).grid(row=1, column=0, pady=5)
        ttk.Button(frame, text="Export Model", command=self.export_model).grid(row=1, column=1, pady=5)
        ttk.Button(frame, text="Delete Model", command=self.delete_model).grid(row=1, column=2, pady=5)
        
        # Refresh button
        ttk.Button(frame, text="Refresh", command=self.refresh_model_list).grid(row=1, column=3, pady=5)
        
    def create_test_tab(self):
        """Create the model testing tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Test")
        
        # Model selection
        ttk.Label(frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W)
        self.test_model_var = tk.StringVar()
        self.test_model_combo = ttk.Combobox(frame, textvariable=self.test_model_var)
        self.test_model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Audio file selection
        ttk.Label(frame, text="Audio File:").grid(row=1, column=0, sticky=tk.W)
        self.audio_path_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.audio_path_var).grid(row=1, column=1, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Browse", command=self.browse_audio).grid(row=1, column=2, padx=5)
        
        # Test button
        ttk.Button(frame, text="Run Test", command=self.run_test).grid(row=2, column=0, columnspan=3, pady=10)
        
        # Results display
        self.results_text = tk.Text(frame, height=10, width=50)
        self.results_text.grid(row=3, column=0, columnspan=3, pady=5)
        
    def create_training_tab(self):
        """Create the model training tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Training")
        
        # Training parameters
        ttk.Label(frame, text="Training Parameters").grid(row=0, column=0, columnspan=2, pady=5)
        
        # Model type selection
        ttk.Label(frame, text="Model Type:").grid(row=1, column=0, sticky=tk.W)
        self.train_model_type = tk.StringVar(value="speechbrain")
        ttk.Radiobutton(frame, text="SpeechBrain", variable=self.train_model_type, value="speechbrain").grid(row=1, column=1, sticky=tk.W)
        ttk.Radiobutton(frame, text="Cry Detection", variable=self.train_model_type, value="cry_detection").grid(row=1, column=2, sticky=tk.W)
        
        # Dataset selection
        ttk.Label(frame, text="Dataset Path:").grid(row=2, column=0, sticky=tk.W)
        self.dataset_path_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.dataset_path_var).grid(row=2, column=1, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Browse", command=self.browse_dataset).grid(row=2, column=2, padx=5)
        
        # Training parameters
        ttk.Label(frame, text="Epochs:").grid(row=3, column=0, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="10")
        ttk.Entry(frame, textvariable=self.epochs_var).grid(row=3, column=1, sticky=tk.W)
        
        ttk.Label(frame, text="Batch Size:").grid(row=4, column=0, sticky=tk.W)
        self.batch_size_var = tk.StringVar(value="32")
        ttk.Entry(frame, textvariable=self.batch_size_var).grid(row=4, column=1, sticky=tk.W)
        
        # Start training button
        ttk.Button(frame, text="Start Training", command=self.start_training).grid(row=5, column=0, columnspan=3, pady=10)
        
        # Training progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Training log
        self.training_log = tk.Text(frame, height=10, width=50)
        self.training_log.grid(row=7, column=0, columnspan=3, pady=5)
        
    def load_model_configs(self):
        """Load model configurations from JSON files"""
        self.model_configs = {}
        
        # Load SpeechBrain models
        speechbrain_config = self.speechbrain_dir / "config.json"
        if speechbrain_config.exists():
            with open(speechbrain_config, 'r') as f:
                self.model_configs['speechbrain'] = json.load(f)
        
        # Load Cry Detection models
        cry_config = self.cry_detection_dir / "config.json"
        if cry_config.exists():
            with open(cry_config, 'r') as f:
                self.model_configs['cry_detection'] = json.load(f)
        
        self.refresh_model_list()
        
    def refresh_model_list(self):
        """Refresh the model list display"""
        try:
            self.logger.info("Refreshing model list...")
            self.model_list.delete(*self.model_list.get_children())
            
            models_found = False
            
            # Add SpeechBrain models
            for model_name, config in self.model_configs.get('speechbrain', {}).items():
                models_found = True
                self.model_list.insert('', 'end', values=(
                    f"SpeechBrain: {model_name}",
                    config.get('status', 'Unknown'),
                    config.get('last_used', 'Never')
                ))
                self.logger.info(f"Added SpeechBrain model: {model_name}")
            
            # Add Cry Detection models
            for model_name, config in self.model_configs.get('cry_detection', {}).items():
                models_found = True
                self.model_list.insert('', 'end', values=(
                    f"Cry Detection: {model_name}",
                    config.get('status', 'Unknown'),
                    config.get('last_used', 'Never')
                ))
                self.logger.info(f"Added Cry Detection model: {model_name}")
            
            # Update test model combo
            model_values = [
                f"{model_type}: {model_name}"
                for model_type, models in self.model_configs.items()
                for model_name in models.keys()
            ]
            self.test_model_combo['values'] = model_values
            
            if not models_found:
                self.logger.warning("No models found in configurations")
            
        except Exception as e:
            self.logger.error(f"Error refreshing model list: {str(e)}")
            messagebox.showerror("Error", f"Error refreshing model list: {str(e)}")
        
    def import_model(self):
        """Import a new model"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        
        if file_path:
            model_type = "speechbrain" if "hubert" in file_path.lower() else "cry_detection"
            model_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Create model directory if it doesn't exist
            model_dir = self.emotion_dir / model_type / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model file
            shutil.copy2(file_path, model_dir / os.path.basename(file_path))
            
            # Update config
            if model_type not in self.model_configs:
                self.model_configs[model_type] = {}
            
            self.model_configs[model_type][model_name] = {
                'path': str(model_dir / os.path.basename(file_path)),
                'status': 'Ready',
                'last_used': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save config
            self.save_model_configs()
            
            self.refresh_model_list()
            messagebox.showinfo("Success", f"Model {model_name} imported successfully!")
    
    def export_model(self):
        """Export a selected model"""
        selection = self.model_list.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a model to export")
            return
        
        item = self.model_list.item(selection[0])
        model_type = "speechbrain" if "SpeechBrain" in item['values'][0] else "cry_detection"
        model_name = item['values'][0].split(": ")[1]
        
        save_path = filedialog.asksaveasfilename(
            title="Save Model As",
            defaultextension=".pt",
            initialfile=f"{model_name}.pt"
        )
        
        if save_path:
            model_path = self.model_configs[model_type][model_name]['path']
            shutil.copy2(model_path, save_path)
            messagebox.showinfo("Success", "Model exported successfully!")
    
    def delete_model(self):
        """Delete a selected model"""
        selection = self.model_list.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a model to delete")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this model?"):
            item = self.model_list.item(selection[0])
            model_type = "speechbrain" if "SpeechBrain" in item['values'][0] else "cry_detection"
            model_name = item['values'][0].split(": ")[1]
            
            # Delete model directory
            model_dir = self.emotion_dir / model_type / model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Update config
            del self.model_configs[model_type][model_name]
            self.save_model_configs()
            
            self.refresh_model_list()
            messagebox.showinfo("Success", "Model deleted successfully!")
    
    def browse_audio(self):
        """Browse for audio file"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3"), ("All Files", "*.*")]
        )
        if file_path:
            self.audio_path_var.set(file_path)
    
    def browse_dataset(self):
        """Browse for dataset directory"""
        dir_path = filedialog.askdirectory(title="Select Dataset Directory")
        if dir_path:
            self.dataset_path_var.set(dir_path)
    
    def run_test(self):
        """Run emotion detection test on selected audio file"""
        if not self.test_model_var.get():
            messagebox.showwarning("Warning", "Please select a model")
            return
        
        if not self.audio_path_var.get():
            messagebox.showwarning("Warning", "Please select an audio file")
            return
        
        try:
            model_type = "speechbrain" if "SpeechBrain" in self.test_model_var.get() else "cry_detection"
            model_name = self.test_model_var.get().split(": ")[1]
            model_path = self.model_configs[model_type][model_name]['path']
            
            # Load audio file
            audio, sr = sf.read(self.audio_path_var.get())
            
            # Load model and run inference
            if model_type == "speechbrain":
                classifier = EncoderClassifier.from_hparams(
                    source=model_path,
                    savedir=f"models/emotion/speechbrain/{model_name}"
                )
                prediction = classifier.classify_batch(torch.tensor(audio))
                result = prediction[3][0]  # Get the predicted emotion
            else:
                # Load cry detection model
                model = torch.load(model_path)
                model.eval()
                with torch.no_grad():
                    audio_tensor = torch.tensor(audio).unsqueeze(0)
                    prediction = model(audio_tensor)
                    result = "Crying" if prediction.item() > 0.5 else "Not Crying"
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Model: {model_name}\n")
            self.results_text.insert(tk.END, f"Audio File: {os.path.basename(self.audio_path_var.get())}\n")
            self.results_text.insert(tk.END, f"Result: {result}\n")
            
            # Update last used time
            self.model_configs[model_type][model_name]['last_used'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.save_model_configs()
            self.refresh_model_list()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during testing: {str(e)}")
    
    def start_training(self):
        """Start model training"""
        if not self.dataset_path_var.get():
            messagebox.showwarning("Warning", "Please select a dataset directory")
            return
        
        try:
            model_type = self.train_model_type.get()
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            
            # Start training in a separate thread
            import threading
            training_thread = threading.Thread(
                target=self.train_model,
                args=(model_type, epochs, batch_size)
            )
            training_thread.start()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for epochs and batch size")
    
    def train_model(self, model_type, epochs, batch_size):
        """Train the model"""
        try:
            # Update progress bar
            self.progress_var.set(0)
            self.training_log.delete(1.0, tk.END)
            self.training_log.insert(tk.END, "Starting training...\n")
            
            # Import training script based on model type
            if model_type == "speechbrain":
                from training.custom_speechbrain import train_speechbrain_model
                train_speechbrain_model(
                    dataset_path=self.dataset_path_var.get(),
                    epochs=epochs,
                    batch_size=batch_size,
                    progress_callback=self.update_training_progress
                )
            else:
                from training.emotion_detection.train_cry_detection import train_cry_detection_model
                train_cry_detection_model(
                    dataset_path=self.dataset_path_var.get(),
                    epochs=epochs,
                    batch_size=batch_size,
                    progress_callback=self.update_training_progress
                )
            
            self.training_log.insert(tk.END, "Training completed successfully!\n")
            messagebox.showinfo("Success", "Model training completed!")
            
        except Exception as e:
            self.training_log.insert(tk.END, f"Error during training: {str(e)}\n")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def update_training_progress(self, progress, message):
        """Update training progress and log"""
        self.progress_var.set(progress)
        self.training_log.insert(tk.END, f"{message}\n")
        self.training_log.see(tk.END)
        self.root.update()
    
    def save_model_configs(self):
        """Save model configurations to JSON files"""
        # Save SpeechBrain config
        speechbrain_config = self.speechbrain_dir / "config.json"
        speechbrain_config.parent.mkdir(parents=True, exist_ok=True)
        with open(speechbrain_config, 'w') as f:
            json.dump(self.model_configs.get('speechbrain', {}), f, indent=4)
        
        # Save Cry Detection config
        cry_config = self.cry_detection_dir / "config.json"
        cry_config.parent.mkdir(parents=True, exist_ok=True)
        with open(cry_config, 'w') as f:
            json.dump(self.model_configs.get('cry_detection', {}), f, indent=4)

    def import_existing_models(self):
        """Import existing models from the models directory"""
        try:
            self.logger.info("Starting model import process...")
            
            # First run the organize_models.bat script
            organize_script = Path("src/babymonitor/tools/organize_models.bat")
            if organize_script.exists():
                self.logger.info("Running organize_models.bat...")
                os.system(str(organize_script))
            
            # Define model patterns to search for
            model_patterns = {
                'speechbrain': [
                    ('models/emotion/speechbrain/**/*.pt', '*.pt'),
                    ('models/*.pt', '*.pt')
                ],
                'cry_detection': [
                    ('models/emotion/cry_detection/**/*.pth', '*.pth'),
                    ('models/*.pth', '*.pth')
                ]
            }
            
            # Initialize configs if not exist
            if not hasattr(self, 'model_configs'):
                self.model_configs = {}
            
            # Search for models using patterns
            for model_type, patterns in model_patterns.items():
                self.logger.info(f"Searching for {model_type} models...")
                if model_type not in self.model_configs:
                    self.model_configs[model_type] = {}
                
                for glob_pattern, file_pattern in patterns:
                    # Use Path.glob to find model files
                    for model_path in Path().glob(glob_pattern):
                        try:
                            self.logger.info(f"Found model: {model_path}")
                            model_name = model_path.stem
                            
                            # Determine target directory
                            if model_type == 'speechbrain':
                                target_dir = self.speechbrain_dir / model_name
                            else:
                                target_dir = self.cry_detection_dir / model_name
                            
                            # Create target directory
                            target_dir.mkdir(parents=True, exist_ok=True)
                            target_file = target_dir / model_path.name
                            
                            # Copy file if it doesn't exist in target location
                            if not target_file.exists():
                                self.logger.info(f"Copying {model_path} to {target_file}")
                                shutil.copy2(model_path, target_file)
                            
                            # Update config
                            self.model_configs[model_type][model_name] = {
                                'path': str(target_file),
                                'status': 'Ready',
                                'last_used': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            self.logger.info(f"Added {model_name} to {model_type} config")
                        except Exception as e:
                            self.logger.error(f"Error processing model {model_path}: {str(e)}")
            
            # Save updated configs
            self.save_model_configs()
            self.logger.info("Saved model configurations")
            
            # Refresh the model list
            self.refresh_model_list()
            self.logger.info("Refreshed model list")
            
            # Log current configurations
            self.logger.info("Current model configurations:")
            self.logger.info(json.dumps(self.model_configs, indent=2))
            
            if not any(self.model_configs.values()):
                messagebox.showwarning(
                    "No Models Found",
                    "No models were found in the models directory.\n"
                    "Please use the Import Model button to add models."
                )
            else:
                messagebox.showinfo(
                    "Success",
                    f"Found {sum(len(models) for models in self.model_configs.values())} models"
                )
            
        except Exception as e:
            self.logger.error(f"Error during model import: {str(e)}")
            messagebox.showerror(
                "Error",
                f"Error importing models: {str(e)}\n"
                "Check the console for more details."
            )

def main():
    root = tk.Tk()
    app = EmotionModelManager(root)
    root.mainloop()

if __name__ == "__main__":
    main() 