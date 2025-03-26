"""
Model Loader
===========
Utility for loading models in the background to prevent memory issues and performance bottlenecks.
"""

import os
import logging
import threading
import queue
import time
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List, Tuple

from ..utils.model_manager import ModelManager

logger = logging.getLogger(__name__)

class ModelLoader:
    """Utility for loading models in the background."""
    
    def __init__(self, max_workers: int = 2):
        """Initialize the model loader.
        
        Args:
            max_workers: Maximum number of worker threads for loading models.
        """
        self.max_workers = max_workers
        self.model_queue = queue.Queue()
        self.loaded_models = {}
        self.loading_status = {}
        self.workers = []
        self.lock = threading.Lock()
        self.is_running = False
        
    def start(self):
        """Start the model loader workers."""
        if self.is_running:
            logger.warning("Model loader is already running")
            return
            
        self.is_running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_thread, daemon=True)
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"Started {self.max_workers} model loader worker threads")
        
    def stop(self):
        """Stop the model loader workers."""
        if not self.is_running:
            logger.warning("Model loader is not running")
            return
            
        self.is_running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
            
        # Clear queues
        while not self.model_queue.empty():
            try:
                self.model_queue.get_nowait()
            except queue.Empty:
                break
                
        logger.info("Stopped model loader worker threads")
        
    def _worker_thread(self):
        """Worker thread for loading models."""
        while self.is_running:
            try:
                # Get a model to load from the queue
                model_id, model_type, model_path, config, callback = self.model_queue.get(timeout=1.0)
                
                # Update status to loading
                with self.lock:
                    self.loading_status[model_id] = {
                        'status': 'loading',
                        'progress': 0,
                        'error': None
                    }
                    
                try:
                    # Load the model
                    logger.info(f"Loading model {model_id} of type {model_type} from {model_path}")
                    
                    # Update progress
                    with self.lock:
                        self.loading_status[model_id]['progress'] = 50
                        
                    # Load the model based on type
                    if model_type == 'yolov8':
                        from ultralytics import YOLO
                        model = YOLO(model_path)
                    elif model_type == 'lightweight':
                        from ..detectors.lightweight_detector import LightweightDetector
                        model = LightweightDetector(
                            model_path=model_path,
                            label_path=config.get('label_path', ''),
                            threshold=config.get('threshold', 0.5),
                            resolution=config.get('resolution', (320, 320))
                        )
                    elif model_type == 'emotion':
                        model = ModelManager.load_emotion_model(device=config.get('device', 'cpu'))
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")
                        
                    # Store the loaded model
                    with self.lock:
                        self.loaded_models[model_id] = model
                        self.loading_status[model_id] = {
                            'status': 'loaded',
                            'progress': 100,
                            'error': None
                        }
                        
                    logger.info(f"Successfully loaded model {model_id}")
                    
                    # Call the callback if provided
                    if callback:
                        callback(model_id, model, None)
                        
                except Exception as e:
                    logger.error(f"Error loading model {model_id}: {e}")
                    
                    # Update status to error
                    with self.lock:
                        self.loading_status[model_id] = {
                            'status': 'error',
                            'progress': 0,
                            'error': str(e)
                        }
                        
                    # Call the callback with error if provided
                    if callback:
                        callback(model_id, None, str(e))
                        
                finally:
                    # Mark the task as done
                    self.model_queue.task_done()
                    
            except queue.Empty:
                # No models to load, just wait
                pass
            except Exception as e:
                logger.error(f"Error in model loader worker: {e}")
                time.sleep(1.0)
                
    def load_model(self, model_id: str, model_type: str, model_path: str, 
                  config: Optional[Dict[str, Any]] = None, 
                  callback: Optional[Callable[[str, Any, Optional[str]], None]] = None):
        """Load a model in the background.
        
        Args:
            model_id: Unique identifier for the model.
            model_type: Type of model ('yolov8', 'lightweight', 'emotion').
            model_path: Path to the model file.
            config: Configuration for the model.
            callback: Callback function to call when the model is loaded.
                     The callback will be called with (model_id, model, error).
        """
        if not self.is_running:
            logger.warning("Model loader is not running, starting it now")
            self.start()
            
        # Check if model is already loaded
        with self.lock:
            if model_id in self.loaded_models:
                logger.info(f"Model {model_id} is already loaded")
                
                # Call the callback if provided
                if callback:
                    callback(model_id, self.loaded_models[model_id], None)
                    
                return
                
            # Check if model is already being loaded
            if model_id in self.loading_status and self.loading_status[model_id]['status'] == 'loading':
                logger.info(f"Model {model_id} is already being loaded")
                return
                
        # Resolve model path using ModelManager
        if not os.path.isabs(model_path):
            try:
                model_path = str(ModelManager.get_model_path(os.path.basename(model_path)))
            except FileNotFoundError:
                logger.warning(f"Model {model_path} not found in standard locations, using original path")
                
        # Initialize config if None
        if config is None:
            config = {}
            
        # Add the model to the queue
        self.model_queue.put((model_id, model_type, model_path, config, callback))
        
        # Update status to queued
        with self.lock:
            self.loading_status[model_id] = {
                'status': 'queued',
                'progress': 0,
                'error': None
            }
            
        logger.info(f"Queued model {model_id} for loading")
        
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a loaded model.
        
        Args:
            model_id: Unique identifier for the model.
            
        Returns:
            The loaded model, or None if the model is not loaded.
        """
        with self.lock:
            return self.loaded_models.get(model_id)
            
    def get_loading_status(self, model_id: str) -> Dict[str, Any]:
        """Get the loading status of a model.
        
        Args:
            model_id: Unique identifier for the model.
            
        Returns:
            Dictionary with status information.
        """
        with self.lock:
            return self.loading_status.get(model_id, {
                'status': 'unknown',
                'progress': 0,
                'error': None
            })
            
    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded.
        
        Args:
            model_id: Unique identifier for the model.
            
        Returns:
            True if the model is loaded, False otherwise.
        """
        with self.lock:
            return model_id in self.loaded_models
            
    def unload_model(self, model_id: str):
        """Unload a model.
        
        Args:
            model_id: Unique identifier for the model.
        """
        with self.lock:
            if model_id in self.loaded_models:
                # Get the model
                model = self.loaded_models[model_id]
                
                # Clean up the model if it has a cleanup method
                if hasattr(model, 'cleanup'):
                    try:
                        model.cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up model {model_id}: {e}")
                        
                # Remove the model
                del self.loaded_models[model_id]
                
                # Remove the loading status
                if model_id in self.loading_status:
                    del self.loading_status[model_id]
                    
                logger.info(f"Unloaded model {model_id}")
            else:
                logger.warning(f"Model {model_id} is not loaded")
                
    def get_all_models(self) -> List[Tuple[str, Any]]:
        """Get all loaded models.
        
        Returns:
            List of (model_id, model) tuples.
        """
        with self.lock:
            return list(self.loaded_models.items())
            
    def get_all_loading_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the loading status of all models.
        
        Returns:
            Dictionary mapping model_id to status information.
        """
        with self.lock:
            return dict(self.loading_status)
            
    def cleanup(self):
        """Clean up all resources."""
        # Stop the workers
        self.stop()
        
        # Unload all models
        with self.lock:
            model_ids = list(self.loaded_models.keys())
            
        for model_id in model_ids:
            self.unload_model(model_id)
            
        logger.info("Cleaned up model loader") 