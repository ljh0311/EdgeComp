"""
Model Manager
============
Utility for managing model paths and loading models.
"""

import os
import logging
import torch
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """Utility for managing model paths and loading models."""
    
    # Define standard model paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    SRC_MODELS_DIR = PROJECT_ROOT / "src" / "models"
    BABYMONITOR_MODELS_DIR = Path(__file__).parent.parent / "models"
    
    # Model file names
    YOLO_MODEL_NAME = "yolov8n.pt"
    EMOTION_MODEL_NAME = "emotion_model.pt"
    LIGHTWEIGHT_MODEL_NAME = "person_detection_model.tflite"
    LIGHTWEIGHT_LABELS_NAME = "person_labels.txt"
    
    @classmethod
    def ensure_model_directories(cls):
        """Ensure all model directories exist."""
        cls.BABYMONITOR_MODELS_DIR.mkdir(exist_ok=True)
        
    @classmethod
    def get_model_path(cls, model_name, ensure_exists=True):
        """Get the path to a model file.
        
        Args:
            model_name: Name of the model file.
            ensure_exists: If True, ensure the model exists at the returned path.
                          This may involve copying from another location.
                          
        Returns:
            Path: Path to the model file.
        """
        # First check in babymonitor models directory
        babymonitor_path = cls.BABYMONITOR_MODELS_DIR / model_name
        if babymonitor_path.exists():
            return babymonitor_path
            
        # Then check in src models directory
        src_path = cls.SRC_MODELS_DIR / model_name
        if src_path.exists():
            if ensure_exists:
                # Copy to babymonitor models directory
                cls.ensure_model_directories()
                logger.info(f"Copying model from {src_path} to {babymonitor_path}")
                shutil.copy(src_path, babymonitor_path)
                return babymonitor_path
            return src_path
            
        # Check for emotion model in emotion directory
        if model_name == cls.EMOTION_MODEL_NAME:
            emotion_path = Path(__file__).parent.parent / "emotion" / "models" / model_name
            if emotion_path.exists():
                if ensure_exists:
                    # Copy to babymonitor models directory
                    cls.ensure_model_directories()
                    logger.info(f"Copying model from {emotion_path} to {babymonitor_path}")
                    shutil.copy(emotion_path, babymonitor_path)
                    return babymonitor_path
                return emotion_path
                
        # If model doesn't exist and ensure_exists is True, raise an error
        if ensure_exists:
            raise FileNotFoundError(f"Model {model_name} not found in any standard location")
            
        # Otherwise return the default path
        return babymonitor_path
        
    @classmethod
    def get_yolo_model_path(cls):
        """Get the path to the YOLOv8 model."""
        return cls.get_model_path(cls.YOLO_MODEL_NAME)
        
    @classmethod
    def get_emotion_model_path(cls):
        """Get the path to the emotion model."""
        return cls.get_model_path(cls.EMOTION_MODEL_NAME)
        
    @classmethod
    def get_lightweight_model_path(cls):
        """Get the path to the lightweight model."""
        return cls.get_model_path(cls.LIGHTWEIGHT_MODEL_NAME)
        
    @classmethod
    def get_lightweight_labels_path(cls):
        """Get the path to the lightweight model labels."""
        return cls.get_model_path(cls.LIGHTWEIGHT_LABELS_NAME)
        
    @classmethod
    def load_emotion_model(cls, device='cpu'):
        """Load the emotion model.
        
        Args:
            device: Device to load the model on ('cpu' or 'cuda').
            
        Returns:
            torch.nn.Module: Loaded emotion model.
        """
        from ..utils.create_emotion_model import EmotionModel
        
        model_path = cls.get_emotion_model_path()
        logger.info(f"Loading emotion model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {
            'input_size': 768,
            'hidden_size': 256,
            'num_emotions': 4
        })
        
        model = EmotionModel(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_emotions=config['num_emotions']
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        return model
        
    @classmethod
    def create_and_save_emotion_model(cls):
        """Create and save the emotion model if it doesn't exist."""
        from ..utils.create_emotion_model import create_model
        
        model_path = cls.BABYMONITOR_MODELS_DIR / cls.EMOTION_MODEL_NAME
        if model_path.exists():
            logger.info(f"Emotion model already exists at {model_path}")
            return model_path
            
        # Create model directory if it doesn't exist
        cls.ensure_model_directories()
        
        # Create and save model
        logger.info(f"Creating emotion model and saving to {model_path}")
        model = create_model()
        model.eval()
        
        torch.save({
            'state_dict': model.state_dict(),
            'config': {
                'input_size': 768,
                'hidden_size': 256,
                'num_emotions': 4
            }
        }, model_path)
        
        return model_path 