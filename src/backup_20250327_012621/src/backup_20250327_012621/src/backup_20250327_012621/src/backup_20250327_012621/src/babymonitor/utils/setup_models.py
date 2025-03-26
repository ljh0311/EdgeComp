"""
Setup Models Script
=================
Script to set up all models in the correct locations.
This script will:
1. Create the models directory in the babymonitor package if it doesn't exist
2. Copy models from src/models to src/babymonitor/models
3. Create the emotion model if it doesn't exist
"""

import os
import shutil
import logging
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_models():
    """Set up all models in the correct locations."""
    # Get paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    src_models_dir = project_root / "src" / "models"
    babymonitor_models_dir = current_dir.parent / "models"
    
    # Create babymonitor models directory if it doesn't exist
    babymonitor_models_dir.mkdir(exist_ok=True)
    logger.info(f"Created models directory at {babymonitor_models_dir}")
    
    # Copy YOLOv8 model
    yolo_model_name = "yolov8n.pt"
    yolo_src_path = src_models_dir / yolo_model_name
    yolo_dest_path = babymonitor_models_dir / yolo_model_name
    
    if yolo_src_path.exists() and not yolo_dest_path.exists():
        logger.info(f"Copying {yolo_model_name} from {yolo_src_path} to {yolo_dest_path}")
        shutil.copy(yolo_src_path, yolo_dest_path)
    elif yolo_dest_path.exists():
        logger.info(f"{yolo_model_name} already exists at {yolo_dest_path}")
    else:
        logger.warning(f"{yolo_model_name} not found at {yolo_src_path}")
        
    # Copy lightweight model and labels if they exist
    lightweight_model_name = "person_detection_model.tflite"
    lightweight_src_path = src_models_dir / lightweight_model_name
    lightweight_dest_path = babymonitor_models_dir / lightweight_model_name
    
    if lightweight_src_path.exists() and not lightweight_dest_path.exists():
        logger.info(f"Copying {lightweight_model_name} from {lightweight_src_path} to {lightweight_dest_path}")
        shutil.copy(lightweight_src_path, lightweight_dest_path)
    elif lightweight_dest_path.exists():
        logger.info(f"{lightweight_model_name} already exists at {lightweight_dest_path}")
    else:
        logger.warning(f"{lightweight_model_name} not found at {lightweight_src_path}")
        
    # Copy labels
    labels_name = "person_labels.txt"
    labels_src_path = src_models_dir / labels_name
    labels_dest_path = babymonitor_models_dir / labels_name
    
    if labels_src_path.exists() and not labels_dest_path.exists():
        logger.info(f"Copying {labels_name} from {labels_src_path} to {labels_dest_path}")
        shutil.copy(labels_src_path, labels_dest_path)
    elif labels_dest_path.exists():
        logger.info(f"{labels_name} already exists at {labels_dest_path}")
    else:
        logger.warning(f"{labels_name} not found at {labels_src_path}")
        
    # Create emotion model if it doesn't exist
    emotion_model_name = "emotion_model.pt"
    emotion_dest_path = babymonitor_models_dir / emotion_model_name
    
    if not emotion_dest_path.exists():
        # Try to find it in the emotion directory
        emotion_src_path = current_dir.parent / "emotion" / "models" / emotion_model_name
        if emotion_src_path.exists():
            logger.info(f"Copying {emotion_model_name} from {emotion_src_path} to {emotion_dest_path}")
            shutil.copy(emotion_src_path, emotion_dest_path)
        else:
            # Create the emotion model
            logger.info(f"Creating {emotion_model_name} at {emotion_dest_path}")
            try:
                from ..utils.create_emotion_model import create_model
                
                model = create_model()
                model.eval()
                
                torch.save({
                    'state_dict': model.state_dict(),
                    'config': {
                        'input_size': 768,
                        'hidden_size': 256,
                        'num_emotions': 4
                    }
                }, emotion_dest_path)
                
                logger.info(f"Created {emotion_model_name} at {emotion_dest_path}")
            except Exception as e:
                logger.error(f"Failed to create emotion model: {e}")
    else:
        logger.info(f"{emotion_model_name} already exists at {emotion_dest_path}")
        
    logger.info("Model setup complete")

if __name__ == "__main__":
    setup_models() 