"""
Setup Models Script
=================
Script to set up all models in the correct locations.
This script will:
1. Create the models directory in the babymonitor package if it doesn't exist
2. Copy models from src/models to src/babymonitor/models
3. Create the emotion model if it doesn't exist
4. Copy all emotion model files from models/emotion directory
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
    global_models_dir = project_root / "models"
    
    # Create babymonitor models directory if it doesn't exist
    babymonitor_models_dir.mkdir(exist_ok=True)
    logger.info(f"Created models directory at {babymonitor_models_dir}")
    
    # Copy YOLOv8 model
    yolo_model_name = "yolov8n.pt"
    yolo_src_path = src_models_dir / yolo_model_name
    yolo_dest_path = babymonitor_models_dir / yolo_model_name
    
    # Also check global models directory
    if not yolo_src_path.exists() and (global_models_dir / yolo_model_name).exists():
        yolo_src_path = global_models_dir / yolo_model_name
    
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
    
    if not lightweight_src_path.exists() and (global_models_dir / lightweight_model_name).exists():
        lightweight_src_path = global_models_dir / lightweight_model_name
    
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
    
    if not labels_src_path.exists() and (global_models_dir / labels_name).exists():
        labels_src_path = global_models_dir / labels_name
    
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
    
    # NEW CODE: Copy all emotion models from models/emotion directory
    emotion_models_dir = global_models_dir / "emotion"
    if emotion_models_dir.exists():
        logger.info(f"Copying emotion models from {emotion_models_dir}")
        
        # Create emotion directory in babymonitor models if it doesn't exist
        emotion_dest_dir = babymonitor_models_dir / "emotion"
        emotion_dest_dir.mkdir(exist_ok=True)
        
        # Copy all files and directories from models/emotion
        copy_directory_recursively(emotion_models_dir, emotion_dest_dir)
        
        logger.info(f"Copied all emotion models to {emotion_dest_dir}")
    else:
        logger.warning(f"Emotion models directory not found at {emotion_models_dir}")
        
    logger.info("Model setup complete")

def copy_directory_recursively(src_dir, dest_dir):
    """Copy a directory and all its contents recursively."""
    dest_dir.mkdir(exist_ok=True)
    
    # Log the directories we're copying
    logger.info(f"Copying directory from {src_dir} to {dest_dir}")
    
    for item in src_dir.iterdir():
        if item.is_file():
            # Copy file if it doesn't exist or is newer
            dest_file = dest_dir / item.name
            if not dest_file.exists() or item.stat().st_mtime > dest_file.stat().st_mtime:
                logger.info(f"Copying file {item.name}")
                shutil.copy2(item, dest_file)
            else:
                logger.info(f"File {item.name} already exists and is up to date")
        elif item.is_dir():
            # Recursively copy subdirectory
            copy_directory_recursively(item, dest_dir / item.name)

def download_emotion_models(emotion_dir=None):
    """Download emotion detection models."""
    if emotion_dir is None:
        # Get paths
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent
        emotion_dir = project_root / "models" / "emotion"
    else:
        emotion_dir = Path(emotion_dir)
    
    emotion_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories for different emotion models
    for subdir in ["speechbrain", "emotion2", "cry_detection", "models"]:
        (emotion_dir / subdir).mkdir(exist_ok=True)
    
    logger.info(f"Created emotion model directories in {emotion_dir}")
    
    # Download emotion models if needed
    # Here you would add code to download models from a repository or storage
    
    logger.info("Emotion model setup complete")

if __name__ == "__main__":
    setup_models() 