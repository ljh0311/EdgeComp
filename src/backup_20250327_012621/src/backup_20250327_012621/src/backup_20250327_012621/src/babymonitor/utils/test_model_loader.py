"""
Test Model Loader
---------------
This script demonstrates the functionality of the ModelLoader class.
It loads models asynchronously and shows how to use the model loader.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run the model loader test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Model Loader')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model to load')
    parser.add_argument('--type', type=str, default='yolov8', help='Model type (yolov8, lightweight, emotion)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Import modules
    from src.babymonitor.utils.model_loader import ModelLoader
    from src.babymonitor.utils.model_manager import ModelManager
    
    # Initialize model loader
    model_loader = ModelLoader(max_workers=2)
    model_loader.start()
    
    logger.info(f"Loading model: {args.model} of type: {args.type}")
    
    # Define callback function
    def model_loaded_callback(model_id, model, error):
        if error:
            logger.error(f"Error loading model {model_id}: {error}")
        else:
            logger.info(f"Model {model_id} loaded successfully")
            
            # Print model information
            if args.type == 'yolov8':
                logger.info(f"Model type: {type(model)}")
                logger.info(f"Model size: {model.model.yaml['yaml_file']}")
            elif args.type == 'lightweight':
                logger.info(f"Model type: {type(model)}")
                logger.info(f"Model resolution: {model.resolution}")
            elif args.type == 'emotion':
                logger.info(f"Model type: {type(model)}")
                
    # Load model asynchronously
    model_id = f"{args.type}_{args.model}"
    
    # Resolve model path
    try:
        model_path = str(ModelManager.get_model_path(args.model))
    except FileNotFoundError:
        model_path = args.model
        logger.warning(f"Model {args.model} not found in standard locations, using as is")
    
    # Configure model based on type
    config = {}
    if args.type == 'yolov8':
        config = {
            'threshold': 0.5,
            'force_cpu': False
        }
    elif args.type == 'lightweight':
        config = {
            'label_path': str(Path(model_path).parent / 'person_labels.txt'),
            'threshold': 0.5,
            'resolution': (320, 320),
            'num_threads': 4
        }
    elif args.type == 'emotion':
        config = {
            'device': 'cpu'
        }
    
    # Queue the model for loading
    model_loader.load_model(
        model_id=model_id,
        model_type=args.type,
        model_path=model_path,
        config=config,
        callback=model_loaded_callback
    )
    
    # Wait for model to load
    logger.info("Waiting for model to load...")
    
    # Monitor loading status
    try:
        while True:
            status = model_loader.get_loading_status(model_id)
            logger.info(f"Loading status: {status['status']}, progress: {status['progress']}%")
            
            if status['status'] == 'loaded':
                logger.info("Model loaded successfully!")
                
                # Get the loaded model
                model = model_loader.get_model(model_id)
                logger.info(f"Model: {model}")
                
                # Unload the model
                logger.info("Unloading model...")
                model_loader.unload_model(model_id)
                
                break
            elif status['status'] == 'error':
                logger.error(f"Error loading model: {status['error']}")
                break
                
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Clean up
        logger.info("Cleaning up...")
        model_loader.cleanup()
        
if __name__ == "__main__":
    main() 