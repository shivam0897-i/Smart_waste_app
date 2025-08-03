import os
import logging
import numpy as np
from PIL import Image
import json

def create_project_structure():
    """Create the project directory structure"""
    directories = [
        'data/train',
        'data/test',
        'models',
        'notebooks',
        'tests',
        'assets'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def setup_logging():
    """Set up logging configuration"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger(__name__)

def save_predictions(predictions, filepath):
    """Save predictions to a JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to {filepath}")
    except Exception as e:
        print(f"Error saving predictions: {e}")

def load_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for prediction"""
    try:
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None