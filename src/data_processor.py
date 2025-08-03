import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataProcessor:
    """Handle data preprocessing and augmentation"""
    
    def __init__(self, config):
        self.config = config
        
    def create_data_generators(self, data_dir):
        """Create training and validation data generators"""
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def preprocess_image(self, image):
        """Preprocess a single image for prediction"""
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize image
        image = image.resize(self.config.IMG_SIZE)
        
        # Convert to array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array