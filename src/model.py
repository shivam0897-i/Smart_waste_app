import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle
import logging

logger = logging.getLogger(__name__)

class WasteClassifier:
    """Main classifier class using transfer learning"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the classification model using ResNet50"""
        
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.IMG_SIZE, 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom classification head
        inputs = base_model.input
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        logger.info("Model built successfully")
        return self.model
    
    def train(self, train_generator, val_generator):
        """Train the model"""
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                self.config.MODEL_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        logger.info("Starting training...")
        self.history = self.model.fit(
            train_generator,
            epochs=self.config.EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        with open(self.config.HISTORY_PATH, 'wb') as f:
            pickle.dump(self.history.history, f)
            
        logger.info("Training completed successfully")
        return self.history
    
    def load_model(self):
        """Load a trained model"""
        if os.path.exists(self.config.MODEL_PATH):
            try:
                self.model = load_model(self.config.MODEL_PATH)
                logger.info(f"Model loaded successfully from {self.config.MODEL_PATH}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise Exception(f"Failed to load model from {self.config.MODEL_PATH}: {str(e)}")
        else:
            error_msg = f"No trained model found at {self.config.MODEL_PATH}. Please train the model first."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
    def predict(self, image):
        """Make prediction on a single image"""
        if self.model is None:
            self.load_model()
            
        if self.model is None:
            raise Exception("Model failed to load. Please check if the model file exists and is valid.")
            
        # Preprocess image - import here to avoid circular imports
        from .data_processor import DataProcessor
        processor = DataProcessor(self.config)
        processed_image = processor.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.config.CLASS_NAMES[predicted_class_idx]
        
        return predicted_class, confidence, predictions