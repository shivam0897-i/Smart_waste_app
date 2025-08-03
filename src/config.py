class Config:
    """Configuration class for the waste classification project"""
    
    # Model parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_CLASSES = 6
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Class names
    CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # Paths
    MODEL_PATH = 'models/improved_waste_classifier.h5'
    HISTORY_PATH = 'models/improved_training_history.pkl'
    
    # Colors for visualization
    CLASS_COLORS = {
        'cardboard': '#8B4513',
        'glass': '#87CEEB',
        'metal': '#C0C0C0',
        'paper': '#F5F5DC',
        'plastic': '#FF6347',
        'trash': '#696969'
    }