import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

class Visualizer:
    """Handle all visualization tasks"""
    
    def __init__(self, config):
        self.config = config
        
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.config.CLASS_NAMES,
                   yticklabels=self.config.CLASS_NAMES, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        return fig
    
    def plot_prediction_confidence(self, predictions, class_names):
        """Plot prediction confidence as bar chart"""
        fig = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=predictions,
                marker_color=[self.config.CLASS_COLORS.get(name, '#1f77b4') 
                             for name in class_names]
            )
        ])
        
        fig.update_layout(
            title='Prediction Confidence',
            xaxis_title='Waste Categories',
            yaxis_title='Confidence Score',
            height=400
        )
        
        return fig