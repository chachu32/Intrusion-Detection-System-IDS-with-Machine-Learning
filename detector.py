import tensorflow as tf
import numpy as np

class IntrusionDetector:
    def __init__(self):
        pass
    
    def detect(self, model, X, model_type='random_forest'):
        """Detect intrusions using the trained model"""
        if model_type == 'random_forest':
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]
        else:  # neural network
            probabilities = model.predict(X).flatten()
            predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    
    def load_neural_network(self, model_path):
        """Load a trained neural network model"""
        return tf.keras.models.load_model(model_path)