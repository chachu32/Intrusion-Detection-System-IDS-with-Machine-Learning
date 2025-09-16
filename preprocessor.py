from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def preprocess_data(self, df, training=True):
        """Preprocess the data for training or prediction"""
        # Separate features and target
        X = df.drop('label', axis=1) if 'label' in df.columns else df
        y = df['label'] if 'label' in df.columns else None
        
        # Scale features
        if training:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise Exception("Scaler not fitted. Please preprocess training data first.")
            X_scaled = self.scaler.transform(X)
        
        # Split data if training
        if training and y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            return X_train, X_test, y_train, y_test
        else:
            return X_scaled, None, None, None