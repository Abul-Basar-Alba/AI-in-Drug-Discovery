"""
Deep Learning Models for Drug Discovery
Neural Networks and CNN for image-based analysis
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import joblib
from pathlib import Path


class DeepLearningModels:
    """Deep learning models for drug discovery"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        self.model = None
        self.scaler = None
        self.history = None
        
    def build_mlp_model(self, input_dim: int, hidden_layers: list = [256, 128, 64], 
                        dropout_rate: float = 0.3) -> keras.Model:
        """Build Multi-Layer Perceptron model"""
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.BatchNormalization()
        ])
        
        # Add hidden layers
        for units in hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'), 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def train_mlp(self, X_train, y_train, X_val, y_val, 
                  epochs: int = 100, batch_size: int = 128) -> Dict:
        """Train MLP model"""
        print("\n=== Training Deep Neural Network ===")
        print(f"Input features: {X_train.shape[1]}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Build model
        self.model = self.build_mlp_model(X_train.shape[1])
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train
        print("\nStarting training...")
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_results = self.model.evaluate(X_val_scaled, y_val, verbose=0)
        
        metrics = {
            'model_name': 'Deep Neural Network',
            'accuracy': val_results[1],
            'auc': val_results[2],
            'precision': val_results[3],
            'recall': val_results[4],
            'loss': val_results[0]
        }
        
        print("\n=== Training Complete ===")
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation AUC: {metrics['auc']:.4f}")
        
        return metrics
    
    def build_cnn_model(self, input_shape: Tuple[int, int, int] = (128, 128, 3)) -> keras.Model:
        """Build CNN model for molecular structure images"""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # Output
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train_cnn(self, X_train, y_train, X_val, y_val, 
                  epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train CNN model on molecular images"""
        print("\n=== Training Convolutional Neural Network ===")
        print(f"Image shape: {X_train.shape[1:]}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        
        # Build model
        self.model = self.build_cnn_model(X_train.shape[1:])
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Data augmentation
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        # Train
        print("\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_results = self.model.evaluate(X_val, y_val, verbose=0)
        
        metrics = {
            'model_name': 'CNN',
            'accuracy': val_results[1],
            'auc': val_results[2],
            'loss': val_results[0]
        }
        
        print("\n=== Training Complete ===")
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation AUC: {metrics['auc']:.4f}")
        
        return metrics
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale if scaler exists (for MLP)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X).flatten()
    
    def save_model(self, filepath: str):
        """Save trained model"""
        self.model.save(filepath)
        
        # Save scaler if exists
        if self.scaler is not None:
            scaler_path = str(filepath).replace('.h5', '_scaler.pkl').replace('.keras', '_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load saved model"""
        self.model = keras.models.load_model(filepath)
        
        # Try to load scaler
        scaler_path = str(filepath).replace('.h5', '_scaler.pkl').replace('.keras', '_scaler.pkl')
        try:
            self.scaler = joblib.load(scaler_path)
        except:
            pass
        
        print(f"✓ Model loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig


class EnsembleModel:
    """Ensemble of ML and DL models"""
    
    def __init__(self, models: list):
        self.models = models
    
    def predict(self, X):
        """Ensemble prediction using voting"""
        predictions = np.array([model.predict(X) for model in self.models])
        # Majority voting
        ensemble_pred = np.round(np.mean(predictions, axis=0))
        return ensemble_pred.astype(int)
    
    def predict_proba(self, X):
        """Ensemble prediction probabilities"""
        probas = np.array([model.predict_proba(X) if hasattr(model, 'predict_proba') 
                          else model.predict(X) for model in self.models])
        return np.mean(probas, axis=0)
