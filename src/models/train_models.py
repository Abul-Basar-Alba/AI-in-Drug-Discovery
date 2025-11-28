"""
Model Training Module for Drug Discovery
Supports multiple ML and DL models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report)
import xgboost as xgb
from typing import Dict, Tuple, Any
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DrugDiscoveryModels:
    """Train and evaluate multiple models for drug discovery"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'target', 
                     test_size: float = 0.2) -> Tuple:
        """Prepare train and test datasets"""
        print("\n=== Preparing Data ===")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove any non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {dict(y.value_counts())}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Logistic Regression model"""
        print("\n--- Training Logistic Regression ---")
        
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(X_train, y_train)
        
        metrics = self.evaluate_model(model, X_test, y_test, 'Logistic Regression')
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = metrics
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, 
                           n_estimators: int = 200, max_depth: int = 15) -> Dict:
        """Train Random Forest model"""
        print("\n--- Training Random Forest ---")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        metrics = self.evaluate_model(model, X_test, y_test, 'Random Forest')
        self.models['random_forest'] = model
        self.results['random_forest'] = metrics
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Gradient Boosting model"""
        print("\n--- Training Gradient Boosting ---")
        
        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        
        metrics = self.evaluate_model(model, X_test, y_test, 'Gradient Boosting')
        self.models['gradient_boosting'] = model
        self.results['gradient_boosting'] = metrics
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def train_xgboost(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train XGBoost model"""
        print("\n--- Training XGBoost ---")
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        metrics = self.evaluate_model(model, X_test, y_test, 'XGBoost')
        self.models['xgboost'] = model
        self.results['xgboost'] = metrics
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def train_svm(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train SVM model (on subset for speed)"""
        print("\n--- Training SVM (on 20% sample for speed) ---")
        
        # Use subset for SVM (it's slow on large datasets)
        sample_size = min(2000, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train.iloc[indices]
        y_train_sample = y_train.iloc[indices]
        
        model = SVC(kernel='rbf', probability=True, random_state=self.random_state)
        model.fit(X_train_sample, y_train_sample)
        
        metrics = self.evaluate_model(model, X_test, y_test, 'SVM')
        self.models['svm'] = model
        self.results['svm'] = metrics
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train all models and compare performance"""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        # Train each model
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        self.train_xgboost(X_train, y_train, X_test, y_test)
        self.train_svm(X_train, y_train, X_test, y_test)
        
        # Find best model
        for model_name, metrics in self.results.items():
            if metrics['accuracy'] > self.best_score:
                self.best_score = metrics['accuracy']
                self.best_model = self.models[model_name]
                self.best_model_name = model_name
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        results_df = pd.DataFrame([
            {
                'Model': metrics['model_name'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}",
                'ROC AUC': f"{metrics['roc_auc']:.4f}"
            }
            for metrics in self.results.values()
        ])
        
        print(results_df.to_string(index=False))
        print("\n" + "="*60)
        print(f"🏆 BEST MODEL: {self.best_model_name} (Accuracy: {self.best_score:.4f})")
        print("="*60 + "\n")
        
        return self.results
    
    def hyperparameter_tuning(self, X_train, y_train, model_type: str = 'random_forest') -> Any:
        """Perform hyperparameter tuning for specified model"""
        print(f"\n=== Hyperparameter Tuning for {model_type} ===")
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        elif model_type == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            }
            model = xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss')
        
        else:
            print(f"Hyperparameter tuning not implemented for {model_type}")
            return None
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        print("Starting grid search (this may take a while)...")
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_models(self, output_dir: str = "models"):
        """Save trained models to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n=== Saving Models ===")
        for model_name, model in self.models.items():
            filepath = output_dir / f"{model_name}_model.pkl"
            joblib.dump(model, filepath)
            print(f"✓ Saved {model_name} to {filepath}")
        
        # Save best model separately
        if self.best_model is not None:
            filepath = output_dir / "best_model.pkl"
            joblib.dump(self.best_model, filepath)
            print(f"✓ Saved best model ({self.best_model_name}) to {filepath}")
        
        # Save results
        results_path = output_dir / "model_results.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved results to {results_path}")
    
    def load_model(self, model_path: str) -> Any:
        """Load a saved model"""
        model = joblib.load(model_path)
        print(f"✓ Loaded model from {model_path}")
        return model
    
    def get_feature_importance(self, model_name: str = None, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for tree-based models"""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models.get(model_name)
        if model is None:
            print(f"Model {model_name} not found")
            return None
        
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not have feature importance")
            return None
        
        # Get feature names (you'll need to pass these when training)
        importance_df = pd.DataFrame({
            'feature': range(len(model.feature_importances_)),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
