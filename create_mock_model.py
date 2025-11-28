#!/usr/bin/env python3
"""
Quick Mock Model Generator
Creates a simple trained model for immediate testing using scikit-learn
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

# Create models directory
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Feature names (15 features)
feature_names = [
    'molecular_weight', 'logP', 'num_h_donors', 'num_h_acceptors',
    'tpsa', 'num_rotatable_bonds', 'num_aromatic_rings',
    'bioavailability_score', 'efficacy_score', 'safety_score',
    'hepatotoxicity_score', 'cardiotoxicity_score', 'nephrotoxicity_score',
    'binding_affinity', 'absorption_rate'
]

print("Creating mock training data...")

# Create simple synthetic training data
# Effective drugs: high efficacy/safety, low toxicity
X_effective = np.array([
    [320, 3.5, 2, 4, 65, 5, 2, 0.90, 9.0, 8.5, 1.5, 1.2, 1.0, -9.5, 0.92],
    [310, 3.2, 3, 5, 70, 4, 2, 0.88, 8.8, 8.2, 1.8, 1.5, 1.3, -9.2, 0.89],
    [335, 3.8, 2, 3, 62, 6, 3, 0.92, 9.2, 8.8, 1.2, 1.0, 0.8, -9.8, 0.94],
    [305, 3.0, 2, 4, 68, 5, 2, 0.85, 8.5, 8.0, 2.0, 1.7, 1.5, -9.0, 0.87],
    [340, 3.9, 1, 3, 60, 7, 3, 0.93, 9.3, 9.0, 1.0, 0.9, 0.7, -10.0, 0.95]
])
y_effective = np.ones(5)

# Non-effective drugs: low efficacy/safety, high toxicity
X_ineffective = np.array([
    [280, 2.5, 4, 6, 85, 3, 1, 0.65, 5.5, 5.2, 6.5, 6.8, 5.9, -7.2, 0.68],
    [290, 2.8, 3, 5, 80, 4, 1, 0.70, 6.0, 5.8, 6.0, 6.2, 5.5, -7.5, 0.72],
    [275, 2.3, 5, 7, 90, 2, 1, 0.60, 5.0, 5.0, 7.0, 7.2, 6.5, -7.0, 0.65],
    [295, 2.9, 3, 6, 82, 3, 1, 0.68, 5.8, 5.5, 6.3, 6.5, 5.8, -7.3, 0.70],
    [285, 2.6, 4, 6, 87, 3, 1, 0.63, 5.3, 5.3, 6.7, 6.9, 6.2, -7.1, 0.67]
])
y_ineffective = np.zeros(5)

# Combine data
X_train = np.vstack([X_effective, X_ineffective])
y_train = np.concatenate([y_effective, y_ineffective])

print("Training simple decision tree model...")
# Use a simple decision tree (easy to pickle and load)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Verify it works
test_sample = X_effective[0:1]
pred = model.predict(test_sample)
proba = model.predict_proba(test_sample)
print(f"Test prediction: {pred[0]} (probability: {proba[0]})")

# Save model
model_path = models_dir / "best_model.pkl"
joblib.dump(model, model_path)
print(f"✓ Saved model to {model_path}")

# Save feature names
feature_names_path = models_dir / "feature_names.pkl"
joblib.dump(feature_names, feature_names_path)
print(f"✓ Saved feature names to {feature_names_path}")

# Create mock metrics
metrics = {
    "DecisionTree": {
        "accuracy": 0.885,
        "precision": 0.892,
        "recall": 0.878,
        "f1_score": 0.885
    },
    "Mock Model": {
        "accuracy": 0.862,
        "precision": 0.870,
        "recall": 0.854,
        "f1_score": 0.862
    }
}

metrics_path = models_dir / "model_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Saved metrics to {metrics_path}")

print("\n✅ Mock model created successfully!")
print("You can now use the web interface for predictions.")
print("For production, train real models with: python run_pipeline.py")
