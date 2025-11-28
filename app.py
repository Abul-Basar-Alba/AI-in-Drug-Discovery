"""
AI Drug Discovery - Web Application
Flask Backend with REST API
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

class DrugPredictionAPI:
    """Backend API for drug predictions"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.feature_names = None
        self.model_metrics = {}
        self.load_resources()
    
    def load_resources(self):
        """Load trained models and metrics"""
        try:
            # Load best model
            best_model_path = self.models_dir / "best_model.pkl"
            if best_model_path.exists():
                self.model = joblib.load(best_model_path)
                print("✓ Model loaded successfully")
            else:
                print("⚠ Model not found. Please train models first.")
                return False
            
            # Load feature names
            feature_path = self.models_dir / "feature_names.pkl"
            if feature_path.exists():
                self.feature_names = joblib.load(feature_path)
                print(f"✓ Feature names loaded ({len(self.feature_names)} features)")
            
            # Load metrics
            metrics_path = self.models_dir / "model_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.model_metrics = json.load(f)
                print("✓ Model metrics loaded")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading resources: {e}")
            return False
    
    def predict_drug(self, drug_data: dict):
        """Predict drug effectiveness"""
        try:
            # Create DataFrame
            df = pd.DataFrame([drug_data])
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0]
            
            # Get confidence
            confidence = float(max(probability) * 100)
            
            # Determine effectiveness
            if prediction == 1:
                if confidence >= 85:
                    effectiveness = "HIGHLY EFFECTIVE"
                    color = "#10b981"  # green
                elif confidence >= 70:
                    effectiveness = "EFFECTIVE"
                    color = "#3b82f6"  # blue
                else:
                    effectiveness = "MODERATELY EFFECTIVE"
                    color = "#f59e0b"  # yellow
            else:
                effectiveness = "NOT EFFECTIVE"
                color = "#ef4444"  # red
            
            # Risk assessment
            toxicity_score = drug_data.get('hepatotoxicity_score', 0) + \
                           drug_data.get('cardiotoxicity_score', 0) + \
                           drug_data.get('nephrotoxicity_score', 0)
            
            if toxicity_score < 5:
                risk = "LOW"
                risk_color = "#10b981"
            elif toxicity_score < 10:
                risk = "MODERATE"
                risk_color = "#f59e0b"
            else:
                risk = "HIGH"
                risk_color = "#ef4444"
            
            # Recommendation
            if prediction == 1 and toxicity_score < 7:
                recommendation = "✓ PROCEED TO CLINICAL TRIALS"
                rec_color = "#10b981"
            elif prediction == 1 and toxicity_score < 10:
                recommendation = "⚠ NEEDS FURTHER TESTING"
                rec_color = "#f59e0b"
            else:
                recommendation = "✗ NOT RECOMMENDED"
                rec_color = "#ef4444"
            
            return {
                'success': True,
                'prediction': int(prediction),
                'effectiveness': effectiveness,
                'confidence': round(confidence, 2),
                'probability_not_effective': round(float(probability[0]) * 100, 2),
                'probability_effective': round(float(probability[1]) * 100, 2),
                'risk_level': risk,
                'toxicity_score': round(toxicity_score, 2),
                'recommendation': recommendation,
                'colors': {
                    'effectiveness': color,
                    'risk': risk_color,
                    'recommendation': rec_color
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_info(self):
        """Get model information and metrics"""
        return {
            'success': True,
            'metrics': self.model_metrics,
            'features_count': len(self.feature_names) if self.feature_names else 0,
            'model_type': str(type(self.model).__name__) if self.model else 'Not loaded'
        }

# Initialize API
api = DrugPredictionAPI()

# Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.json
        result = api.predict_drug(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify(api.get_model_info())

@app.route('/api/sample-drugs', methods=['GET'])
def sample_drugs():
    """Get sample drug test cases"""
    try:
        with open('data/sample_drugs.json', 'r') as f:
            samples = json.load(f)
        return jsonify({
            'success': True,
            'samples': samples
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': api.model is not None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 AI DRUG DISCOVERY - WEB APPLICATION")
    print("="*60 + "\n")
    
    if api.model is None:
        print("⚠ WARNING: Models not loaded!")
        print("Please train models first by running:")
        print("  python run_pipeline.py")
        print("  OR")
        print("  jupyter notebook notebooks/train_model.ipynb")
        print("\n")
    
    print("🌐 Starting server at http://127.0.0.1:5000")
    print("📊 Open browser to see the dashboard")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
