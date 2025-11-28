"""
Manual Drug Testing Interface
Test new drug candidates with trained models
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


class DrugTester:
    """Interactive drug testing interface"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_names = None
        self.preprocessor = None
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        print("\n" + "="*60)
        print("LOADING TRAINED MODELS")
        print("="*60 + "\n")
        
        try:
            # Load feature names
            feature_path = self.models_dir / "feature_names.pkl"
            if feature_path.exists():
                self.feature_names = joblib.load(feature_path)
                print(f"✓ Loaded feature names ({len(self.feature_names)} features)")
            
            # Load best model
            best_model_path = self.models_dir / "best_model.pkl"
            if best_model_path.exists():
                self.models['best'] = joblib.load(best_model_path)
                print(f"✓ Loaded best model")
            
            # Load deep learning model
            try:
                from tensorflow import keras
                dl_model_path = self.models_dir / "deep_neural_network.keras"
                if dl_model_path.exists():
                    self.models['deep_learning'] = keras.models.load_model(dl_model_path)
                    
                    # Load scaler
                    scaler_path = self.models_dir / "deep_neural_network_scaler.pkl"
                    if scaler_path.exists():
                        self.models['dl_scaler'] = joblib.load(scaler_path)
                    
                    print(f"✓ Loaded deep learning model")
            except Exception as e:
                print(f"⚠ Could not load deep learning model: {e}")
            
            print("\n" + "="*60)
            print(f"✓ Successfully loaded {len(self.models)} model(s)")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please train models first by running the training notebook.")
            sys.exit(1)
    
    def create_sample_drug(self) -> dict:
        """Create a sample drug with default values"""
        return {
            'molecular_weight': 350.0,
            'logP': 2.5,
            'h_bond_donors': 2,
            'h_bond_acceptors': 4,
            'rotatable_bonds': 5,
            'bioavailability': 70.0,
            'half_life': 8.0,
            'clearance': 4.0,
            'hepatotoxicity_score': 3.0,
            'cardiotoxicity_score': 2.5,
            'solubility': 3.0,
            'melting_point': 180.0,
            'pKa': 7.0,
            'efficacy_score': 7.5,
            'safety_score': 8.0,
        }
    
    def get_user_input(self) -> dict:
        """Get drug properties from user"""
        print("\n" + "="*60)
        print("ENTER DRUG PROPERTIES")
        print("="*60)
        print("(Press Enter to use default values shown in brackets)\n")
        
        default_drug = self.create_sample_drug()
        drug_data = {}
        
        # Define property descriptions
        properties = {
            'molecular_weight': ('Molecular Weight (Da)', 50, 800),
            'logP': ('Lipophilicity (logP)', -2, 8),
            'h_bond_donors': ('Hydrogen Bond Donors', 0, 10),
            'h_bond_acceptors': ('Hydrogen Bond Acceptors', 0, 15),
            'rotatable_bonds': ('Rotatable Bonds', 0, 20),
            'bioavailability': ('Bioavailability (%)', 0, 100),
            'half_life': ('Half-life (hours)', 0.5, 50),
            'clearance': ('Clearance (L/h)', 0.1, 30),
            'hepatotoxicity_score': ('Hepatotoxicity Score (0-10)', 0, 10),
            'cardiotoxicity_score': ('Cardiotoxicity Score (0-10)', 0, 10),
            'solubility': ('Solubility (log scale)', -5, 8),
            'melting_point': ('Melting Point (°C)', 50, 350),
            'pKa': ('pKa', 2, 14),
            'efficacy_score': ('Efficacy Score (1-10)', 1, 10),
            'safety_score': ('Safety Score (1-10)', 1, 10),
        }
        
        for prop, (description, min_val, max_val) in properties.items():
            default_val = default_drug[prop]
            while True:
                try:
                    user_input = input(f"{description} [{default_val}]: ").strip()
                    if user_input == '':
                        drug_data[prop] = default_val
                        break
                    else:
                        value = float(user_input)
                        if min_val <= value <= max_val:
                            drug_data[prop] = value
                            break
                        else:
                            print(f"  ⚠ Value must be between {min_val} and {max_val}")
                except ValueError:
                    print("  ⚠ Please enter a valid number")
        
        return drug_data
    
    def preprocess_input(self, drug_data: dict) -> pd.DataFrame:
        """Preprocess user input to match training data format"""
        # Create DataFrame
        df = pd.DataFrame([drug_data])
        
        # Add derived features (simple version - just the core features)
        # In production, you'd want to apply the same preprocessing as training
        
        # Ensure all expected features are present
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in df.columns:
                    # Add missing features with default value
                    df[feature] = 0
            
            # Reorder columns to match training
            df = df[self.feature_names]
        
        return df
    
    def predict(self, drug_data: dict) -> dict:
        """Make predictions using all available models"""
        print("\n" + "="*60)
        print("MAKING PREDICTIONS")
        print("="*60 + "\n")
        
        # Preprocess input
        X = self.preprocess_input(drug_data)
        
        predictions = {}
        
        # Best ML model prediction
        if 'best' in self.models:
            pred = self.models['best'].predict(X)[0]
            pred_proba = self.models['best'].predict_proba(X)[0]
            
            predictions['best_ml'] = {
                'prediction': int(pred),
                'confidence': float(pred_proba[1]),
                'effectiveness_probability': float(pred_proba[1]) * 100
            }
            
            print(f"Best ML Model Prediction:")
            print(f"  Result: {'✓ EFFECTIVE' if pred == 1 else '✗ NOT EFFECTIVE'}")
            print(f"  Confidence: {pred_proba[1]*100:.2f}%")
            print()
        
        # Deep learning model prediction
        if 'deep_learning' in self.models:
            try:
                X_scaled = X
                if 'dl_scaler' in self.models:
                    X_scaled = self.models['dl_scaler'].transform(X)
                
                pred_proba = self.models['deep_learning'].predict(X_scaled, verbose=0)[0][0]
                pred = int(pred_proba > 0.5)
                
                predictions['deep_learning'] = {
                    'prediction': pred,
                    'confidence': float(pred_proba),
                    'effectiveness_probability': float(pred_proba) * 100
                }
                
                print(f"Deep Learning Model Prediction:")
                print(f"  Result: {'✓ EFFECTIVE' if pred == 1 else '✗ NOT EFFECTIVE'}")
                print(f"  Confidence: {pred_proba*100:.2f}%")
                print()
            except Exception as e:
                print(f"⚠ Deep learning prediction failed: {e}\n")
        
        # Ensemble prediction (average of all models)
        if len(predictions) > 1:
            avg_prob = np.mean([p['effectiveness_probability'] for p in predictions.values()])
            ensemble_pred = 1 if avg_prob > 50 else 0
            
            predictions['ensemble'] = {
                'prediction': ensemble_pred,
                'effectiveness_probability': avg_prob
            }
            
            print(f"Ensemble Prediction (Average):")
            print(f"  Result: {'✓ EFFECTIVE' if ensemble_pred == 1 else '✗ NOT EFFECTIVE'}")
            print(f"  Confidence: {avg_prob:.2f}%")
            print()
        
        return predictions
    
    def display_results(self, drug_data: dict, predictions: dict):
        """Display formatted results"""
        print("\n" + "="*60)
        print("DRUG ANALYSIS SUMMARY")
        print("="*60 + "\n")
        
        # Display drug properties
        print("Drug Properties:")
        for prop, value in drug_data.items():
            print(f"  {prop}: {value}")
        
        print("\n" + "-"*60)
        
        # Display predictions
        print("\nPrediction Results:")
        
        if 'ensemble' in predictions:
            result = predictions['ensemble']
            effectiveness = result['effectiveness_probability']
            
            print(f"\n  🎯 FINAL VERDICT: ", end="")
            if effectiveness >= 70:
                print(f"✓ HIGHLY EFFECTIVE ({effectiveness:.1f}%)")
            elif effectiveness >= 50:
                print(f"✓ POTENTIALLY EFFECTIVE ({effectiveness:.1f}%)")
            else:
                print(f"✗ NOT LIKELY EFFECTIVE ({effectiveness:.1f}%)")
            
            # Risk assessment
            print(f"\n  Risk Assessment:")
            hep_risk = drug_data.get('hepatotoxicity_score', 0)
            card_risk = drug_data.get('cardiotoxicity_score', 0)
            
            if hep_risk < 3:
                print(f"    Hepatotoxicity: ✓ LOW RISK ({hep_risk}/10)")
            elif hep_risk < 6:
                print(f"    Hepatotoxicity: ⚠ MODERATE RISK ({hep_risk}/10)")
            else:
                print(f"    Hepatotoxicity: ✗ HIGH RISK ({hep_risk}/10)")
            
            if card_risk < 3:
                print(f"    Cardiotoxicity: ✓ LOW RISK ({card_risk}/10)")
            elif card_risk < 6:
                print(f"    Cardiotoxicity: ⚠ MODERATE RISK ({card_risk}/10)")
            else:
                print(f"    Cardiotoxicity: ✗ HIGH RISK ({card_risk}/10)")
            
            # Recommendations
            print(f"\n  Recommendations:")
            if effectiveness >= 70 and hep_risk < 5 and card_risk < 5:
                print("    ✓ Proceed to preclinical trials")
                print("    ✓ Monitor for long-term safety")
            elif effectiveness >= 50:
                print("    ⚠ Consider chemical modifications to improve effectiveness")
                print("    ⚠ Reduce toxicity scores if possible")
            else:
                print("    ✗ Not recommended for development")
                print("    ✗ Consider alternative compounds")
        
        print("\n" + "="*60 + "\n")
    
    def interactive_mode(self):
        """Run interactive testing mode"""
        print("\n" + "="*70)
        print("DRUG DISCOVERY AI - MANUAL TESTING INTERFACE")
        print("="*70)
        print("\nThis tool allows you to test drug candidates using trained AI models.")
        print("Enter drug properties to get effectiveness predictions.\n")
        
        while True:
            # Get user input
            drug_data = self.get_user_input()
            
            # Make predictions
            predictions = self.predict(drug_data)
            
            # Display results
            self.display_results(drug_data, predictions)
            
            # Ask to continue
            continue_testing = input("Test another drug? (y/n): ").strip().lower()
            if continue_testing != 'y':
                break
        
        print("\n✓ Thank you for using Drug Discovery AI!")
        print("="*70 + "\n")
    
    def batch_test_from_csv(self, csv_path: str):
        """Test multiple drugs from CSV file"""
        print(f"\nLoading drugs from {csv_path}...")
        
        df = pd.read_csv(csv_path)
        results = []
        
        for idx, row in df.iterrows():
            drug_data = row.to_dict()
            predictions = self.predict(drug_data)
            
            result = drug_data.copy()
            if 'ensemble' in predictions:
                result['predicted_effectiveness'] = predictions['ensemble']['prediction']
                result['effectiveness_probability'] = predictions['ensemble']['effectiveness_probability']
            
            results.append(result)
            
            print(f"Processed drug {idx+1}/{len(df)}")
        
        # Save results
        results_df = pd.DataFrame(results)
        output_path = csv_path.replace('.csv', '_predictions.csv')
        results_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Results saved to {output_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test drug candidates with AI models')
    parser.add_argument('--batch', type=str, help='CSV file with drugs to test in batch mode')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory containing trained models')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = DrugTester(models_dir=args.models_dir)
    
    if args.batch:
        # Batch mode
        tester.batch_test_from_csv(args.batch)
    else:
        # Interactive mode
        tester.interactive_mode()


if __name__ == "__main__":
    main()
