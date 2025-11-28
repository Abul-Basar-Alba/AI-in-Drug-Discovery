"""
Utility functions for the Drug Discovery project
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_molecular_descriptors(smiles: str) -> Dict[str, float]:
    """
    Calculate molecular descriptors from SMILES string
    Note: This is a placeholder. Use RDKit for real implementation.
    """
    # Simple placeholder implementation
    return {
        'molecular_weight': len(smiles) * 12.5,  # Rough estimate
        'num_atoms': len(smiles),
        'complexity': len(set(smiles)) / len(smiles)
    }


def plot_confusion_matrix(cm, class_names=['Not Effective', 'Effective'], 
                          title='Confusion Matrix', figsize=(8, 6)):
    """Plot a confusion matrix"""
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()


def plot_feature_importance(importance_df, top_n=20, figsize=(12, 8)):
    """Plot feature importance"""
    plt.figure(figsize=figsize)
    importance_df = importance_df.head(top_n)
    plt.barh(range(len(importance_df)), importance_df['importance'], color='coral')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return plt.gcf()


def plot_roc_curves(models_dict, X_test, y_test, figsize=(10, 8)):
    """Plot ROC curves for multiple models"""
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=figsize)
    
    for model_name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            continue
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def create_drug_profile(drug_data: Dict[str, Any]) -> str:
    """Create a formatted drug profile summary"""
    profile = []
    profile.append("="*60)
    profile.append("DRUG PROFILE")
    profile.append("="*60)
    
    for key, value in drug_data.items():
        profile.append(f"{key}: {value}")
    
    profile.append("="*60)
    return "\n".join(profile)


def save_model_comparison(results: Dict, output_path: str = "reports/model_comparison.csv"):
    """Save model comparison results to CSV"""
    comparison_data = []
    
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': metrics.get('model_name', model_name),
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1 Score': metrics.get('f1_score', 0),
            'ROC AUC': metrics.get('roc_auc', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    df.to_csv(output_path, index=False)
    print(f"✓ Model comparison saved to {output_path}")
    return df


def validate_drug_properties(drug_data: Dict[str, float]) -> Dict[str, List[str]]:
    """
    Validate drug properties are within acceptable ranges
    Returns dict with 'errors' and 'warnings' lists
    """
    errors = []
    warnings = []
    
    # Define acceptable ranges
    ranges = {
        'molecular_weight': (50, 800, 'error'),
        'logP': (-2, 8, 'warning'),
        'bioavailability': (0, 100, 'error'),
        'hepatotoxicity_score': (0, 10, 'error'),
        'cardiotoxicity_score': (0, 10, 'error'),
    }
    
    for prop, (min_val, max_val, severity) in ranges.items():
        if prop in drug_data:
            value = drug_data[prop]
            if value < min_val or value > max_val:
                msg = f"{prop} ({value}) outside acceptable range [{min_val}, {max_val}]"
                if severity == 'error':
                    errors.append(msg)
                else:
                    warnings.append(msg)
    
    return {'errors': errors, 'warnings': warnings}


def calculate_drug_score(drug_data: Dict[str, float]) -> float:
    """
    Calculate overall drug score based on multiple factors
    Returns score from 0-100
    """
    weights = {
        'efficacy_score': 0.3,
        'safety_score': 0.3,
        'bioavailability': 0.2,
    }
    
    score = 0
    total_weight = 0
    
    for prop, weight in weights.items():
        if prop in drug_data:
            value = drug_data[prop]
            # Normalize to 0-10 scale
            if prop == 'bioavailability':
                value = value / 10  # Convert from 0-100 to 0-10
            
            score += value * weight
            total_weight += weight
    
    # Penalize for toxicity
    if 'hepatotoxicity_score' in drug_data:
        score -= drug_data['hepatotoxicity_score'] * 0.1
    if 'cardiotoxicity_score' in drug_data:
        score -= drug_data['cardiotoxicity_score'] * 0.1
    
    # Normalize to 0-100
    final_score = (score / total_weight) * 10
    return max(0, min(100, final_score))


def generate_recommendations(drug_data: Dict[str, float], 
                            prediction: int, 
                            confidence: float) -> List[str]:
    """Generate recommendations based on drug properties and predictions"""
    recommendations = []
    
    if prediction == 1 and confidence > 0.8:
        recommendations.append("✓ Highly promising candidate for further development")
    elif prediction == 1 and confidence > 0.6:
        recommendations.append("⚠ Promising but requires additional validation")
    else:
        recommendations.append("✗ Not recommended for development in current form")
    
    # Check toxicity
    if drug_data.get('hepatotoxicity_score', 0) > 6:
        recommendations.append("⚠ High hepatotoxicity risk - consider structural modifications")
    if drug_data.get('cardiotoxicity_score', 0) > 6:
        recommendations.append("⚠ High cardiotoxicity risk - additional safety studies needed")
    
    # Check drug-like properties
    if drug_data.get('molecular_weight', 0) > 500:
        recommendations.append("⚠ High molecular weight may affect bioavailability")
    if drug_data.get('logP', 0) > 5:
        recommendations.append("⚠ High lipophilicity may cause absorption issues")
    
    # Check efficacy
    if drug_data.get('efficacy_score', 0) < 5:
        recommendations.append("⚠ Low efficacy score - consider optimizing lead structure")
    
    return recommendations
