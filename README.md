# AI-Driven Drug Discovery System

## 🧬 Overview

This project implements a comprehensive AI system for drug discovery using machine learning, deep learning, and natural language processing techniques. The system analyzes drug properties, molecular structures, and interactions to predict drug effectiveness with high accuracy.

## ✨ Key Features

- **Multi-Source Data Handling**: Processes CSV, JSON, and image data
- **10,000+ Sample Dataset**: Large-scale synthetic drug discovery dataset
- **Advanced Data Processing**:
  - Missing value imputation (median, KNN, mode)
  - Feature engineering (interactions, polynomials, statistical features)
  - Outlier detection and removal
  - Categorical encoding and feature scaling

- **Multiple ML Models**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Support Vector Machine (SVM)

- **Deep Learning Models**:
  - Multi-Layer Perceptron (MLP) for tabular data
  - Convolutional Neural Network (CNN) for molecular structures

- **Comprehensive Evaluation**:
  - Accuracy, Precision, Recall, F1 Score, ROC-AUC
  - Confusion matrices
  - ROC curves
  - Feature importance analysis

- **Interactive Testing Interface**: Manual drug testing with trained models

## 📁 Project Structure

```
AI-Assignment/
├── data/
│   ├── raw/                    # Raw datasets (CSV, JSON)
│   ├── processed/              # Processed datasets
│   └── images/                 # Molecular structure images
├── models/                     # Saved trained models
├── notebooks/
│   └── train_model.ipynb       # Complete training pipeline
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Multi-source data loading
│   │   ├── preprocessing.py    # Data preprocessing & feature engineering
│   │   └── generate_data.py    # Dataset generation
│   ├── models/
│   │   ├── train_models.py     # ML model training
│   │   └── deep_learning.py    # Deep learning models
│   └── utils/
├── manual_drug_test.py         # Interactive testing script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA for GPU-accelerated deep learning

### Installation

1. **Navigate to project directory**:
```bash
cd "AI-Assignment"
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Generate Dataset

Generate the 10,000+ sample drug discovery dataset:

```bash
python -c "from src.data.generate_data import generate_all_datasets; generate_all_datasets(n_samples=10000, n_images=500)"
```

This creates:
- `data/raw/drug_data.csv`: 10,000 drug records with molecular properties
- `data/raw/drug_interactions.json`: Drug interaction data
- `data/images/`: 500 molecular structure images

## 📊 Training Models

### Using Jupyter Notebook (Recommended)

1. **Start Jupyter Notebook**:
```bash
jupyter notebook
```

2. **Open and run** `notebooks/train_model.ipynb`

This notebook provides:
- Step-by-step data loading and exploration
- Visualization of data distributions
- Feature engineering pipeline
- Training of all models (ML + DL)
- Model comparison and evaluation
- Model saving

## 🧪 Testing Drugs

### Interactive Mode

Test individual drug candidates:

```bash
python manual_drug_test.py
```

### Batch Mode

Test multiple drugs from CSV:

```bash
python manual_drug_test.py --batch data/test_drugs.csv
```

## 📈 Model Performance

Expected accuracy on test data: **85-90%**

## ⚠️ Disclaimer

This is an educational project using synthetic data. Not for actual drug discovery or medical decisions.

---

**Made with ❤️ for AI in Drug Discovery**

- Or create a venv using a compatible Python (3.10 or 3.11) if available:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



Feel free to ask me to implement specific dataset downloads (ChEMBL, Tox21), add GNNs, or expand the notebook.
