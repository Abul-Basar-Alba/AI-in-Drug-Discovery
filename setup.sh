#!/bin/bash

# Setup script for Drug Discovery AI Project

echo "=================================================="
echo "Drug Discovery AI - Setup Script"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
echo "Python version: $python_version"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -q numpy pandas scikit-learn matplotlib seaborn opencv-python Pillow joblib jupyter notebook tqdm xgboost 2>/dev/null

# Try installing TensorFlow
echo ""
echo "Installing TensorFlow (this may take a while)..."
pip install -q tensorflow 2>/dev/null || echo "Warning: TensorFlow installation failed. Deep learning features may not work."

echo ""
echo "=================================================="
echo "✓ Setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Generate datasets: python src/data/generate_data.py"
echo "  2. Train models: jupyter notebook notebooks/train_model.ipynb"
echo "  3. Test drugs: python manual_drug_test.py"
echo ""
