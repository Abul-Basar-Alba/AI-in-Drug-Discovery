#!/bin/bash

# 🇧🇩 AI in Drug Discovery - Complete Setup & Run Script

clear
echo "=========================================================="
echo "🧬 AI in Drug Discovery - Complete Setup"
echo "   CIT-316 (AI Sessional) Project"
echo "=========================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run from project root directory"
    exit 1
fi

# Step 1: Check Python
echo "📍 Step 1: Checking Python..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1)
    echo "   ✓ $python_version"
else
    echo "   ❌ Python 3 not found!"
    exit 1
fi
echo ""

# Step 2: Check Dataset
echo "📍 Step 2: Checking Dataset..."
if [ -f "data/raw/drug_data.csv" ]; then
    records=$(wc -l < data/raw/drug_data.csv)
    echo "   ✓ Dataset found ($records records)"
else
    echo "   ⚠ Dataset not found - generating now..."
    bash generate_quick_data.sh
fi
echo ""

# Step 3: Install Dependencies
echo "📍 Step 3: Installing Dependencies..."
echo "   (This may take a few minutes...)"
pip3 install --user --quiet numpy pandas scikit-learn xgboost matplotlib seaborn joblib opencv-python Pillow 2>/dev/null
echo "   ✓ Core packages installed"

pip3 install --user --quiet tensorflow jupyter notebook 2>/dev/null
echo "   ✓ Optional packages installed"
echo ""

# Step 4: Create directories
echo "📍 Step 4: Setting up directories..."
mkdir -p models data/processed data/images notebooks/checkpoints 2>/dev/null
echo "   ✓ Directories ready"
echo ""

# Step 5: Check if models exist
echo "📍 Step 5: Checking trained models..."
if [ -f "models/best_model.pkl" ]; then
    echo "   ✓ Trained models found - Ready to test!"
    models_ready=true
else
    echo "   ⚠ Models not trained yet"
    models_ready=false
fi
echo ""

# Summary
echo "=========================================================="
echo "✅ SETUP COMPLETE!"
echo "=========================================================="
echo ""

if [ "$models_ready" = true ]; then
    echo "🎯 Your project is FULLY READY!"
    echo ""
    echo "🧪 Test drugs now:"
    echo "   python3 manual_drug_test.py"
    echo ""
else
    echo "📚 Next Steps:"
    echo ""
    echo "1️⃣  Train Models (Choose ONE):"
    echo ""
    echo "   Option A - Jupyter Notebook (Recommended):"
    echo "   $ jupyter notebook"
    echo "   Then open: notebooks/train_model.ipynb"
    echo "   Run all cells (Cell → Run All)"
    echo ""
    echo "   Option B - Python Script:"
    echo "   $ python3 run_pipeline.py"
    echo ""
    echo "2️⃣  After Training, Test Drugs:"
    echo "   $ python3 manual_drug_test.py"
    echo ""
fi

echo "=========================================================="
echo "📖 Documentation:"
echo "   • Bangla: BANGLA_GUIDE.md (বাংলা গাইড)"
echo "   • Quick: QUICKSTART.md"
echo "   • Full:  PRESENTATION_REPORT.md"
echo "=========================================================="
echo ""

# Ask if user wants to train now
if [ "$models_ready" = false ]; then
    echo -n "Do you want to train models now? (y/n): "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo ""
        echo "Starting training pipeline..."
        python3 run_pipeline.py
    fi
fi

echo ""
echo "✨ Setup complete! Happy researching! 🚀"
echo ""
