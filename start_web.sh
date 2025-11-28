#!/bin/bash

# AI Drug Discovery - Web Application Launcher
# Run this script to start the web interface

echo "========================================"
echo "🧬 AI Drug Discovery - Web Interface"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found! Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if in correct directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found!"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask not installed. Installing required packages..."
    echo ""
    pip3 install flask --user
    echo ""
fi

# Check if models exist
if [ ! -d "models" ] || [ ! -f "models/best_model.pkl" ]; then
    echo "⚠️  WARNING: Trained models not found!"
    echo ""
    echo "You need to train the models first. Choose an option:"
    echo ""
    echo "Option 1 (Recommended): Train via Jupyter Notebook"
    echo "  jupyter notebook notebooks/train_model.ipynb"
    echo "  (Run all cells in the notebook)"
    echo ""
    echo "Option 2: Train via Python script"
    echo "  python3 run_pipeline.py"
    echo ""
    echo "After training, run this script again."
    echo ""
    read -p "Do you want to continue without models? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if dataset exists
if [ ! -f "data/raw/drug_data.csv" ]; then
    echo "⚠️  Dataset not found. Generating sample data..."
    if [ -f "generate_quick_data.sh" ]; then
        bash generate_quick_data.sh
    else
        python3 -c "
import sys
sys.path.append('src')
from data.generate_data import generate_drug_data
generate_drug_data(n_samples=10000)
print('✓ Dataset generated')
"
    fi
    echo ""
fi

echo "========================================"
echo "🚀 Starting Web Server..."
echo "========================================"
echo ""
echo "📍 Server will start at: http://127.0.0.1:5000"
echo ""
echo "🌐 Open your browser and go to:"
echo "   http://127.0.0.1:5000"
echo ""
echo "📊 Features:"
echo "  ✓ Interactive drug testing interface"
echo "  ✓ Real-time predictions with visualizations"
echo "  ✓ 10 pre-loaded sample test cases"
echo "  ✓ Accuracy metrics and model info"
echo "  ✓ Beautiful responsive design"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo ""
echo "========================================"
echo ""

# Start the Flask app
python3 app.py
