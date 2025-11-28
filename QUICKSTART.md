# Drug Discovery AI Project - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

You have two options:

**Option A - Using pip (if allowed on your system):**
```bash
pip install -r requirements.txt
```

**Option B - Using conda (recommended):**
```bash
conda create -n drug-ai python=3.9 -y
conda activate drug-ai
pip install -r requirements.txt
```

**Option C - If pip is restricted:**
The project works with basic Python libraries. Core functionality will work with:
```bash
python3 -m pip install --user numpy pandas scikit-learn matplotlib joblib
```

### Step 2: Generate Dataset (Already Done!)

The dataset has been generated with 10,000+ samples:
```bash
# Already completed - files are in data/raw/
✓ data/raw/drug_data.csv (10,000 records)
✓ data/raw/drug_interactions.json (10,000 records)
```

If you need to regenerate:
```bash
bash generate_quick_data.sh
```

### Step 3: Train Models

**Option A - Using Jupyter Notebook (Recommended):**
```bash
jupyter notebook
# Open notebooks/train_model.ipynb and run all cells
```

**Option B - Using Python Script:**
```bash
python run_pipeline.py
```

This will:
- Load and preprocess data
- Engineer features
- Train 5+ ML models
- Evaluate and compare models
- Save the best models to `models/` directory

**Expected Training Time:** 5-15 minutes depending on your hardware

### Step 4: Test Your Own Drugs

**Interactive Testing:**
```bash
python manual_drug_test.py
```

Enter drug properties when prompted. The system will predict effectiveness and provide recommendations.

**Example Input:**
```
Molecular Weight [350.0]: 420
Lipophilicity (logP) [2.5]: 3.2
Hepatotoxicity Score (0-10) [3.0]: 2.5
... (and so on)
```

**Example Output:**
```
🎯 FINAL VERDICT: ✓ HIGHLY EFFECTIVE (78.5%)

Risk Assessment:
  Hepatotoxicity: ✓ LOW RISK (2.5/10)
  Cardiotoxicity: ✓ LOW RISK (2.0/10)

Recommendations:
  ✓ Proceed to preclinical trials
  ✓ Monitor for long-term safety
```

## 📊 Project Structure Quick Reference

```
AI-Assignment/
├── data/raw/               # Your 10,000+ sample dataset
├── models/                 # Trained models (created after training)
├── notebooks/              # Jupyter training notebook
├── src/                    # All source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # ML and DL model implementations
│   └── utils/             # Helper functions
├── manual_drug_test.py     # Interactive testing script
├── requirements.txt        # Python dependencies
└── README.md              # Full documentation
```

## 🎯 Key Features

### Data Processing
- ✅ Handles missing values (7% introduced and handled)
- ✅ Feature engineering (interactions, polynomials, statistical)
- ✅ Outlier detection and removal
- ✅ Multiple data formats (CSV, JSON, images)

### Models Included
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ Gradient Boosting
- ✅ XGBoost
- ✅ SVM
- ✅ Deep Neural Network (MLP)
- ✅ CNN (for images, optional)

### Evaluation Metrics
- ✅ Accuracy, Precision, Recall, F1 Score
- ✅ ROC-AUC curves
- ✅ Confusion matrices
- ✅ Feature importance analysis

## 🔧 Common Issues & Solutions

### Issue: "No module named 'pandas'"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "Dataset not found"
**Solution:** Generate the dataset:
```bash
bash generate_quick_data.sh
```

### Issue: "Model files not found"
**Solution:** Train models first:
```bash
jupyter notebook notebooks/train_model.ipynb
# OR
python run_pipeline.py
```

### Issue: TensorFlow installation fails
**Solution:** Deep learning is optional. The project works with ML models only:
```bash
# Install without TensorFlow
pip install numpy pandas scikit-learn xgboost matplotlib seaborn opencv-python Pillow joblib
```

## 📈 Expected Performance

After training, you should see:
- **Best Model Accuracy:** 85-90%
- **Training Time:** 5-15 minutes
- **Model Size:** ~5-50 MB
- **Prediction Time:** <1 second per drug

## 🎓 Learning Path

1. **Start Here:** Run `notebooks/train_model.ipynb` to understand the full pipeline
2. **Explore Code:** Check `src/` directory for implementation details
3. **Test Models:** Use `manual_drug_test.py` for hands-on testing
4. **Customize:** Modify `src/data/preprocessing.py` for custom features
5. **Extend:** Add new models in `src/models/`

## 💡 Tips for Best Results

1. **Feature Engineering is Key:** The preprocessing pipeline creates 50+ features from the original data
2. **Use Ensemble Methods:** XGBoost and Random Forest typically perform best
3. **Validate Predictions:** Check toxicity scores along with effectiveness
4. **Understand Trade-offs:** High efficacy with high toxicity is not desirable

## 📚 Next Steps

After completing the quick start:

1. **Experiment with Parameters:**
   - Modify model hyperparameters
   - Try different feature engineering techniques
   - Adjust train/test split ratios

2. **Add Real Data:**
   - Replace synthetic data with actual drug databases
   - Add molecular descriptors from RDKit
   - Include clinical trial outcomes

3. **Enhance Models:**
   - Implement ensemble voting
   - Add cross-validation
   - Try deep learning architectures

4. **Deploy:**
   - Create a web interface with Flask/FastAPI
   - Build a REST API for predictions
   - Package as a Docker container

## 🤝 Need Help?

- Review the full `README.md` for comprehensive documentation
- Check `reports/summary.md` for project analysis
- Explore Jupyter notebook cells for step-by-step explanations

## ⚠️ Important Notes

- This is an **educational project** with synthetic data
- **Do NOT use** for actual drug discovery or medical decisions
- Real drug development requires extensive testing and FDA approval
- Models are for demonstration purposes only

## 📧 Feedback

Found a bug? Have suggestions? Want to contribute?
This project is designed for learning and can be extended in many ways!

---

**Happy Drug Discovering! 🧬💊🔬**
