# AI in Drug Discovery — Project Scaffold

This project demonstrates an end-to-end scaffold for using AI in drug discovery. It includes data acquisition, preprocessing, feature engineering, multiple model training examples, and a notebook for experimentation.

Quick start

1. Create a conda environment and install RDKit (recommended):

```bash
conda create -n ai-drug python=3.10 -y
conda activate ai-drug
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

2. Generate a synthetic dataset (10k+ samples) and train a model using the notebook in `notebooks/train_model.ipynb`.

Structure

- `data/` — raw and processed datasets
- `notebooks/` — exploratory and training notebooks
- `src/` — data, featurization, and model code
- `models/` — saved model artifacts
- `reports/` — short report and results

Notes
- RDKit is optional for basic functionality; install via conda for full cheminformatics support.
- This scaffold includes a synthetic dataset generator so you can run everything locally without downloading large external datasets.

Installing RDKit (optional)

RDKit binary wheels are not reliably available via pip on many platforms (especially on newer Python versions). If you need full RDKit support:

- Use conda (recommended):

```bash
conda create -n ai-drug python=3.10 -y
conda activate ai-drug
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

- If you cannot use conda, remove the `rdkit-pypi` entry and rely on the fallback featurizers included in `src/featurizers.py`.

If pip install fails on your system

- Try the minimal requirements which are sufficient to run the sample pipeline (no RDKit, no PyTorch/XGBoost):

```bash
pip install -r requirements-min.txt
```

- Or create a venv using a compatible Python (3.10 or 3.11) if available:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



Feel free to ask me to implement specific dataset downloads (ChEMBL, Tox21), add GNNs, or expand the notebook.
