"""Run the synthetic data generation and train a RandomForest model quickly.

This script is intended to be run inside the project's virtual environment.
"""
import argparse
import os
import sys

from src.data_acquisition import generate_synthetic_dataset
from src.data_loader import load_raw_csv, simple_preprocess
from src.featurizers import featurize_smiles_series
from src.models import train_random_forest
import numpy as np


def main(out_dir='data', n=1000, n_bits=256):
    print(f'Generating synthetic dataset: n={n} -> {out_dir}/raw')
    res = generate_synthetic_dataset(out_dir, n_samples=n)
    csv_path = res['csv']
    print('Loading CSV:', csv_path)
    df = load_raw_csv(csv_path)
    print('Preprocessing...')
    df = simple_preprocess(df)
    print('Featurizing SMILES...')
    X_fp = featurize_smiles_series(df['smiles'], n_bits=n_bits)
    X_numeric = df[['feat1','feat2','feat3','feat4','feat5']].fillna(0).values
    X = np.hstack([X_fp, X_numeric])
    y = df['target'].values
    print('Training RandomForest...')
    metrics = train_random_forest(X, y, out_path=os.path.join('models', 'rf.joblib'))
    print('Training complete. Metrics:')
    for k, v in metrics.items():
        if k != 'model':
            print(f'  {k}: {v}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data')
    p.add_argument('--n', type=int, default=1000, help='Number of synthetic samples to generate')
    p.add_argument('--bits', type=int, default=256, help='Fingerprint size')
    args = p.parse_args()
    main(out_dir=args.out, n=args.n, n_bits=args.bits)
