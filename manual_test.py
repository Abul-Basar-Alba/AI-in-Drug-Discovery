"""Manual testing helpers and example usage."""
import argparse
import joblib
import numpy as np
from src.data_loader import load_raw_csv, simple_preprocess
from src.featurizers import featurize_smiles_series


def run_manual_test(model_path, csv_path):
    df = load_raw_csv(csv_path)
    df = simple_preprocess(df)
    X_fp = featurize_smiles_series(df['smiles'], n_bits=256)
    X = np.hstack([X_fp, df[['feat1','feat2','feat3','feat4','feat5']].values])
    clf = joblib.load(model_path)
    preds = clf.predict_proba(X)[:, 1]
    df['pred'] = preds
    print(df[['id', 'target', 'pred']].head())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/rf.joblib')
    p.add_argument('--csv', default='data/raw/dataset.csv')
    args = p.parse_args()
    run_manual_test(args.model, args.csv)
