"""Loaders and preprocessing utilities."""
import os
from pathlib import Path
import pandas as pd
import numpy as np


def load_raw_csv(path):
    path = Path(path)
    return pd.read_csv(path)


def simple_preprocess(df, drop_cols=None, fill_method='median'):
    df = df.copy()
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])
    # numeric imputation
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].isna().any():
            if fill_method == 'median':
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(0)
    # categorical fill
    cat_cols = df.select_dtypes(include=['object']).columns
    for c in cat_cols:
        df[c] = df[c].fillna('missing')
    return df
