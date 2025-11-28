"""Simple Streamlit app to inspect sample data and model predictions."""
import streamlit as st
import pandas as pd
import joblib
import os
from src.featurizers import featurize_smiles_series


MODEL_PATH = 'models/rf.joblib'
CSV_PATH = 'data/raw/dataset.csv'


def load_model(path):
    if not os.path.exists(path):
        st.warning('Model not found. Run `python run_pipeline.py` first to train a model.')
        return None
    return joblib.load(path)


def main():
    st.title('AI Drug Discovery — Prediction Demo')
    st.write('Load synthetic dataset and show model predictions for selected samples.')

    if not os.path.exists(CSV_PATH):
        st.info('No synthetic data found. Run `python run_pipeline.py --n 1000` to generate data.')
        return

    df = pd.read_csv(CSV_PATH)
    st.sidebar.write('Data summary')
    st.sidebar.write(df.describe(include='all'))

    model = load_model(MODEL_PATH)

    sample_id = st.selectbox('Pick a sample id', df['id'].tolist()[:200])
    row = df[df['id'] == sample_id].iloc[0]
    st.write('Selected sample')
    st.write(row[['id','smiles','cat','feat1','feat2','feat3','feat4','feat5','target']])

    if model is not None:
        X_fp = featurize_smiles_series(pd.Series([row['smiles']]), n_bits=256)
        X_numeric = row[['feat1','feat2','feat3','feat4','feat5']].values.reshape(1, -1)
        import numpy as np
        X = np.hstack([X_fp, X_numeric])
        pred = model.predict_proba(X)[0, 1]
        st.metric('Predicted probability (positive class)', f'{pred:.4f}')

    st.write('Manual SMILES input')
    smi = st.text_input('Enter SMILES-like string', '')
    if smi and model is not None:
        X_fp = featurize_smiles_series(pd.Series([smi]), n_bits=256)
        X_numeric = pd.Series([0,0,0,0,0]).values.reshape(1,-1)
        import numpy as np
        X = np.hstack([X_fp, X_numeric])
        pred = model.predict_proba(X)[0, 1]
        st.write('Prediction for manual input:', pred)


if __name__ == '__main__':
    main()
