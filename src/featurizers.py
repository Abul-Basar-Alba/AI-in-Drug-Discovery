"""Feature engineering utilities: chemical fingerprints (RDKit if available), text embeddings and image transforms."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False


def morgan_fingerprint(smiles, n_bits=2048):
    if not RDKit_AVAILABLE:
        # fallback: simple hashing into a vector
        h = hash(smiles)
        vec = np.zeros(n_bits, dtype=int)
        vec[abs(h) % n_bits] = 1
        return vec
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize_smiles_series(series, n_bits=2048):
    # returns ndarray (n_samples, n_bits)
    feats = [morgan_fingerprint(s, n_bits=n_bits) for s in series.fillna('')]
    return np.vstack(feats)


def tfidf_text(series, max_features=512):
    vec = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(series.fillna(''))
    return X.toarray(), vec
