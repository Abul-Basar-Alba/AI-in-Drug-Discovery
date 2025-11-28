"""Model training helpers: scikit-learn and simple PyTorch MLP."""
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

def train_random_forest(X, y, out_path=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    acc = accuracy_score(y_test, preds > 0.5)
    f1 = f1_score(y_test, preds > 0.5)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, out_path)
    return {'model': clf, 'auc': float(auc), 'accuracy': float(acc), 'f1': float(f1)}


def train_xgboost(X, y, out_path=None):
    if not XGBOOST_AVAILABLE:
        raise RuntimeError('xgboost is not available')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    acc = accuracy_score(y_test, preds > 0.5)
    f1 = f1_score(y_test, preds > 0.5)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, out_path)
    return {'model': clf, 'auc': float(auc), 'accuracy': float(acc), 'f1': float(f1)}
