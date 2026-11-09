"""Tiny subject-level logistic-regression smoke test, not final evaluation."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

summary = pd.read_csv(ROOT / "data" / "processed" / "subjects.csv")
Xs, ys = [], []
for row in summary.itertuples():
    corr = np.load(ROOT / "data" / "processed" / f"{row.file_id}.npz")["corr"]
    u, v = np.triu_indices(corr.shape[0], 1)
    Xs.append(corr[u, v])
    ys.append(int(row.label == 1))  # ASD = 1
X, y = np.asarray(Xs), np.asarray(ys)
if len(np.unique(y)) != 2 or min(np.bincount(y)) < 2:
    raise RuntimeError("Need at least 2 subjects per class; increase --n-subjects")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)
p = model.predict_proba(X_test)[:, 1]
pred = (p >= 0.5).astype(int)
print("Pilot only — n_train, n_test:", len(y_train), len(y_test))
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("Balanced accuracy:", round(balanced_accuracy_score(y_test, pred), 3))
if len(np.unique(y_test)) == 2:
    print("ROC-AUC:", round(roc_auc_score(y_test, p), 3))
