"""Tiny subject-level logistic-regression smoke test, not final evaluation."""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Project root directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run-name", type=str, default="pilot_100")
args = parser.parse_args()

# Locate processed data for this run
processed = ROOT / "data" / "processed" / args.run_name

# Load participant metadata
summary = pd.read_csv(processed / "subjects.csv")

# Extract features and labels
Xs, ys = [], []

for row in summary.itertuples():
    corr = np.load(processed / f"{row.file_id}.npz")["corr"]

    # Use only upper triangle to avoid duplicate connections
    u, v = np.triu_indices(corr.shape[0], 1)

    Xs.append(corr[u, v])
    ys.append(int(row.label == 1))  # ASD = 1, Control = 0

X, y = np.asarray(Xs), np.asarray(ys)

# Ensure both classes are represented
if len(np.unique(y)) != 2 or min(np.bincount(y)) < 2:
    raise RuntimeError(
        "Need at least 2 subjects per class; increase --n-subjects"
    )

# Subject-level train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Logistic regression baseline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

model.fit(X_train, y_train)

# Predictions
p = model.predict_proba(X_test)[:, 1]
pred = (p >= 0.5).astype(int)

# Evaluation
print("Pilot only — n_train, n_test:", len(y_train), len(y_test))
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("Balanced accuracy:", round(balanced_accuracy_score(y_test, pred), 3))

if len(np.unique(y_test)) == 2:
    print("ROC-AUC:", round(roc_auc_score(y_test, p), 3))
