"""Load processed ABIDE subjects into arrays and PyG graphs."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]


def load_all(run_name="full"):
    """Load data/processed/<run_name>/ written by `prepare_data.py --run-name <run_name>`."""
    processed = ROOT / "data" / "processed" / run_name
    summary = pd.read_csv(processed / "subjects.csv")
    corrs, graphs, y = [], [], []
    for row in summary.itertuples():
        z = np.load(processed / f"{row.file_id}.npz")
        corr = z["corr"].astype(np.float32)
        label = int(row.label == 1)  # ASD = 1, control = 0
        corrs.append(corr)
        y.append(label)
        graphs.append(Data(
            x=torch.from_numpy(corr),  # node features = correlation profile
            edge_index=torch.from_numpy(z["edge_index"]).long(),
            edge_weight=torch.from_numpy(z["edge_weight"]).float(),   # |corr|
            edge_signed=torch.from_numpy(z["edge_signed"]).float(),   # signed corr
            y=torch.tensor([float(label)]),
        ))
    return summary, np.stack(corrs), np.array(y), graphs
