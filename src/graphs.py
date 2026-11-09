import numpy as np


def correlation_matrix(timeseries):
    """Build a symmetric ROI x ROI Pearson correlation matrix."""
    ts = np.asarray(timeseries, dtype=float)
    if ts.ndim != 2 or ts.shape[0] < 3:
        raise ValueError(f"Expected time x ROI matrix, got {ts.shape}")
    if not np.isfinite(ts).all():
        raise ValueError("Non-finite input time series")
    if np.any(np.std(ts, axis=0) < 1e-12):
        raise ValueError("Constant ROI signal in time series")
    corr = np.corrcoef(ts, rowvar=False)
    if not np.isfinite(corr).all():
        raise ValueError("Non-finite correlations")
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


def sparse_graph_from_corr(corr, density=0.10):
    """Returns (edge_index, nonnegative GCN weights, signed edge correlations).

    Uses unique upper-triangle edges to retain the strongest absolute correlations,
    then duplicates them in both directions for PyG.
    """
    r = corr.shape[0]
    if corr.shape != (r, r) or not 0 < density <= 1:
        raise ValueError("corr must be square and density within (0, 1]")
    u, v = np.triu_indices(r, k=1)
    values = corr[u, v]
    k = max(1, int(np.ceil(density * len(values))))
    pick = np.argsort(np.abs(values), kind="stable")[-k:]
    src, dst = u[pick], v[pick]
    signed = values[pick]
    edge_index = np.stack([np.r_[src, dst], np.r_[dst, src]])
    signed_edges = np.r_[signed, signed].astype(np.float32)
    # Baseline GCN: nonnegative strengths avoid invalid degree normalization.
    edge_weight = np.abs(signed_edges).astype(np.float32)
    return edge_index.astype(np.int64), edge_weight, signed_edges
