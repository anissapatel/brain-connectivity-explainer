"""Shared GNN training utilities (graph prep with train-only statistics, training loop, prediction)."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from src.models import BrainGNN, HybridGNN
from src.graphs import sparse_graph_from_corr

CONFIGS = {
    "z_mean":         dict(feat="corrz", readout="mean",    conv="gcn",   signed=False),
    "z_flat":         dict(feat="corrz", readout="flatten", conv="gcn",   signed=False),
    "z_flat_signed":  dict(feat="corrz", readout="flatten", conv="graph", signed=True),
    "id_mean_signed": dict(feat="id",    readout="mean",    conv="graph", signed=True),
    "id_flat_signed": dict(feat="id",    readout="flatten", conv="graph", signed=True),
    # stronger regularisation: small hidden size, heavy weight decay, high dropout
    "z_flat_reg":     dict(feat="corrz", readout="flatten", conv="gcn",   signed=False, hidden=16, wd=1e-2, dropout=0.7),
    "z_mean_reg":     dict(feat="corrz", readout="mean",    conv="gcn",   signed=False, hidden=16, wd=1e-2, dropout=0.7),
    "id_mean_reg":    dict(feat="id",    readout="mean",    conv="graph", signed=True,  hidden=16, wd=1e-2, dropout=0.7),
    "id_flat_reg":    dict(feat="id",    readout="flatten", conv="graph", signed=True,  hidden=16, wd=1e-2, dropout=0.7),
    # v3: linear edge path inside the graph (explainable), optionally + edge-only GNN; graph density 10% / 30%
    "id_mean_signed_d30": dict(feat="id", readout="mean", conv="graph", signed=True, density=0.3),
}
for d in (0.1, 0.3):
    for wl in (0.1, 1.0):
        for g in (False, True):
            CONFIGS[f"{'hyb' if g else 'lin'}_d{int(d*100)}_wl{wl:g}"] = dict(
                model="hybrid", use_gnn=g, feat="id", readout="mean", conv="graph", signed=True,
                density=d, wd_lin=wl)

_GRAPH_CACHE = {}


def graphs_at_density(graphs, corrs, density):
    """Rebuild graphs at another top-|r| density (same conventions as prepare_data.py)."""
    if abs(density - 0.1) < 1e-9:
        return graphs
    if density not in _GRAPH_CACHE:
        out = []
        for g, c in zip(graphs, corrs):
            ei, w, sg = sparse_graph_from_corr(c, density=density)
            h = g.clone()
            h.edge_index = torch.from_numpy(ei).long()
            h.edge_weight = torch.from_numpy(w).float()
            h.edge_signed = torch.from_numpy(sg).float()
            out.append(h)
        _GRAPH_CACHE[density] = out
    return _GRAPH_CACHE[density]


def prep_graphs(graphs, corrs, tr_idx, cfg, no_edges=False):
    """Copy graphs; standardise node features with TRAIN statistics; pick abs/signed edge weights.
    For hybrid models add edge_z = Fisher-z of the edge standardised with TRAIN per-pair mean/SD."""
    graphs = graphs_at_density(graphs, corrs, cfg.get("density", 0.1))
    mu = corrs[tr_idx].mean(0)
    sd = corrs[tr_idx].std(0) + 1e-6
    if cfg.get("model") == "hybrid":
        Zf = np.arctanh(np.clip(corrs, -0.999, 0.999))
        Zs = ((Zf - Zf[tr_idx].mean(0)) / (Zf[tr_idx].std(0) + 1e-6)).astype(np.float32)
    out = []
    for i, g in enumerate(graphs):
        h = g.clone()
        if cfg.get("model") == "hybrid":
            u, v = g.edge_index[0].numpy(), g.edge_index[1].numpy()
            h.edge_z = torch.from_numpy(Zs[i, u, v])
        if cfg["feat"] == "corrz":
            h.x = torch.from_numpy(((corrs[i] - mu) / sd).astype(np.float32))
        if cfg["signed"]:
            h.edge_weight = g.edge_signed.clone()
        if no_edges:
            h.edge_index = torch.zeros((2, 0), dtype=torch.long)
            h.edge_weight = torch.zeros(0)
            h.edge_signed = torch.zeros(0)
            if hasattr(h, "edge_z"):
                h.edge_z = torch.zeros(0)
        out.append(h)
    return out


def build_model(cfg):
    if cfg.get("model") == "hybrid":
        return HybridGNN(use_gnn=cfg["use_gnn"], feat=cfg["feat"], readout=cfg["readout"], conv=cfg["conv"],
                         hidden=cfg.get("hidden", 64), dropout=cfg.get("dropout", 0.5))
    return BrainGNN(feat=cfg["feat"], readout=cfg["readout"], conv=cfg["conv"],
                    hidden=cfg.get("hidden", 64), dropout=cfg.get("dropout", 0.5))


def _fwd(model, b):
    return model(b.x, b.edge_index, b.batch, b.edge_weight, getattr(b, "edge_z", None))


@torch.no_grad()
def predict(model, G, idx, dev):
    model.eval(); ps = []
    for b in DataLoader([G[i] for i in idx], batch_size=128):
        ps.append(torch.sigmoid(_fwd(model, b.to(dev))).squeeze(1).cpu())
    return torch.cat(ps).numpy()


def train_model(G, y, tr_idx, cfg, seed, dev, epochs=100, bs=32):
    """Train on tr_idx; 15% of it is a validation split used ONLY to pick the best epoch."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = train_test_split(tr_idx, test_size=0.15, stratify=y[tr_idx], random_state=seed)
    model = build_model(cfg).to(dev)
    lin = [p for n, p in model.named_parameters() if n.startswith("lin.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("lin.")]
    groups = [dict(params=rest, weight_decay=cfg.get("wd", 5e-4))]
    if lin:
        groups.append(dict(params=lin, weight_decay=cfg.get("wd_lin", 1.0)))
    opt = torch.optim.Adam(groups, lr=cfg.get("lr", 1e-3))
    tl = DataLoader([G[i] for i in tr], batch_size=bs, shuffle=True, drop_last=True)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for b in tl:
            b = b.to(dev); opt.zero_grad()
            F.binary_cross_entropy_with_logits(_fwd(model, b).squeeze(1), b.y).backward()
            opt.step()
        auc = roc_auc_score(y[va], predict(model, G, va, dev))
        if auc > best:
            best, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    return model, best
