"""Single train/val/test split (or 10-fold CV) for site-only LR / edge LR / GNN, + GNNExplainer & edge-removal fidelity.

Usage: python scripts/run_experiments.py --scheme {random,site} --holdout --config z_flat [--no-edges]
GNN model selection inside src.train uses a validation split carved from TRAIN only.
Config choice itself is made by scripts/select_gcn.py on the train portion only.
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch_geometric.explain import Explainer, GNNExplainer
from src.data import load_all
from src.train import CONFIGS, prep_graphs, train_model, predict

ap = argparse.ArgumentParser()
ap.add_argument("--scheme", choices=["random", "site"], required=True)
ap.add_argument("--run-name", default="full", help="processed-data run (data/processed/<run-name>)")
ap.add_argument("--config", choices=list(CONFIGS), required=True)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--explainer-epochs", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model-seed", type=int, default=None, help="seed for GNN init/val split/explainer; split stays fixed by --seed")
ap.add_argument("--holdout", action="store_true")
ap.add_argument("--holdout-fold", type=int, default=0, help="which of the 5 folds is the test set (0 = original)")
ap.add_argument("--no-edges", action="store_true", help="ablation: delete ALL edges, skip explainer")
args = ap.parse_args()
cfg = CONFIGS[args.config]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = ROOT / "results" / args.run_name / (("holdout_" if args.holdout else "") + args.scheme + "_" + args.config
                          + ("_noedges" if args.no_edges else "") + (f"_split{args.seed}" if args.seed else "") + (f"_f{args.holdout_fold}" if args.holdout_fold else "") + (f"_ms{args.model_seed}" if args.model_seed is not None else ""))
out.mkdir(parents=True, exist_ok=True)
timings = {}

def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(acc=accuracy_score(y, pred), bacc=balanced_accuracy_score(y, pred),
                auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"))

summary, corrs, y, graphs = load_all(args.run_name)
sites = summary["site"].astype(str).values
N, R = len(y), corrs.shape[1]
iu = np.triu_indices(R, 1)
X = corrs[:, iu[0], iu[1]]
k_ = 5 if args.holdout else 10
if args.scheme == "random":
    splits = list(StratifiedKFold(k_, shuffle=True, random_state=args.seed).split(X, y))
else:
    splits = list(StratifiedGroupKFold(k_, shuffle=True, random_state=args.seed).split(X, y, groups=sites))
if args.holdout:
    splits = splits[args.holdout_fold:args.holdout_fold + 1]

p_lr, p_site, p_gnn = (np.full(N, np.nan) for _ in range(3))
fold_of = np.full(N, -1)
masks = np.full((N, R, R), np.nan, dtype=np.float16)
fid = {k: np.full(N, np.nan) for k in ["p0", "p_top10", "p_rand10", "p_bot10", "p_top25", "p_rand25", "p_bot25"]}
fold_rows = []

@torch.no_grad()
def prob_removed(model, g, drop_unique):
    m = g.edge_index.size(1) // 2
    keep = torch.ones(m, dtype=torch.bool); keep[drop_unique] = False
    k2 = torch.cat([keep, keep])
    ez = g.edge_z[k2].to(dev) if hasattr(g, "edge_z") else None
    return torch.sigmoid(model(g.x.to(dev), g.edge_index[:, k2].to(dev), None,
                               g.edge_weight[k2].to(dev), ez)).item()

torch.manual_seed(args.model_seed or 0)
rng = np.random.default_rng(args.seed if args.model_seed is None else args.model_seed)
for k, (tr, te) in enumerate(splits):
    fold_of[te] = k
    row = dict(fold=k, n_train=len(tr), n_test=len(te), sites_test=sorted(set(sites[te])),
               test_asd_frac=float(y[te].mean()))
    t0 = time.time()
    lr = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1], cv=3, max_iter=2000, scoring="roc_auc", n_jobs=8))
    lr.fit(X[tr], y[tr]); p_lr[te] = lr.predict_proba(X[te])[:, 1]
    timings["lr"] = timings.get("lr", 0) + time.time() - t0
    oh = OneHotEncoder(handle_unknown="ignore").fit(sites[tr].reshape(-1, 1))
    sm = LogisticRegression(max_iter=1000).fit(oh.transform(sites[tr].reshape(-1, 1)), y[tr])
    p_site[te] = sm.predict_proba(oh.transform(sites[te].reshape(-1, 1)))[:, 1]

    t0 = time.time()
    G = prep_graphs(graphs, corrs, tr, cfg, no_edges=args.no_edges)
    model, val_auc = train_model(G, y, tr, cfg, seed=(args.seed if args.model_seed is None else args.model_seed) + k, dev=dev, epochs=args.epochs)
    p_gnn[te] = predict(model, G, te, dev)
    timings["gnn_train"] = timings.get("gnn_train", 0) + time.time() - t0
    row.update(val_auc_gcn=val_auc,
               **{f"lr_{a}": b for a, b in metrics(y[te], p_lr[te]).items()},
               **{f"site_{a}": b for a, b in metrics(y[te], p_site[te]).items()},
               **{f"gcn_{a}": b for a, b in metrics(y[te], p_gnn[te]).items()})
    msg = (f"[{args.scheme}/{args.config}{'/noedges' if args.no_edges else ''}] fold {k}: n_test={len(te)} "
           f"GNN auc={row['gcn_auc']:.3f} bacc={row['gcn_bacc']:.3f} (val {val_auc:.3f}) "
           f"LR auc={row['lr_auc']:.3f} | GNN p-range [{p_gnn[te].min():.2f},{p_gnn[te].max():.2f}]")
    if not args.no_edges:
        t0 = time.time()
        expl = Explainer(model=model, algorithm=GNNExplainer(epochs=args.explainer_epochs),
                         explanation_type="model", node_mask_type=None, edge_mask_type="object",
                         model_config=dict(mode="binary_classification", task_level="graph", return_type="raw"))
        for i in te:
            g = G[i]; m = g.edge_index.size(1) // 2
            gd = g.clone().to(dev)
            em = expl(gd.x, gd.edge_index, batch=None, edge_weight=gd.edge_weight,
                      edge_z=getattr(gd, "edge_z", None)).edge_mask.detach().cpu()
            und = 0.5 * (em[:m] + em[m:])
            u, v = g.edge_index[0, :m].numpy(), g.edge_index[1, :m].numpy()
            masks[i, u, v] = und.numpy(); masks[i, v, u] = und.numpy()
            order = torch.argsort(und, descending=True)
            fid["p0"][i] = prob_removed(model, g, torch.tensor([], dtype=torch.long))
            for frac, tag in [(0.10, "10"), (0.25, "25")]:
                kk = int(round(frac * m))
                fid[f"p_top{tag}"][i] = prob_removed(model, g, order[:kk])
                fid[f"p_bot{tag}"][i] = prob_removed(model, g, order[-kk:])
                fid[f"p_rand{tag}"][i] = prob_removed(model, g, torch.from_numpy(rng.choice(m, kk, replace=False)))
        timings["explain"] = timings.get("explain", 0) + time.time() - t0
        msg += f" ({time.time()-t0:.0f}s explain)"
    fold_rows.append(row)
    print(msg, flush=True)

np.savez_compressed(out / "predictions.npz", y=y, fold=fold_of, p_lr=p_lr, p_site=p_site,
                    p_gcn=p_gnn, masks=masks, **fid)
json.dump(dict(folds=fold_rows, timings_sec=timings, args=vars(args), cfg=cfg),
          open(out / "cv_results.json", "w"), indent=1, default=float)
print("timings:", timings)
