"""Choose a GCN configuration by 3-fold CV INSIDE the training portion of the holdout split (test untouched)."""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from src.data import load_all
from src.train import CONFIGS, prep_graphs, train_model, predict

ap = argparse.ArgumentParser(); ap.add_argument("--scheme", choices=["random", "site"], required=True)
ap.add_argument("--run-name", default="full", help="processed-data run (data/processed/<run-name>)")
ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--only", nargs="*", default=None)
args = ap.parse_args()
dev = torch.device("cuda")
RES = ROOT / "results" / args.run_name; RES.mkdir(parents=True, exist_ok=True)
summary, corrs, y, graphs = load_all(args.run_name); sites = summary["site"].astype(str).values
X = np.zeros((len(y), 1))
if args.scheme == "random":
    tr, _ = next(iter(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)))
    inner = list(StratifiedKFold(3, shuffle=True, random_state=1).split(tr, y[tr]))
else:
    tr, _ = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups=sites)))
    inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=1).split(tr, y[tr], groups=sites[tr]))
path = RES / f"gcn_selection_{args.scheme}.json"
out = json.load(open(path))["configs"] if path.exists() else {}
if "LR_reference" not in out:   # linear reference on the same inner folds (C fixed, no tuning)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    iu = np.triu_indices(corrs.shape[1], 1); Xe = corrs[:, iu[0], iu[1]]; aucs = []
    for a, b in inner:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1e-3, max_iter=2000)).fit(Xe[tr[a]], y[tr[a]])
        aucs.append(float(roc_auc_score(y[tr[b]], m.predict_proba(Xe[tr[b]])[:, 1])))
    out["LR_reference"] = dict(inner_aucs=aucs, mean=float(np.mean(aucs)), sd=float(np.std(aucs)), sec=0)
    print(f"[{args.scheme}] LR reference inner AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}", flush=True)
for name, cfg in CONFIGS.items():
    if args.only is not None and name not in args.only: continue
    t0 = time.time(); aucs = []
    for k, (a, b) in enumerate(inner):
        G = prep_graphs(graphs, corrs, tr[a], cfg)
        model, _ = train_model(G, y, tr[a], cfg, seed=k, dev=dev, epochs=args.epochs)
        p = predict(model, G, tr[b], dev); aucs.append(float(roc_auc_score(y[tr[b]], p)))
    out[name] = dict(inner_aucs=aucs, mean=float(np.mean(aucs)), sd=float(np.std(aucs)), sec=time.time() - t0)
    print(f"[{args.scheme}] {name:16s} inner AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  {aucs}  ({time.time()-t0:.0f}s)", flush=True)
best = max((k for k in out if k != "LR_reference"), key=lambda k: out[k]["mean"])
json.dump(dict(best=best, configs=out), open(RES / f"gcn_selection_{args.scheme}.json", "w"), indent=1)
print("BEST:", best)
