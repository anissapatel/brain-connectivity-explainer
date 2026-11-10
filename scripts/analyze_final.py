"""Final v3 analysis over 5 model seeds (same train/test split per scheme).

Pre-specified before looking at test results:
  * primary model per scheme = best inner-CV config (scripts/select_gcn.py); the other run is an ablation (+/- GNN part)
  * report per-seed test AUC (mean ± SD) and the 5-seed ensemble (mean probability) with bootstrap CI
  * explanation = GNNExplainer mask averaged over seeds; fidelity / stability reported per seed
"""
import json, sys
from itertools import combinations
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from src.data import load_all

RUNS = {"random": {"primary": "lin_d30_wl0.1", "ablation": "hyb_d30_wl0.1"},
        "site":   {"primary": "hyb_d10_wl1",   "ablation": "lin_d10_wl1"}}
SEEDS = range(5)
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--fold", type=int, default=0)
_ap.add_argument("--run-name", default="full"); _args = _ap.parse_args(); FOLD, RUN = _args.fold, _args.run_name
SUF = f"_f{FOLD}" if FOLD else ""
res = ROOT / "results" / RUN; OUT = res / ("final_v3" + SUF); OUT.mkdir(exist_ok=True)
summary, corrs, y, _ = load_all(RUN); sites = summary["site"].astype(str).values
R = 116; iu = np.triu_indices(R, 1); rng = np.random.default_rng(0)

def agg_mean(M, minc):
    cnt = (~np.isnan(M)).sum(0)
    with np.errstate(all="ignore"): out = np.nanmean(M, 0)
    out[cnt < minc] = np.nan; return out
def topk(v, k=100): return set(np.argsort(-np.nan_to_num(v, nan=-np.inf))[:k])
def boot(yt, ps, n=2000):
    """ps: list of prob vectors; returns CI per vector and of the first-minus-second difference"""
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[i])) == 2: vals.append([roc_auc_score(yt[i], p[i]) for p in ps])
    return np.array(vals)

# independent evidence: site-adjusted group t, LR coefficients (all subjects; used only for agreement)
Z = np.arctanh(np.clip(corrs[:, iu[0], iu[1]], -0.999, 0.999)); Zr = Z.copy()
for s in np.unique(sites): Zr[sites == s] -= Zr[sites == s].mean(0)
t, pv = stats.ttest_ind(Zr[y == 1], Zr[y == 0], equal_var=False)
qv = multipletests(pv, method="fdr_bh")[1]
X = corrs[:, iu[0], iu[1]]
coef = LogisticRegression(C=1e-3, max_iter=3000).fit(StandardScaler().fit_transform(X), y).coef_[0]
try:
    from nilearn.datasets import fetch_atlas_aal
    _a = fetch_atlas_aal(version="SPM12")
    labels = list(_a.labels); codes = [int(c) for c in _a.indices]
    if labels[0] == "Background": labels, codes = labels[1:], codes[1:]
    # PCP rois_aal columns are ordered by ascending AAL code (#2001, #2002, ...), checked in the .1D header
    assert len(labels) == R and codes == sorted(codes)
except Exception:
    labels = [f"ROI{i+1}" for i in range(R)]
def ename(e): return f"{labels[iu[0][e]]} – {labels[iu[1][e]]}"

S = {"labels_source": "AAL SPM12 (nilearn)" if not labels[0].startswith("ROI") else "indices"}
perf_rows, fid_rows = [], []
for scheme, runs in RUNS.items():
    for role, cfg in runs.items():
        if not (res / f"holdout_{scheme}_{cfg}{SUF}_ms0").exists():
            print("skip (not run):", scheme, cfg); continue
        Ps = [np.load(res / f"holdout_{scheme}_{cfg}{SUF}_ms{m}" / "predictions.npz") for m in SEEDS]
        te = np.where(Ps[0]["fold"] == 0)[0]; yt = y[te]
        for P in Ps: assert np.array_equal(np.where(P["fold"] == 0)[0], te)
        p_seed = [P["p_gcn"][te] for P in Ps]; p_ens = np.mean(p_seed, 0); p_lr = Ps[0]["p_lr"][te]
        a_seed = [roc_auc_score(yt, p) for p in p_seed]
        B = boot(yt, [p_ens, p_lr]); d = B[:, 0] - B[:, 1]
        perf_rows.append(dict(scheme=scheme, role=role, config=cfg,
            auc_seeds_mean=np.mean(a_seed), auc_seeds_sd=np.std(a_seed), auc_seeds=[round(a, 3) for a in a_seed],
            bacc_seeds_mean=np.mean([balanced_accuracy_score(yt, p >= .5) for p in p_seed]),
            auc_ensemble=roc_auc_score(yt, p_ens), auc_ens_ci=list(np.percentile(B[:, 0], [2.5, 97.5]).round(3)),
            bacc_ensemble=balanced_accuracy_score(yt, p_ens >= .5),
            auc_lr=roc_auc_score(yt, p_lr), auc_lr_ci=list(np.percentile(B[:, 1], [2.5, 97.5]).round(3)),
            ens_minus_lr=float(d.mean()), ens_minus_lr_ci=list(np.percentile(d, [2.5, 97.5]).round(3)),
            p_ens_minus_lr_le0=float((d <= 0).mean())))
        # fidelity per seed
        for frac in ["10", "25"]:
            for kind in ["top", "rand", "bot"]:
                drops, flips = [], []
                for P in Ps:
                    p0 = P["p0"][te]; pa = P[f"p_{kind}{frac}"][te]; pred = p0 >= .5
                    drops.append(np.mean(np.where(pred, p0, 1 - p0) - np.where(pred, pa, 1 - pa)))
                    flips.append(np.mean((pa >= .5) != pred))
                fid_rows.append(dict(scheme=scheme, config=cfg, removed=f"{frac}%", edges=kind,
                                     drop_mean=np.mean(drops), drop_sd=np.std(drops),
                                     flip_mean=np.mean(flips), flip_sd=np.std(flips)))
        # stability across seeds + seed-averaged explanation
        Ms = [P["masks"][te][:, iu[0], iu[1]].astype(np.float32) for P in Ps]
        A = [agg_mean(M, 15) for M in Ms]
        sp, jc, ps_ = [], [], []
        for i, j in combinations(range(len(Ms)), 2):
            ok = ~np.isnan(A[i]) & ~np.isnan(A[j])
            sp.append(stats.spearmanr(A[i][ok], A[j][ok])[0]); jc.append(len(topk(A[i]) & topk(A[j])) / len(topk(A[i]) | topk(A[j])))
            r = []
            for s_ in range(len(te)):
                ok = ~np.isnan(Ms[i][s_]) & ~np.isnan(Ms[j][s_])
                if ok.sum() > 10: r.append(stats.spearmanr(Ms[i][s_][ok], Ms[j][s_][ok])[0])
            ps_.append(np.nanmean(r))
        avg = np.nanmean(np.stack(A), 0)            # seed-averaged explanation
        ok = ~np.isnan(avg)
        # split-half reliability of the seed-averaged explanation across seeds (seeds {0,1} vs {2,3})
        h1, h2 = np.nanmean(np.stack(A[:2]), 0), np.nanmean(np.stack(A[2:4]), 0); okh = ~np.isnan(h1) & ~np.isnan(h2)
        S[f"{scheme}/{cfg}"] = dict(
            cross_seed_spearman=float(np.mean(sp)), cross_seed_spearman_sd=float(np.std(sp)),
            cross_seed_top100_jaccard=float(np.mean(jc)), per_subject_spearman=float(np.mean(ps_)),
            seedpair_avg_spearman_2v2=float(stats.spearmanr(h1[okh], h2[okh])[0]),
            chance_top100_jaccard=float((100 * 100 / 6670) / (200 - 100 * 100 / 6670)),
            n_edges_scored=int(ok.sum()),
            agree_spearman_abs_t=float(stats.spearmanr(avg[ok], np.abs(t)[ok])[0]),
            agree_spearman_abs_lrcoef=float(stats.spearmanr(avg[ok], np.abs(coef)[ok])[0]),
            top100_overlap_t=len(topk(avg) & topk(np.abs(t))), top100_overlap_lrcoef=len(topk(avg) & topk(np.abs(coef))),
            top100_explainer_frac_fdr_sig=float(np.mean(qv[list(topk(avg))] < 0.05)),
            base_rate_fdr_sig_among_scored=float(np.mean(qv[ok] < 0.05)),
            enrichment_hypergeom_p=float(stats.hypergeom.sf(int((qv[list(topk(avg))] < .05).sum()) - 1,
                                                            int(ok.sum()), int((qv[ok] < .05).sum()), 100)),
            chance_overlap_top100=100 * 100 / 6670, base_rate_fdr_sig=float(np.mean(qv < 0.05)))
        if role == "primary":
            top = list(np.argsort(-np.nan_to_num(avg, nan=-np.inf))[:20])
            pd.DataFrame(dict(edge=[ename(e) for e in top], mean_mask=avg[top], group_t=t[top], q_fdr=qv[top],
                              lr_coef=coef[top])).to_csv(OUT / f"top_edges_{scheme}.csv", index=False)
            np.save(OUT / f"seedavg_mask_{scheme}.npy", avg)

perf = pd.DataFrame(perf_rows); fid = pd.DataFrame(fid_rows)
perf.to_csv(OUT / "performance.csv", index=False); fid.to_csv(OUT / "fidelity.csv", index=False)
top_t = np.argsort(-np.abs(t))[:20]
pd.DataFrame(dict(edge=[ename(e) for e in top_t], t=t[top_t], q_fdr=qv[top_t])).to_csv(OUT / "top_edges_group_t.csv", index=False)
S["group"] = dict(n_sig=int((qv < .05).sum()), top20_all_negative=bool((t[top_t] < 0).all()))
json.dump(S, open(OUT / "summary.json", "w"), indent=1, default=float)

# figures
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, scheme in zip(ax, ["random", "site"]):
    d = perf[perf.scheme == scheme]
    names = ["Edge LR"] + [f"{r.config}\n({r.role})" for r in d.itertuples()]
    a.bar(0, d.auc_lr.iloc[0], color="#4c78a8")
    for k, r in enumerate(d.itertuples(), 1):
        a.bar(k, r.auc_ensemble, color="#e45756" if r.role == "primary" else "#f2a39e")
        a.scatter([k] * 5, r.auc_seeds, c="k", s=10, zorder=3)
    a.set_xticks(range(len(names))); a.set_xticklabels(names, fontsize=8); a.axhline(.5, ls="--", c="grey", lw=.8)
    a.set_title({"random": "Stratified random split", "site": "Site-held-out (NYU)"}[scheme]); a.set_ylim(.4, .85)
ax[0].set_ylabel("Test ROC-AUC (bar: 5-seed ensemble; dots: seeds)")
plt.tight_layout(); plt.savefig(OUT / "fig_auc.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for a, scheme in zip(ax, ["random", "site"]):
    d = fid[(fid.scheme == scheme) & (fid.config == RUNS[scheme]["primary"])]
    for j, kind in enumerate(["top", "rand", "bot"]):
        dd = d[d.edges == kind]
        a.bar(np.arange(2) + (j - 1) * .27, dd.flip_mean, .27, yerr=dd.flip_sd, capsize=3,
              label={"top": "most important", "rand": "random", "bot": "least important"}[kind])
    a.set_xticks(range(2)); a.set_xticklabels(["remove 10%", "remove 25%"]); a.set_title(f"{scheme}: {RUNS[scheme]['primary']}")
ax[0].set_ylabel("Fraction of predictions flipped\n(mean ± SD over 5 seeds)"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig_fidelity.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
def full(v):
    M = np.full((R, R), np.nan); M[iu] = v; M[(iu[1], iu[0])] = v; return M
for a, (v, ttl, cm) in zip(ax, [(np.load(OUT / "seedavg_mask_site.npy"), "Seed-averaged edge importance\n(site split, primary model)", "viridis"),
                                (np.load(OUT / "seedavg_mask_random.npy"), "Seed-averaged edge importance\n(random split, primary model)", "viridis"),
                                (t, "Site-adjusted group t (ASD − TC)", "coolwarm")]):
    im = a.imshow(full(v), cmap=cm); a.set_title(ttl, fontsize=10); plt.colorbar(im, ax=a, fraction=.046)
plt.tight_layout(); plt.savefig(OUT / "fig_edges.png", dpi=150); plt.close()

pd.set_option("display.width", 250)
print(perf.drop(columns=["auc_seeds"]).round(3).to_string(index=False))
print(fid.round(3).to_string(index=False))
print(json.dumps(S, indent=1, default=float))
