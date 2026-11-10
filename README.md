# Which Connections Matter? Explaining GNN Predictions of Autism on ABIDE

Starter pipeline for ABIDE I / PCP / C-PAC / filt_noglobal / rois_aal.

## Quickstart

```bash
python -m venv .venv
# Activate: Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py --n-subjects 20
python scripts/baseline.py
```

If Python and pandas issues persist, install Python 3.14.10 and pandas 2.3.3

The first script downloads regional time series, builds Pearson correlation matrices,
creates sparse graphs, writes a class/site summary, and saves a sample connectivity
heatmap. The baseline trains/evaluates a logistic-regression classifier on a
subject-level holdout split. These are pilot experiments, **not** final results.

## Full experiment: explainable edge model

Results and discussion: [REPORT.md](REPORT.md). Summary figures and tables: `results/full/final_v3/`
(main test fold) and `results/full/final_v3_f1/` (disjoint confirmation fold).

```bash
python scripts/prepare_data.py --n-subjects 1112 --run-name full      # all 817 usable subjects
python scripts/select_gcn.py --scheme random                          # config search, train portion only
python scripts/select_gcn.py --scheme site
for m in 0 1 2 3 4; do                                                 # 5 seeds; add --holdout-fold 1 for confirmation
  python scripts/run_experiments.py --scheme random --holdout --config lin_d30_wl0.1 --model-seed $m
  python scripts/run_experiments.py --scheme site   --holdout --config hyb_d10_wl1   --model-seed $m
done
python scripts/analyze_final.py              # -> results/full/final_v3/
python scripts/analyze_final.py --fold 1     # -> results/full/final_v3_f1/
```

- `src/models.py`: `EdgeLinear` (per-edge linear path written as message passing, so GNNExplainer masks gate it),
  `HybridGNN` (edge path + optional edge-only GNN), `BrainGNN`.
- `src/train.py`: configuration grid, graph preparation with train-only statistics, training loop.
- `scripts/run_experiments.py`: logistic-regression and site-only baselines, model training, GNNExplainer,
  edge-removal fidelity. Per-run outputs (`results/full/holdout_*`) are git-ignored.
- `scripts/analyze_final.py`: 5-seed ensemble AUC with bootstrap CIs, fidelity, explanation stability,
  site-adjusted group t-tests (FDR) and enrichment.

## Data provenance
- Collection: ABIDE I
- Source: Preprocessed Connectomes Project (PCP)
- Pipeline: C-PAC
- Strategy: band-pass filtered, no global signal regression (`filt_noglobal`)
- Derivative: AAL ROI time series (`rois_aal`)
- Subject selection: Nilearn quality_checked=True, downloaded order for pilot
- Matrix: Pearson correlations, `np.corrcoef(ts, rowvar=False)`
- Sparsification: top 10% absolute off-diagonal edges, undirected
- Pilot GCN edge weights: absolute correlation magnitudes (nonnegative);
  signed weights are retained separately in matrix files for later analyses.

## Scope and caveats
- The pilot scripts above (`prepare_data.py` with 20 subjects, `baseline.py`) are pipeline checks only;
  the full experiment (graph models, GNNExplainer, edge-removal tests, group inference) is described in REPORT.md.
- A pilot sample of 20 is for pipeline validation, not clinical performance.
- Before final experiments, document cohort inclusion/exclusion, use full data,
  stratify diagnosis and consider scanner-site effects.
- Correlation-based node features and adjacency duplicate information. Edge-only
  masking will not remove all information about a pair of ROIs.
- Preserve site-aware splits and do not tune on the final test set.

Official data documentation: https://preprocessed-connectomes-project.org/abide/download.html
