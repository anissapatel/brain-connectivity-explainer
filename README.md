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

The first script downloads regional time series, builds Pearson correlation matrices,
creates sparse graphs, writes a class/site summary, and saves a sample connectivity
heatmap. The baseline trains/evaluates a logistic-regression classifier on a
subject-level holdout split. These are pilot experiments, **not** final results.

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
- No full 10-fold CV, GCN training, GNNExplainer, edge masking, or ASD-vs-control
  group inference is included yet.
- A pilot sample of 20 is for pipeline validation, not clinical performance.
- Before final experiments, document cohort inclusion/exclusion, use full data,
  stratify diagnosis and consider scanner-site effects.
- Correlation-based node features and adjacency duplicate information. Edge-only
  masking will not remove all information about a pair of ROIs.
- Preserve site-aware splits and do not tune on the final test set.

Official data documentation: https://preprocessed-connectomes-project.org/abide/download.html
