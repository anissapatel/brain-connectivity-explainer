"""Download a small real ABIDE sample and build connectivity graphs.
Run from repository root: python scripts/prepare_data.py --n-subjects 20
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_abide_pcp
from src.graphs import correlation_matrix, sparse_graph_from_corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=20)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this processing run"
    )
    args = parser.parse_args()

    run_name = args.run_name or f"pilot_{args.n_subjects}"

    raw = ROOT / "data" / "raw"
    out = ROOT / "data" / "processed" / run_name
    results = ROOT / "results" / run_name

    for folder in (raw, out, results):
        folder.mkdir(parents=True, exist_ok=True)

    dataset = fetch_abide_pcp(
        data_dir=str(raw), n_subjects=args.n_subjects, pipeline="cpac",
        band_pass_filtering=True, global_signal_regression=False,
        derivatives=["rois_aal"], quality_checked=True,
    )
    phen = pd.DataFrame.from_records(dataset.phenotypic)
    records, matrices = [], []
    for idx, path in enumerate(dataset.rois_aal):
        if isinstance(path, (str, Path)):
            ts = np.loadtxt(path)
        else:
            ts = np.asarray(path, dtype=float)
        if ts.ndim != 2:
            print(f"Skipping {path}: non-2D time series")
            continue
        
        roi_std = np.std(ts, axis=0)
        constant_rois = np.where(roi_std < 1e-12)[0]

        if len(constant_rois) > 0:
            print(
                f"Subject {idx}: "
                f"{len(constant_rois)} constant ROI(s), "
                f"indices = {constant_rois.tolist()}"
            )
        try:
            corr = correlation_matrix(ts)
        except ValueError as exc:
            print(f"Skipping {path}: {exc}")
            continue
        edge_index, weights, signed = sparse_graph_from_corr(corr, density=0.10)
        row = phen.iloc[idx]
        label = int(row["DX_GROUP"])
        if label not in (1, 2):
            continue
        subject_id = row["FILE_ID"]
        if isinstance(subject_id, bytes):
            subject_id = subject_id.decode("utf-8")
        subject_id = str(subject_id)
        np.savez_compressed(out / f"{subject_id}.npz", corr=corr,
                            edge_index=edge_index, edge_weight=weights,
                            edge_signed=signed, label=label)
        matrices.append(corr)
        records.append({"file_id": subject_id, "label": label,
                        "site": str(row["SITE_ID"]), "n_rois": corr.shape[0],
                        "n_timepoints": ts.shape[0],
                        "n_undirected_edges": len(weights) // 2})
    if not records:
        raise RuntimeError("No subjects processed: inspect input data and metadata")
    summary = pd.DataFrame(records)
    if summary.n_rois.nunique() != 1:
        raise RuntimeError("ROI counts differ across subjects; inspect dataset")
    summary.to_csv(out / "subjects.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrices[0], cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set(title="ABIDE subject: AAL connectivity", xlabel="ROI index", ylabel="ROI index")
    fig.tight_layout()
    fig.savefig(results / "example_connectivity.png", dpi=160)
    plt.close(fig)
    print(f"Processed {len(summary)} subjects")
    print("Diagnosis counts (1=ASD; 2=control):", summary.label.value_counts().to_dict())
    print("Distinct sites:", summary.site.nunique())
    print("ROI count:", summary.n_rois.iloc[0])
    print("Saved results/example_connectivity.png")


if __name__ == "__main__":
    main()
