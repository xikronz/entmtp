"""Per-task regime divergence figure for the draft-tree-optimization section.

Two panels:
  (a) Pearson correlation between curated features and next-step accept length,
      one row per dataset (heatmap).
  (b) MLP AUROC for predicting `accept_length == 1` (a useful "short-step risk"
      target) per dataset.

Numbers come from the per-task analyses in
``entmtp/outputs/regime_predictability.ipynb`` (final cell outputs that the
manuscript draws on). They are hard-coded here so the figure is reproducible
without re-running the notebook.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["sharegpt", "litbench", "gsm", "humaneval", "arc"]

FEATURES = [
    "prev_accept_ema",
    "prev_5_accept_avg",
    "prev_3_accept_avg",
    "prev_5_accept_min",
    "block_start_entropy",
    "entropy_smoothed_w4",
    "entropy_smoothed_w16",
]

# Pearson r between each feature and `next_accept_length` per dataset.
# Cells without a top-7 entry from the notebook are left as np.nan and
# rendered as light gray ("not in top correlated").
PEARSON: dict[str, dict[str, float]] = {
    "sharegpt": {
        "prev_accept_ema": 0.485,
        "prev_5_accept_avg": 0.453,
        "prev_3_accept_avg": 0.425,
        "prev_5_accept_min": 0.376,
    },
    "litbench": {
        "prev_accept_ema": 0.267,
        "prev_5_accept_avg": 0.239,
        "prev_3_accept_avg": 0.213,
        "prev_5_accept_min": 0.182,
        "block_start_entropy": -0.180,
        "entropy_smoothed_w4": -0.193,
    },
    "gsm": {
        "prev_accept_ema": 0.222,
        "prev_5_accept_avg": 0.192,
        "prev_3_accept_avg": 0.169,
        "block_start_entropy": -0.175,
        "entropy_smoothed_w4": -0.204,
        "entropy_smoothed_w16": -0.197,
    },
    "humaneval": {
        "prev_accept_ema": 0.203,
        "prev_5_accept_avg": 0.191,
        "prev_3_accept_avg": 0.171,
        "prev_5_accept_min": 0.163,
        "block_start_entropy": -0.158,
    },
    "arc": {
        "prev_accept_ema": 0.192,
        "prev_5_accept_avg": 0.171,
        "prev_3_accept_avg": 0.151,
        "block_start_entropy": -0.165,
        "entropy_smoothed_w4": -0.159,
        "entropy_smoothed_w16": -0.130,
    },
}

# AUROC of an MLP predicting `accept_length == 1` (short-step risk).
AUROC: dict[str, float] = {
    "sharegpt": 0.704,
    "gsm": 0.629,
    "humaneval": 0.622,
    "litbench": 0.587,
    "arc": 0.547,
}


def build_heatmap_matrix() -> np.ndarray:
    M = np.full((len(DATASETS), len(FEATURES)), np.nan, dtype=float)
    for i, ds in enumerate(DATASETS):
        for j, feat in enumerate(FEATURES):
            v = PEARSON.get(ds, {}).get(feat)
            if v is not None:
                M[i, j] = v
    return M


def main(out_path: Path) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 3.6),
        gridspec_kw={"width_ratios": [2.4, 1.0]},
    )

    M = build_heatmap_matrix()
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#eeeeee")

    ax = axes[0]
    im = ax.imshow(
        np.ma.masked_invalid(M),
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(
        [f.replace("_", "\n") for f in FEATURES],
        fontsize=8,
        rotation=0,
    )
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels([d.replace("_", " ") for d in DATASETS], fontsize=9)

    for i in range(len(DATASETS)):
        for j in range(len(FEATURES)):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j,
                i,
                f"{v:+.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="black" if abs(v) < 0.6 * vmax else "white",
            )
    ax.set_title(
        "(a) Pearson(feature, next-step accept length) per task",
        fontsize=10,
    )
    ax.tick_params(axis="x", which="both", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Pearson r", fontsize=8)

    ax = axes[1]
    order = sorted(AUROC, key=AUROC.get, reverse=True)
    vals = [AUROC[d] for d in order]
    bars = ax.barh(
        order,
        vals,
        color="#3a7ca5",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.invert_yaxis()
    ax.axvline(0.5, color="0.4", linestyle="--", linewidth=0.8, label="chance")
    for bar, v in zip(bars, vals):
        ax.text(
            v + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}",
            va="center",
            fontsize=8.5,
        )
    ax.set_xlim(0.45, max(vals) + 0.06)
    ax.set_xlabel("AUROC", fontsize=9)
    ax.set_title("(b) Per-task short-step predictability", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(loc="lower right", fontsize=7, frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[ok] saved {out_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "outputs" / "regime_per_task.png"
    main(out)
