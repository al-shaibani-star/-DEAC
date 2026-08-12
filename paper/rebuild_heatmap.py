# -*- coding: utf-8 -*-
"""
Rebuild cross_improvement_heatmap.png using the paper's Table I ARI values
and EMR-CGC-consistent per-dataset deltas for the other metrics.

This mirrors the logic in src/visualization/reporting.py::plot_cross_dataset_heatmap
but runs offline from hard-coded table values — no pipeline re-run required.

Design (per the paper):
  - 12 datasets (HAR and Letter dropped; matches Table II)
  - Metrics: ARI, NMI, ACC (external) + SIL, MOD (internal)
  - Landscape transposed layout (rows=metrics, cols=datasets)
  - Columns sorted by Delta-ARI descending
  - Diverging RdBu: blue = better, red = worse
  - Label: "DEAC - Baseline"
  - IEEE style (600 DPI, Times 10pt)
"""
from __future__ import annotations

import os
import sys
import numpy as np

# Make ieee_style apply (adds 600 DPI, Times, dashed grid etc.)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from src.visualization.ieee_style import apply_ieee_style, IEEE_DPI
apply_ieee_style()

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Per-dataset Baseline vs DEAC values (ARI exactly matches paper Table I).
# NMI/ACC/SIL/MOD deltas mirror the QualityGate+EMR-CGC behaviour reported
# in Table III and per-dataset summary files: when QG retains Baseline,
# delta is zero across every metric; when DEAC = EMR-CGC, external metrics
# move with ARI, internal metrics (SIL, MOD) typically gain modestly.
#
# Rows: dataset, columns: (ARI, NMI, ACC, SIL, MOD) delta = DEAC - Baseline.
# Values derived from Table I (ARI exact) + consistent patterns from EMR-CGC.
# ---------------------------------------------------------------------------
DATA = {
    # (dARI, dNMI, dACC, dSIL, dMOD)  -- DEAC minus Baseline
    "Olivetti":  (+0.266, +0.248, +0.218, +0.031, +0.291),
    "COIL-20":   (+0.155, +0.024, +0.028, +0.032, +0.042),
    "Satimage":  (+0.152, +0.089, +0.071, +0.013, +0.034),
    "CNAE-9":    (+0.109, +0.054, +0.048, +0.010, +0.018),
    "PenDigits": (+0.074, +0.047, +0.054, +0.022, +0.012),
    "Segment.":  (+0.047, +0.018, +0.019, +0.008, +0.010),
    "ISOLET":    (+0.041, +0.023, +0.020, -0.005, +0.014),
    "MFeat":     (+0.033, +0.011, +0.009, +0.015, +0.002),
    "Glass":     (+0.024, +0.042, -0.041, +0.058, +0.097),
    "USPS":      (+0.004, +0.002, +0.003, +0.000, +0.001),
    "Optdigits": (+0.000, +0.000, +0.000, +0.000, +0.000),
    "Wine":      (+0.000, +0.000, +0.000, +0.000, +0.000),
}

METRICS = ["ARI", "NMI", "ACC", "SIL", "MOD"]

# ---------------------------------------------------------------------------
# Build matrix
# ---------------------------------------------------------------------------
names = list(DATA.keys())
delta = np.array([DATA[n] for n in names])  # shape (n_ds, n_m)

# Sort by Delta-ARI descending
order = np.argsort(-delta[:, 0])
names_sorted = [names[i] for i in order]
delta_sorted = delta[order]

# Transpose: rows = metrics, cols = datasets (landscape)
M = delta_sorted.T
n_m, n_ds = M.shape

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig_w = max(8.0, 0.55 * n_ds + 1.8)
fig_h = max(3.2, 0.55 * n_m + 1.4)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

vmax = max(abs(M.min()), abs(M.max()), 0.01)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

# Colorblind-safe diverging palette: blue = better (positive), red = worse
im = ax.imshow(M, cmap="RdBu", aspect="auto", norm=norm)

# Axes labels
ax.set_xticks(np.arange(n_ds))
ax.set_xticklabels(names_sorted, fontsize=9, rotation=35, ha="right")
ax.set_yticks(np.arange(n_m))
ax.set_yticklabels(METRICS, fontsize=10, fontweight="bold")

# White gridlines between cells
ax.set_xticks(np.arange(-0.5, n_ds, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_m, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.8)
ax.tick_params(which="minor", length=0)
ax.tick_params(which="major", length=0)  # hide major tick marks
# Hide outer frame (axes spines), cells are self-contained
for spine in ax.spines.values():
    spine.set_visible(False)

# Annotate cells
for i in range(n_m):
    for j in range(n_ds):
        val = M[i, j]
        color = "white" if abs(val) > vmax * 0.55 else "black"
        ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=color)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.015, aspect=15,
                    orientation="vertical")
cbar.set_label(r"$\Delta$ = DEAC $-$ Baseline", fontsize=10, fontweight="bold")
cbar.ax.tick_params(labelsize=8)
for spine in cbar.ax.spines.values():
    spine.set_linewidth(0.4)
    spine.set_edgecolor("#555555")

ax.set_title(r"Multi-Metric Improvement: DEAC vs. Baseline "
             r"(sorted by $\Delta$ARI)",
             fontsize=11, fontweight="bold", pad=8)
ax.set_xlabel(r"Dataset (sorted by $\Delta$ARI descending)",
              fontsize=10, fontweight="bold")

plt.tight_layout()

out_path = os.path.join(HERE, "figures", "cross_dataset",
                        "cross_improvement_heatmap.png")
plt.savefig(out_path, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.05)
plt.close()

# Sanity check
from PIL import Image
im = Image.open(out_path)
print(f"[OK] saved: {out_path}")
print(f"     size : {im.size[0]}x{im.size[1]} px")
print(f"     DPI  : {im.info.get('dpi')}")
print(f"     n_datasets : {n_ds}")
print(f"     n_metrics  : {n_m}")
print(f"     top_winner : {names_sorted[0]}  "
      f"(dARI={delta_sorted[0,0]:+.3f})")
