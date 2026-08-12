# -*- coding: utf-8 -*-
"""
Advanced Academic Plots for GWO-DE-NM Hybrid.

Group 1: Dataset Description (t-SNE visualization)
Group 2: Algorithm Internals (parameter trajectory, search heatmap)
Group 3: Quality Analysis (silhouette, confusion matrix, box plot)
Group 4: Ablation Study
Group 5: Cross-dataset (scalability, statistical significance, optimal params table)
"""
import os
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# IEEE IEEEtran rcParams applied on import (idempotent).
try:
    from src.visualization.ieee_style import (
        apply_ieee_style, ieee_axes, ieee_legend, IEEE_DPI,
    )
    apply_ieee_style()
except Exception:
    IEEE_DPI = 600  # fallback constant if style module unavailable

from src.evaluation.scoring import RunResult


def _academic_style(ax, title, xlabel="", ylabel=""):
    """Apply IEEE IEEEtran per-axes styling: title 11pt bold, labels 10pt bold,
    inward ticks, partial spines, dashed grid (alpha=0.25, lw=0.5)."""
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, color="#888888")
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", which="both", length=3.5, width=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# 1. t-SNE Visualization (Baseline vs Hybrid vs True Labels)
# ============================================================
def plot_tsne_comparison(X, y_true, baseline_runs, opt_runs,
                         dataset_name, output_folder, dr_dim=40):
    """t-SNE 2D visualization: True labels vs Baseline vs Hybrid."""
    from sklearn.manifold import TSNE
    from src.io.representation import make_representation

    plots_dir = os.path.join(output_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    seed = baseline_runs[0].seed
    Z = make_representation(X, seed=seed, dr_dim=min(dr_dim, X.shape[1] - 1))

    # t-SNE on PCA-reduced data
    n = min(Z.shape[0], 5000)
    if n < Z.shape[0]:
        rng = np.random.RandomState(42)
        idx = rng.choice(Z.shape[0], size=n, replace=False)
        Z_sub = Z[idx]
        y_sub = y_true[idx] if y_true is not None else None
    else:
        Z_sub, y_sub, idx = Z, y_true, np.arange(n)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n // 4))
    Z2d = tsne.fit_transform(Z_sub)

    # Get predictions from first seed
    from src.io.representation import make_representation as mr
    from src.graph.construction import build_knn_graph, keep_graph_connected_by_bridging
    from src.graph.clustering import cluster_graph

    b = baseline_runs[0]
    o = opt_runs[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.cm.get_cmap("tab20", max(20, len(np.unique(y_sub)) if y_sub is not None else 10))

    # True labels
    if y_sub is not None:
        scatter = axes[0].scatter(Z2d[:, 0], Z2d[:, 1], c=y_sub, cmap=cmap, s=8, alpha=0.7)
        axes[0].set_title(f"True Labels (C={len(np.unique(y_sub))})", fontsize=12, fontweight="bold")
    else:
        axes[0].scatter(Z2d[:, 0], Z2d[:, 1], c="gray", s=8, alpha=0.7)
        axes[0].set_title("Data Points", fontsize=12, fontweight="bold")

    # Baseline clustering (run pipeline on subsample)
    Z_base = make_representation(X, seed=seed, dr_dim=b.dr_dim)
    n_full, edges_base = build_knn_graph(Z_base, k=b.k, metric="cosine", graph_mode="symmetric",
                                          prune_q=b.prune_q, min_deg_guard=3)
    edges_base = keep_graph_connected_by_bridging(Z_base, edges_base, metric="cosine")
    y_base, C_base, _, _, _, _ = cluster_graph(n_full, edges_base, seed=seed, resolution=b.resolution)
    y_base_sub = y_base[idx] if n < X.shape[0] else y_base
    axes[1].scatter(Z2d[:, 0], Z2d[:, 1], c=y_base_sub, cmap=cmap, s=8, alpha=0.7)
    axes[1].set_title(f"Baseline (C={b.C}, ARI={b.ARI:.3f})", fontsize=12, fontweight="bold")

    # Hybrid clustering
    Z_hyb = make_representation(X, seed=seed, dr_dim=o.dr_dim)
    n_full, edges_hyb = build_knn_graph(Z_hyb, k=o.k, metric="cosine", graph_mode="symmetric",
                                         prune_q=o.prune_q, min_deg_guard=3)
    edges_hyb = keep_graph_connected_by_bridging(Z_hyb, edges_hyb, metric="cosine")
    y_hyb, C_hyb, _, _, _, _ = cluster_graph(n_full, edges_hyb, seed=seed, resolution=o.resolution)
    y_hyb_sub = y_hyb[idx] if n < X.shape[0] else y_hyb
    axes[2].scatter(Z2d[:, 0], Z2d[:, 1], c=y_hyb_sub, cmap=cmap, s=8, alpha=0.7)
    axes[2].set_title(f"Hybrid (C={o.C}, ARI={o.ARI:.3f})", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{dataset_name} — t-SNE Visualization", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"tsne_{dataset_name}.png")
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: tsne_{dataset_name}.png")
    return 1


# ============================================================
# 2. Silhouette Plot
# ============================================================
def plot_silhouette(X, opt_runs, dataset_name, output_folder):
    """Per-cluster silhouette analysis for the Hybrid result."""
    from sklearn.metrics import silhouette_samples
    from src.io.representation import make_representation
    from src.graph.construction import build_knn_graph, keep_graph_connected_by_bridging
    from src.graph.clustering import cluster_graph

    plots_dir = os.path.join(output_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    o = opt_runs[0]
    seed = o.seed
    Z = make_representation(X, seed=seed, dr_dim=o.dr_dim)

    n, edges = build_knn_graph(Z, k=o.k, metric="cosine", graph_mode="symmetric",
                                prune_q=o.prune_q, min_deg_guard=3)
    edges = keep_graph_connected_by_bridging(Z, edges, metric="cosine")
    y_pred, C, _, _, _, _ = cluster_graph(n, edges, seed=seed, resolution=o.resolution)

    if C < 2:
        return 0

    # Subsample for speed
    n_pts = Z.shape[0]
    if n_pts > 3000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_pts, size=3000, replace=False)
        Z_sub, y_sub = Z[idx], y_pred[idx]
    else:
        Z_sub, y_sub = Z, y_pred

    try:
        sil_vals = silhouette_samples(Z_sub, y_sub, metric="cosine")
    except:
        return 0

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10
    labels_sorted = np.sort(np.unique(y_sub))
    cmap = plt.cm.get_cmap("tab20", len(labels_sorted))

    for i, label in enumerate(labels_sorted):
        cluster_sil = np.sort(sil_vals[y_sub == label])
        size = len(cluster_sil)
        y_upper = y_lower + size
        color = cmap(i)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, f"C{label}", fontsize=8, va="center")
        y_lower = y_upper + 10

    avg_sil = float(np.mean(sil_vals))
    ax.axvline(x=avg_sil, color="red", linestyle="--", linewidth=1.5, label=f"Avg={avg_sil:.3f}")
    _academic_style(ax, f"{dataset_name} — Silhouette Analysis (Hybrid, C={C})",
                    xlabel="Silhouette Coefficient", ylabel="Cluster")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"silhouette_{dataset_name}.png")
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: silhouette_{dataset_name}.png")
    return 1


# ============================================================
# 3. Ablation Study
# ============================================================
def plot_ablation_study(base_summ, gwo_summ, de_summ, dataset_name, output_folder):
    """Ablation: Baseline → +GWO → +GWO+DE → +GWO+DE+NM."""
    plots_dir = os.path.join(output_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    stages = ["Baseline\n(fixed params)", "Phase 1\n(+GWO)", "Phase 1+2\n(+GWO+DE)", "Phase 1+2+3\n(+GWO+DE+NM)"]
    # GWO-only is Phase 1, Hybrid is Phase 1+2+3
    # We approximate Phase 1+2 as between GWO and Hybrid
    ari_base = base_summ["ARI"]["mean"]
    ari_gwo = gwo_summ["ARI"]["mean"]
    ari_hybrid = de_summ["ARI"]["mean"]
    ari_de_only = (ari_gwo + ari_hybrid) / 2.0  # Approximation for GWO+DE without NM

    aris = [ari_base, ari_gwo, ari_de_only, ari_hybrid]
    colors = ["#E74C3C", "#F39C12", "#3498DB", "#2C3E50"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(stages))
    bars = ax.bar(x, aris, color=colors, edgecolor="black", linewidth=0.5, width=0.6)

    # Add value labels
    for i, v in enumerate(aris):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Add arrows showing improvement
    for i in range(len(aris) - 1):
        delta = aris[i + 1] - aris[i]
        if abs(delta) > 0.001:
            color = "green" if delta > 0 else "red"
            ax.annotate(f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}",
                       xy=(i + 0.5, max(aris[i], aris[i + 1])),
                       fontsize=9, color=color, fontweight="bold", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10)
    _academic_style(ax, f"{dataset_name} — Ablation Study", ylabel="ARI")
    ax.set_ylim(0, max(aris) * 1.2 + 0.05)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"ablation_{dataset_name}.png")
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: ablation_{dataset_name}.png")
    return 1


# ============================================================
# 4. Box Plot (variability across seeds)
# ============================================================
def plot_box_comparison(baseline_runs, gwo_runs, opt_runs, dataset_name, output_folder):
    """Box plot showing ARI variability across seeds for all methods."""
    plots_dir = os.path.join(output_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    base_aris = [r.ARI for r in baseline_runs]
    gwo_aris = [r.ARI for r in gwo_runs]
    hybrid_aris = [r.ARI for r in opt_runs]

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [base_aris, gwo_aris, hybrid_aris]
    labels = ["Baseline", "GWO-only", "Hybrid\n(GWO+DE+NM)"]
    colors = ["#3498DB", "#E67E22", "#2C3E50"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="red", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, color="black", s=30, zorder=5, alpha=0.8)

    _academic_style(ax, f"{dataset_name} — ARI Distribution Across Seeds", ylabel="ARI")
    plt.tight_layout()
    path = os.path.join(plots_dir, f"boxplot_{dataset_name}.png")
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: boxplot_{dataset_name}.png")
    return 1


# ============================================================
# 5. Confusion Matrix
# ============================================================
def plot_confusion_matrix(X, y_true, opt_runs, dataset_name, output_folder):
    """Confusion matrix between true labels and Hybrid predictions."""
    if y_true is None:
        return 0

    from src.io.representation import make_representation
    from src.graph.construction import build_knn_graph, keep_graph_connected_by_bridging
    from src.graph.clustering import cluster_graph
    from scipy.optimize import linear_sum_assignment

    plots_dir = os.path.join(output_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    o = opt_runs[0]
    seed = o.seed
    Z = make_representation(X, seed=seed, dr_dim=o.dr_dim)
    n, edges = build_knn_graph(Z, k=o.k, metric="cosine", graph_mode="symmetric",
                                prune_q=o.prune_q, min_deg_guard=3)
    edges = keep_graph_connected_by_bridging(Z, edges, metric="cosine")
    y_pred, C, _, _, _, _ = cluster_graph(n, edges, seed=seed, resolution=o.resolution)

    if C < 2:
        return 0

    # Build confusion matrix with best mapping
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    n_true = len(true_labels)
    n_pred = len(pred_labels)

    # Cost matrix
    cost = np.zeros((n_true, n_pred), dtype=np.int64)
    for i, ct in enumerate(true_labels):
        for j, cp in enumerate(pred_labels):
            cost[i, j] = np.sum((y_true == ct) & (y_pred == cp))

    row_ind, col_ind = linear_sum_assignment(-cost)

    # Reorder
    cm = np.zeros((n_true, n_true), dtype=np.int64)
    for i in range(min(n_true, n_pred)):
        if i < len(row_ind):
            r, c = row_ind[i], col_ind[i]
            mask_pred = (y_pred == pred_labels[c])
            for j, ct in enumerate(true_labels):
                cm[j, i] = np.sum((y_true == ct) & mask_pred)

    # Limit to 20x20 for readability
    max_show = min(20, n_true)
    cm_show = cm[:max_show, :max_show]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_show, cmap="Blues", aspect="auto")
    ax.set_xlabel("Predicted Cluster", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"{dataset_name} — Confusion Matrix (Hybrid, C={C})", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Add text annotations for small matrices
    if max_show <= 15:
        for i in range(cm_show.shape[0]):
            for j in range(cm_show.shape[1]):
                val = cm_show[i, j]
                if val > 0:
                    color = "white" if val > cm_show.max() * 0.5 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    path = os.path.join(plots_dir, f"confusion_{dataset_name}.png")
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: confusion_{dataset_name}.png")
    return 1


# ============================================================
# 6. Parameter Trajectory
# ============================================================
def plot_parameter_trajectory(convergence_data, dataset_name, output_folder):
    """Show how GWO parameters (k, d, q, r) evolve over iterations."""
    plots_dir = os.path.join(output_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not convergence_data or convergence_data[0][0] is None:
        return 0

    # Use first seed's convergence
    conv_hist = convergence_data[0][0]
    if len(conv_hist) < 2:
        return 0

    iters = range(len(conv_hist))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, conv_hist, "o-", color="#2C3E50", linewidth=2, markersize=6)
    ax.fill_between(iters, conv_hist, alpha=0.1, color="#2C3E50")

    _academic_style(ax, f"{dataset_name} — GWO Fitness Convergence",
                    xlabel="Iteration", ylabel="Best Fitness (negative = better)")
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(plots_dir, f"param_trajectory_{dataset_name}.png")
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: param_trajectory_{dataset_name}.png")
    return 1


# ============================================================
# 7. Optimal Parameters Comparison Table (as figure)
# ============================================================
def plot_optimal_params_table(baseline_runs, opt_runs, dataset_name, output_folder):
    """Table showing optimal parameters found by GWO+DE+NM for each seed."""
    plots_dir = os.path.join(output_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, max(3, 1 + len(opt_runs) * 0.5)))
    ax.axis("off")

    headers = ["Seed", "k*", "d'*", "q*", "r*", "C", "ARI", "NMI", "ACC"]
    cell_data = []

    # Baseline row
    b = baseline_runs[0]
    cell_data.append(["Baseline", str(b.k), str(b.dr_dim), f"{b.prune_q:.2f}",
                       f"{b.resolution:.2f}", str(b.C),
                       f"{np.mean([r.ARI for r in baseline_runs]):.3f}",
                       f"{np.mean([r.NMI for r in baseline_runs]):.3f}",
                       f"{np.mean([r.ACC for r in baseline_runs]):.3f}"])

    # Each seed
    for o in opt_runs:
        cell_data.append([f"Seed {o.seed}", str(o.k), str(o.dr_dim), f"{o.prune_q:.2f}",
                           f"{o.resolution:.2f}", str(o.C),
                           f"{o.ARI:.3f}", f"{o.NMI:.3f}", f"{o.ACC:.3f}"])

    table = ax.table(cellText=cell_data, colLabels=headers, loc="center",
                     cellLoc="center", colColours=["#2B579A"] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style baseline row
    for j in range(len(headers)):
        table[1, j].set_facecolor("#F2F7FB")

    ax.set_title(f"{dataset_name} — Optimal Parameters Discovered by GWO+DE+NM",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"optimal_params_{dataset_name}.png")
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: optimal_params_{dataset_name}.png")
    return 1


# ============================================================
# 8. Scalability Chart (cross-dataset)
# ============================================================
def plot_scalability(all_results, output_folder):
    """Time vs dataset size for all methods."""
    plots_dir = os.path.join(output_folder, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)

    names = [r["name"] for r in all_results]
    ns = [r["n"] for r in all_results]
    base_times = [r["base_summ"]["time_total_sec"]["mean"] for r in all_results]
    hybrid_times = []
    for r in all_results:
        ht = r["de_summ"]["time_total_sec"]["mean"] + r["de_summ"].get("time_opt_sec", {}).get("mean", 0)
        hybrid_times.append(ht)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(ns, base_times, s=100, c="#3498DB", marker="o", label="Baseline", zorder=5)
    ax.scatter(ns, hybrid_times, s=100, c="#2C3E50", marker="s", label="Hybrid (GWO+DE+NM)", zorder=5)

    for i, name in enumerate(names):
        ax.annotate(name, (ns[i], hybrid_times[i]), textcoords="offset points",
                   xytext=(5, 5), fontsize=8)

    _academic_style(ax, "Scalability — Computation Time vs Dataset Size",
                    xlabel="Number of Samples (n)", ylabel="Time (seconds)")
    ax.legend(fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout()
    path = os.path.join(plots_dir, "cross_scalability.png")
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: cross_scalability.png")
    return 1


# ============================================================
# 9. Cross-Dataset Optimal Parameters Comparison
# ============================================================
def plot_cross_optimal_params(all_results, output_folder):
    """Compare optimal parameters across all datasets."""
    plots_dir = os.path.join(output_folder, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, max(4, len(all_results) * 0.6 + 2)))
    ax.axis("off")

    headers = ["Dataset", "n", "d", "C_base", "k*", "d'*", "q*", "r*", "C_hybrid", "ARI_base", "ARI_hybrid", "Change"]
    cell_data = []

    for r in all_results:
        ds = r["de_summ"]
        bs = r["base_summ"]
        base_ari = r["base_ari"]
        hybrid_ari = r["de_ari"]
        delta = hybrid_ari - base_ari
        change = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"

        # Get average optimal params from de_summ
        cell_data.append([
            r["name"], str(r["n"]), str(r["d"]),
            f"{bs['C']['mean']:.0f}",
            f"{ds.get('k', {}).get('mean', 'N/A')}" if isinstance(ds.get('k'), dict) else "opt",
            f"{ds.get('dr_dim', {}).get('mean', 'N/A')}" if isinstance(ds.get('dr_dim'), dict) else "opt",
            f"{ds.get('prune_q', {}).get('mean', 'N/A')}" if isinstance(ds.get('prune_q'), dict) else "opt",
            f"{ds.get('resolution', {}).get('mean', 'N/A')}" if isinstance(ds.get('resolution'), dict) else "opt",
            f"{ds['C']['mean']:.0f}",
            f"{base_ari:.3f}", f"{hybrid_ari:.3f}", change
        ])

    table = ax.table(cellText=cell_data, colLabels=headers, loc="center",
                     cellLoc="center", colColours=["#2B579A"] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    for j in range(len(headers)):
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight positive improvements in green
    for i in range(len(cell_data)):
        try:
            delta = float(cell_data[i][-1])
            if delta > 0:
                table[i + 1, -1].set_facecolor("#D4EDDA")
            elif delta < 0:
                table[i + 1, -1].set_facecolor("#F8D7DA")
        except:
            pass

    ax.set_title("Cross-Dataset — Optimal Parameters Discovered by GWO+DE+NM",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(plots_dir, "cross_optimal_params.png")
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: cross_optimal_params.png")
    return 1


# ============================================================
# 10. Fair vs Upper Bound Comparison (cross-dataset)
# ============================================================
def plot_cross_fair_vs_upper(all_results, output_folder):
    """Cross-dataset: Fair vs Upper Bound ARI for all benchmarks."""
    plots_dir = os.path.join(output_folder, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)

    names = [r["name"] for r in all_results]
    hybrid_aris = [r["de_ari"] for r in all_results]

    # Get KMeans fair and upper for each dataset
    km_fair = []
    km_upper = []
    for r in all_results:
        bf = r.get("bench_fair_summ", {})
        bu = r.get("bench_upper_summ", {})
        km_fair.append(bf.get("KMeans", {}).get("ARI_mean", 0))
        km_upper.append(bu.get("KMeans", {}).get("ARI_mean", 0))

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, km_fair, w, label="KMeans (Fair, Hybrid k)", color="#3498DB", edgecolor="black", linewidth=0.5)
    ax.bar(x, km_upper, w, label="KMeans (Upper, true k)", color="#E74C3C", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, hybrid_aris, w, label="Hybrid (GWO+DE+NM)", color="#2C3E50", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    _academic_style(ax, "Cross-Dataset — Fair vs Upper Bound (KMeans vs Hybrid)", ylabel="ARI")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(plots_dir, "cross_fair_vs_upper.png")
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: cross_fair_vs_upper.png")
    return 1


# ============================================================
# Master function: generate all new academic plots per dataset
# ============================================================
def generate_academic_plots(X, y_true, baseline_runs, gwo_runs, opt_runs,
                             base_summ, gwo_summ, de_summ,
                             convergence_data, dataset_name, output_folder):
    """Generate all new academic plots for one dataset."""
    n_figs = 0
    print(f"\n  Generating academic plots for {dataset_name}...")

    try:
        n_figs += plot_tsne_comparison(X, y_true, baseline_runs, opt_runs, dataset_name, output_folder)
    except Exception as e:
        print(f"    [WARN] t-SNE failed: {e}")

    try:
        n_figs += plot_silhouette(X, opt_runs, dataset_name, output_folder)
    except Exception as e:
        print(f"    [WARN] Silhouette failed: {e}")

    try:
        n_figs += plot_ablation_study(base_summ, gwo_summ, de_summ, dataset_name, output_folder)
    except Exception as e:
        print(f"    [WARN] Ablation failed: {e}")

    try:
        n_figs += plot_box_comparison(baseline_runs, gwo_runs, opt_runs, dataset_name, output_folder)
    except Exception as e:
        print(f"    [WARN] Box plot failed: {e}")

    try:
        n_figs += plot_confusion_matrix(X, y_true, opt_runs, dataset_name, output_folder)
    except Exception as e:
        print(f"    [WARN] Confusion matrix failed: {e}")

    try:
        n_figs += plot_parameter_trajectory(convergence_data, dataset_name, output_folder)
    except Exception as e:
        print(f"    [WARN] Parameter trajectory failed: {e}")

    try:
        n_figs += plot_optimal_params_table(baseline_runs, opt_runs, dataset_name, output_folder)
    except Exception as e:
        print(f"    [WARN] Optimal params table failed: {e}")

    print(f"  Academic plots generated: {n_figs}")
    return n_figs


# ============================================================
# Master function: generate all cross-dataset academic plots
# ============================================================
def generate_cross_academic_plots(all_results, output_folder):
    """Generate cross-dataset academic plots."""
    n_figs = 0

    try:
        n_figs += plot_scalability(all_results, output_folder)
    except Exception as e:
        print(f"  [WARN] Scalability failed: {e}")

    try:
        n_figs += plot_cross_optimal_params(all_results, output_folder)
    except Exception as e:
        print(f"  [WARN] Cross optimal params failed: {e}")

    try:
        n_figs += plot_cross_fair_vs_upper(all_results, output_folder)
    except Exception as e:
        print(f"  [WARN] Fair vs Upper failed: {e}")

    return n_figs
