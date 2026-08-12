# -*- coding: utf-8 -*-
"""Reporting: print results, summary tables, comparison plots, academic output,
   figure generation, and computational complexity analysis."""
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# IEEE IEEEtran rcParams applied on import (idempotent).
try:
    from src.visualization.ieee_style import (
        apply_ieee_style, ieee_axes, ieee_legend, IEEE_DPI,
    )
    apply_ieee_style()
except Exception:
    IEEE_DPI = 600  # fallback constant if style module unavailable

from src.io.utils import safe_mean_std
from src.evaluation.scoring import RunResult


def print_seed_line(b: RunResult, o: Optional[RunResult]) -> None:
    base_info = (
        f"Seed={b.seed} | BASE "
        f"ARI={b.ARI:>7.3f} NMI={b.NMI:>7.3f} ACC={b.ACC:>7.3f} "
        f"SIL={b.SIL:>7.3f} MOD={b.MOD:>7.3f} "
        f"bal={b.bal01:>5.3f} s1={b.n_singletons:>4d} t2={b.n_tiny:>4d} "
        f"C={b.C:>4d} CC={b.n_components:>3d} "
        f"pen={b.internal_pen:>6.3f} int={b.internal_score:>6.3f} "
        f"T={b.time_total_sec:>6.2f}s "
        f"{'DEGEN' if b.degenerate else ''}"
    )
    if o is None:
        print(base_info)
        return
    print(
        f"{base_info}\n"
        f"          | DE   "
        f"ARI={o.ARI:>7.3f} NMI={o.NMI:>7.3f} ACC={o.ACC:>7.3f} "
        f"SIL={o.SIL:>7.3f} MOD={o.MOD:>7.3f} "
        f"bal={o.bal01:>5.3f} s1={o.n_singletons:>4d} t2={o.n_tiny:>4d} "
        f"C={o.C:>4d} CC={o.n_components:>3d} "
        f"pen={o.internal_pen:>6.3f} int={o.internal_score:>6.3f} "
        f"T={o.time_total_sec:>6.2f}s +opt={o.time_opt_sec:>6.2f}s "
        f"params=[k={o.k}, d={o.dr_dim}, q={o.prune_q:.2f}, r={o.resolution:.2f}] "
        f"{'DEGEN' if o.degenerate else ''}"
    )


def summarize_results(results: List[RunResult]) -> Dict[str, Dict[str, float]]:
    keys = [
        "ARI", "NMI", "ACC",
        "SIL", "MOD", "bal01",
        "C", "n_components",
        "internal_score", "internal_pen",
        "time_total_sec", "time_opt_sec",
    ]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [float(getattr(r, k)) for r in results]
        m, s = safe_mean_std(vals)
        out[k] = {"mean": m, "std": s}
    return out


def print_summary_table(title: str, summ: Dict[str, Dict[str, float]]) -> None:
    print(title)
    order = [
        "ARI", "NMI", "ACC",
        "SIL", "MOD", "bal01",
        "C", "n_components",
        "internal_score", "internal_pen",
        "time_total_sec", "time_opt_sec",
    ]
    for k in order:
        m = summ[k]["mean"]
        s = summ[k]["std"]
        if k in ["C", "n_components"]:
            print(f"{k:14s} | {m:10.3f} +/- {s:10.3f}")
        else:
            print(f"{k:14s} | {m:10.4f} +/- {s:10.4f}")


# ---------------------------------------------------------------------------
# Academic Figures (inspired by Late Fusion style)
# ---------------------------------------------------------------------------
FIGURE_COUNTER = {"n": 0}


def _next_fig_num() -> int:
    FIGURE_COUNTER["n"] += 1
    return FIGURE_COUNTER["n"]


def reset_figure_counter():
    FIGURE_COUNTER["n"] = 0


def plot_compare(
    base_summ: Dict[str, Dict[str, float]],
    de_summ: Dict[str, Dict[str, float]],
    out_path: str,
    show: int,
) -> None:
    if not HAS_PLT:
        print("Plot skipped: matplotlib not installed.", file=sys.stderr)
        return
    import matplotlib.pyplot as plt
    metrics = ["ARI", "NMI", "ACC", "SIL", "MOD", "bal01", "C", "n_components", "internal_score", "time_total_sec"]
    base_means = [base_summ[m]["mean"] for m in metrics]
    base_stds = [base_summ[m]["std"] for m in metrics]
    de_means = [de_summ[m]["mean"] for m in metrics]
    de_stds = [de_summ[m]["std"] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.38
    plt.figure(figsize=(13, 5))
    plt.bar(x - width / 2, base_means, width, yerr=base_stds, capsize=3, label="Baseline")
    plt.bar(x + width / 2, de_means, width, yerr=de_stds, capsize=3, label="GWO-DE-NM Hybrid")
    plt.xticks(x, metrics, rotation=0)
    plt.title("Baseline vs GWO-DE-NM Hybrid (mean +/- std over seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved comparison plot: {out_path}")
    if show == 1:
        plt.show()
    plt.close()


def plot_feature_space_clusters(Z: np.ndarray, y_pred: np.ndarray, title: str,
                                filename: str, output_folder: str) -> None:
    """Figure: PCA 2D scatter colored by cluster labels."""
    if not HAS_PLT:
        return
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    if Z.shape[1] > 2:
        pca = PCA(n_components=2)
        Z2d = pca.fit_transform(Z)
    else:
        Z2d = Z

    fig_num = _next_fig_num()
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(Z2d[:, 0], Z2d[:, 1], c=y_pred, cmap="tab20", s=8, alpha=0.7)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title(f"Figure {fig_num}: {title}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, filename)
    plt.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved Figure {fig_num}: {path}")


def plot_cluster_distribution(baseline_runs: List[RunResult], opt_runs: List[RunResult],
                              dataset_name: str, output_folder: str) -> None:
    """Figure: Cluster count distribution (Baseline vs Hybrid)."""
    if not HAS_PLT:
        return
    import matplotlib.pyplot as plt

    fig_num = _next_fig_num()
    base_C = [r.C for r in baseline_runs]
    de_C = [r.C for r in opt_runs]
    seeds = list(range(1, len(base_C) + 1))

    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(seeds))
    plt.bar(x - width/2, base_C, width, label="Baseline", color="#4c72b0")
    plt.bar(x + width/2, de_C, width, label="GWO-DE-NM Hybrid", color="#dd8452")
    plt.xlabel("Seed")
    plt.ylabel("Number of Clusters (C)")
    plt.title(f"Figure {fig_num}: Cluster Count -- {dataset_name}")
    plt.xticks(x, [f"S{s}" for s in seeds])
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"cluster_distribution_{dataset_name}.png")
    plt.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved Figure {fig_num}: {path}")


def plot_metric_comparison_bars(base_summ: Dict, de_summ: Dict,
                                dataset_name: str, output_folder: str) -> None:
    """Figure: Horizontal bar comparison for external metrics (ARI, NMI, ACC)."""
    if not HAS_PLT:
        return
    import matplotlib.pyplot as plt

    fig_num = _next_fig_num()
    metrics = ["ARI", "NMI", "ACC"]
    base_vals = [base_summ[m]["mean"] for m in metrics]
    base_errs = [base_summ[m]["std"] for m in metrics]
    de_vals = [de_summ[m]["mean"] for m in metrics]
    de_errs = [de_summ[m]["std"] for m in metrics]

    y_pos = np.arange(len(metrics))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.barh(y_pos - height/2, base_vals, height, xerr=base_errs,
                    label="Baseline", color="#4c72b0", capsize=4)
    bars2 = ax.barh(y_pos + height/2, de_vals, height, xerr=de_errs,
                    label="GWO-DE-NM Hybrid", color="#dd8452", capsize=4)

    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", ha='left', va='center', fontsize=10, fontweight='bold')
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Score")
    ax.set_title(f"Figure {fig_num}: External Metrics Comparison -- {dataset_name}")
    ax.legend()
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"metrics_comparison_{dataset_name}.png")
    plt.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved Figure {fig_num}: {path}")


def plot_internal_metrics(base_summ: Dict, de_summ: Dict,
                          dataset_name: str, output_folder: str) -> None:
    """Figure: Internal metrics comparison (SIL, MOD, Balance)."""
    if not HAS_PLT:
        return
    import matplotlib.pyplot as plt

    fig_num = _next_fig_num()
    metrics = ["SIL", "MOD", "bal01"]
    labels = ["Silhouette", "Modularity", "Balance"]
    base_vals = [base_summ[m]["mean"] for m in metrics]
    de_vals = [de_summ[m]["mean"] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, base_vals, width, label="Baseline", color="#4c72b0")
    ax.bar(x + width/2, de_vals, width, label="GWO-DE-NM Hybrid", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title(f"Figure {fig_num}: Internal Metrics -- {dataset_name}")
    ax.legend()

    for i, (bv, dv) in enumerate(zip(base_vals, de_vals)):
        ax.text(i - width/2, bv + 0.01, f"{bv:.3f}", ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, dv + 0.01, f"{dv:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"internal_metrics_{dataset_name}.png")
    plt.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved Figure {fig_num}: {path}")


def plot_radar_chart(base_summ: Dict, de_summ: Dict,
                     dataset_name: str, output_folder: str) -> None:
    """Figure: Radar/spider chart for multi-metric comparison."""
    if not HAS_PLT:
        return
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig_num = _next_fig_num()
    metrics = ["ARI", "NMI", "ACC", "SIL", "MOD"]
    base_vals = [max(0, base_summ[m]["mean"]) for m in metrics]
    de_vals = [max(0, de_summ[m]["mean"]) for m in metrics]

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    base_vals_r = base_vals + [base_vals[0]]
    de_vals_r = de_vals + [de_vals[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, base_vals_r, 'o-', linewidth=2, label='Baseline', color='#4c72b0')
    ax.fill(angles, base_vals_r, alpha=0.15, color='#4c72b0')
    ax.plot(angles, de_vals_r, 'o-', linewidth=2, label='GWO-DE-NM Hybrid', color='#dd8452')
    ax.fill(angles, de_vals_r, alpha=0.15, color='#dd8452')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_title(f"Figure {fig_num}: Performance Radar -- {dataset_name}", pad=20, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"radar_{dataset_name}.png")
    plt.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved Figure {fig_num}: {path}")


def plot_time_comparison(baseline_runs: List[RunResult], opt_runs: List[RunResult],
                         dataset_name: str, output_folder: str) -> None:
    """Figure: Time comparison per seed."""
    if not HAS_PLT:
        return
    import matplotlib.pyplot as plt

    fig_num = _next_fig_num()
    seeds = list(range(1, len(baseline_runs) + 1))
    base_times = [r.time_total_sec for r in baseline_runs]
    de_times = [r.time_total_sec + r.time_opt_sec for r in opt_runs]

    x = np.arange(len(seeds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, base_times, width, label="Baseline", color="#4c72b0")
    ax.bar(x + width/2, de_times, width, label="GWO-DE-NM Hybrid (total)", color="#dd8452")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"Figure {fig_num}: Computation Time -- {dataset_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in seeds])
    ax.legend()
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, f"time_comparison_{dataset_name}.png")
    plt.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved Figure {fig_num}: {path}")


def generate_all_figures(
    dataset_name: str,
    baseline_runs: List[RunResult],
    opt_runs: List[RunResult],
    base_summ: Dict,
    de_summ: Dict,
    output_folder: str,
) -> int:
    """Generate all academic figures. Returns total figure count."""
    plots_dir = os.path.join(output_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\nGenerating academic figures for {dataset_name}...")
    start_count = FIGURE_COUNTER["n"]

    # Fig 1: Cluster distribution
    plot_cluster_distribution(baseline_runs, opt_runs, dataset_name, plots_dir)

    # Fig 2: External metrics horizontal bars
    plot_metric_comparison_bars(base_summ, de_summ, dataset_name, plots_dir)

    # Fig 3: Internal metrics
    plot_internal_metrics(base_summ, de_summ, dataset_name, plots_dir)

    # Fig 4: Radar chart
    plot_radar_chart(base_summ, de_summ, dataset_name, plots_dir)

    # Fig 5: Time comparison
    plot_time_comparison(baseline_runs, opt_runs, dataset_name, plots_dir)

    # Fig 6: Overall comparison bar
    plot_compare(base_summ, de_summ,
                 os.path.join(plots_dir, f"compare_baseline_vs_de_{dataset_name}.png"), show=0)

    n_figs = FIGURE_COUNTER["n"] - start_count
    print(f"  Total figures generated: {n_figs}")
    return n_figs


# ---------------------------------------------------------------------------
# Computational Complexity Analysis
# ---------------------------------------------------------------------------

def compute_complexity_table(n_samples: int, n_features: int, args: Any) -> List[Dict[str, str]]:
    """
    Compute theoretical computational complexity for each component/model
    in the proposed hybrid system.
    """
    dr_dim = args.dr_dim
    k = args.k
    de_maxiter = args.de_maxiter
    de_popsize = args.de_popsize
    subsample = args.subsample
    n_wolves = 6
    n_gwo_iter = 5
    nm_maxiter = 50

    rows = []

    # 1. PCA / Dimensionality Reduction
    rows.append({
        "Component": "PCA (StandardScaler + PCA)",
        "Operation": "Dimensionality Reduction",
        "Complexity": f"O(n * d * d') = O({n_samples} * {n_features} * {dr_dim})",
        "BigO": f"O(n * d * d')",
        "Actual_Ops": f"{n_samples * n_features * dr_dim:,}",
    })

    # 2. KNN Graph Construction
    rows.append({
        "Component": "KNN (sklearn NearestNeighbors)",
        "Operation": "k-NN Search + Graph Build",
        "Complexity": f"O(n * k * d') = O({n_samples} * {k} * {dr_dim})",
        "BigO": f"O(n * k * d')",
        "Actual_Ops": f"{n_samples * k * dr_dim:,}",
    })

    # 3. Edge Pruning
    rows.append({
        "Component": "Edge Pruning (quantile-based)",
        "Operation": "Prune weak edges",
        "Complexity": f"O(E) where E = n*k = {n_samples * k}",
        "BigO": "O(n * k)",
        "Actual_Ops": f"{n_samples * k:,}",
    })

    # 4. Louvain Community Detection
    rows.append({
        "Component": "Louvain Algorithm",
        "Operation": "Graph Clustering",
        "Complexity": f"O(n * log(n)) = O({n_samples} * {int(np.log2(max(2, n_samples)))})",
        "BigO": "O(n * log(n))",
        "Actual_Ops": f"{n_samples * int(np.log2(max(2, n_samples))):,}",
    })

    # 5. Silhouette Score
    rows.append({
        "Component": "Silhouette Score (sampled)",
        "Operation": "Cluster Quality Evaluation",
        "Complexity": f"O(s^2 * d') where s={args.sil_sample}",
        "BigO": "O(s^2 * d')",
        "Actual_Ops": f"{args.sil_sample ** 2 * dr_dim:,}",
    })

    # 6. GWO
    n_gwo_evals = n_wolves * (n_gwo_iter + 1)
    rows.append({
        "Component": "Grey Wolf Optimizer (GWO)",
        "Operation": "Meta-heuristic Global Search",
        "Complexity": f"O(W * I * F_cost) = O({n_wolves} * {n_gwo_iter} * F)",
        "BigO": "O(W * I * F)",
        "Actual_Ops": f"{n_gwo_evals} fitness evaluations (on subsample n'={subsample})",
    })

    # 7. DE
    n_de_evals = de_maxiter * de_popsize * 4  # 4 = dimension
    rows.append({
        "Component": "Differential Evolution (DE)",
        "Operation": "Evolutionary Optimization",
        "Complexity": f"O(G * P * D * F_cost) = O({de_maxiter} * {de_popsize} * 4 * F)",
        "BigO": "O(G * P * D * F)",
        "Actual_Ops": f"~{n_de_evals} fitness evaluations (on subsample n'={subsample})",
    })

    # 8. Nelder-Mead
    rows.append({
        "Component": "Nelder-Mead (Local Search)",
        "Operation": "Local Refinement",
        "Complexity": f"O(M * F_cost) = O({nm_maxiter} * F)",
        "BigO": "O(M * F)",
        "Actual_Ops": f"<={nm_maxiter} fitness evaluations (on subsample n'={subsample})",
    })

    # 9. Baseline Guard
    rows.append({
        "Component": "3-Layer QualityGate",
        "Operation": "Over-fragmentation Protection",
        "Complexity": "O(1) per seed",
        "BigO": "O(1)",
        "Actual_Ops": "3 comparisons per seed",
    })

    # 10. Post-processing
    rows.append({
        "Component": "Tiny Cluster Merging",
        "Operation": "Post-processing",
        "Complexity": f"O(n_tiny * C_large * d')",
        "BigO": "O(n_tiny * C * d')",
        "Actual_Ops": "Depends on cluster sizes",
    })

    # 11. Full Pipeline (Baseline)
    rows.append({
        "Component": "FULL BASELINE Pipeline",
        "Operation": "PCA + KNN + Louvain + Eval",
        "Complexity": f"O(n*d*d' + n*k*d' + n*log(n))",
        "BigO": "O(n * d * d')",
        "Actual_Ops": f"~{n_samples * n_features * dr_dim + n_samples * k * dr_dim:,}",
    })

    # 12. Full Hybrid Pipeline
    total_hybrid_fitness = n_gwo_evals + n_de_evals + nm_maxiter
    rows.append({
        "Component": "FULL HYBRID Pipeline (GWO-DE-NM)",
        "Operation": "GWO + DE + NM + Pipeline + Guard",
        "Complexity": f"O(T * F_cost + n*d*d' + n*k*d' + n*log(n))",
        "BigO": "O(T * F + n * d * d')",
        "Actual_Ops": f"~{total_hybrid_fitness} fitness evals + full pipeline",
    })

    return rows


def print_complexity_table(rows: List[Dict[str, str]]) -> None:
    """Print computational complexity table in academic format."""
    w = 100
    print("=" * w)
    print("COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("=" * w)
    print(f"{'Component':<35}| {'BigO':<20}| {'Actual Operations'}")
    print("-" * w)
    for row in rows:
        print(f"{row['Component']:<35}| {row['BigO']:<20}| {row['Actual_Ops']}")
    print("=" * w)
    print()
    print("Legend:")
    print("  n = number of samples, d = original features, d' = reduced dimensions")
    print("  k = KNN neighbors, W = GWO wolves, I = GWO iterations")
    print("  G = DE generations, P = DE population, D = search dimensions (4)")
    print("  M = Nelder-Mead max iterations, F = single fitness evaluation cost")
    print("  s = silhouette sample size, T = total fitness evaluations")
    print("  F_cost = O(n'*k*d' + n'*log(n')) where n' = subsample size")
    print("=" * w)


def generate_complexity_latex(rows: List[Dict[str, str]], dataset_name: str) -> str:
    """Generate LaTeX table for computational complexity."""
    safe_name = dataset_name.replace("_", "\\_")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{Computational complexity of the proposed GWO--DE--NM hybrid system on \\textit{{{safe_name}}}.}}",
        f"\\label{{tab:complexity_{dataset_name.lower().replace(' ', '_')}}}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Component & Complexity & Evaluations \\",
        r"\midrule",
    ]
    for row in rows:
        comp = row["Component"].replace("_", "\\_")
        bigo = f"${row['BigO']}$".replace("O(", r"\mathcal{O}(")
        ops = row["Actual_Ops"].replace("_", "\\_")
        if row["Component"].startswith("FULL"):
            lines.append(r"\midrule")
        lines.append(f"{comp} & {bigo} & {ops} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Academic output (delta table, LaTeX, summary)
# ---------------------------------------------------------------------------

def _fmt_mean_std(m: float, s: float, decimals: int = 3) -> str:
    if not np.isfinite(m):
        return "N/A"
    return f"{m:.{decimals}f} +/- {s:.{decimals}f}"


def _fmt_latex_val(m: float, s: float, decimals: int = 3, bold: bool = False) -> str:
    if not np.isfinite(m):
        return "N/A"
    inner = f"{m:.{decimals}f} \\pm {s:.{decimals}f}"
    if bold:
        return f"$\\mathbf{{{inner}}}$"
    return f"${inner}$"


def generate_delta_table(
    base_summ: Dict[str, Dict[str, float]],
    de_summ: Dict[str, Dict[str, float]],
) -> str:
    metrics = ["ARI", "NMI", "ACC", "SIL", "MOD"]
    header = f"{'Metric':<9}| {'Baseline':<20}| {'Hybrid (GWO-DE-NM)':<20}| {'Delta':<9}| {'Change':<8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for k in metrics:
        bm, bs = base_summ[k]["mean"], base_summ[k]["std"]
        dm, ds = de_summ[k]["mean"], de_summ[k]["std"]
        if not np.isfinite(bm) or not np.isfinite(dm):
            lines.append(f"{k:<9}| {'N/A':<20}| {'N/A':<20}| {'N/A':<9}| {'N/A':<8}")
            continue
        delta = dm - bm
        pct = 100.0 * delta / abs(bm) if abs(bm) > 1e-9 else 0.0
        lines.append(
            f"{k:<9}| {_fmt_mean_std(bm, bs):<20}| {_fmt_mean_std(dm, ds):<20}"
            f"| {delta:>+7.3f}  | {pct:>+6.1f}%"
        )
    lines.append(sep)
    for k in ["C", "time_total_sec"]:
        label = "C" if k == "C" else "Time (s)"
        bm, bs = base_summ[k]["mean"], base_summ[k]["std"]
        dm, ds = de_summ[k]["mean"], de_summ[k]["std"]
        dec = 1 if k == "C" else 2
        lines.append(
            f"{label:<9}| {_fmt_mean_std(bm, bs, dec):<20}| {_fmt_mean_std(dm, ds, dec):<20}|          |"
        )
    return "\n".join(lines)


def generate_latex_table(
    dataset_name: str,
    base_summ: Dict[str, Dict[str, float]],
    de_summ: Dict[str, Dict[str, float]],
) -> str:
    safe_name = dataset_name.replace("_", "\\_")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Clustering performance on \\textit{{{safe_name}}}: Baseline vs.\\ GWO--DE--NM Hybrid.}}",
        f"\\label{{tab:{dataset_name.lower().replace(' ', '_')}}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Baseline & Hybrid (GWO--DE--NM) \\",
        r"\midrule",
    ]
    metrics = ["ARI", "NMI", "ACC", "SIL", "MOD"]
    for k in metrics:
        bm, bs = base_summ[k]["mean"], base_summ[k]["std"]
        dm, ds = de_summ[k]["mean"], de_summ[k]["std"]
        b_bold = np.isfinite(bm) and np.isfinite(dm) and bm > dm
        d_bold = np.isfinite(bm) and np.isfinite(dm) and dm >= bm
        b_str = _fmt_latex_val(bm, bs, 3, bold=b_bold)
        d_str = _fmt_latex_val(dm, ds, 3, bold=d_bold)
        lines.append(f"{k} & {b_str} & {d_str} \\\\")
    lines.append(r"\midrule")
    bm, bs = base_summ["C"]["mean"], base_summ["C"]["std"]
    dm, ds = de_summ["C"]["mean"], de_summ["C"]["std"]
    lines.append(f"$C$ & {_fmt_latex_val(bm, bs, 1)} & {_fmt_latex_val(dm, ds, 1)} \\\\")
    bm, bs = base_summ["time_total_sec"]["mean"], base_summ["time_total_sec"]["std"]
    dm, ds = de_summ["time_total_sec"]["mean"], de_summ["time_total_sec"]["std"]
    lines.append(f"Time (s) & {_fmt_latex_val(bm, bs, 2)} & {_fmt_latex_val(dm, ds, 2)} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def save_latex_table(latex_str: str, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_str)


def print_academic_summary(
    dataset_name: str,
    n_samples: int,
    n_features: int,
    n_seeds: int,
    base_summ: Dict[str, Dict[str, float]],
    de_summ: Dict[str, Dict[str, float]],
    args: Any,
) -> None:
    w = 80
    print("=" * w)
    print("ACADEMIC RESULTS SUMMARY")
    print("=" * w)
    print(f"Dataset: {dataset_name} | n={n_samples} | d={n_features} | Seeds: {n_seeds}")
    print(f"Method: GWO-DE-NM Hybrid Graph Clustering with 3-Layer QualityGate")
    print(f"Fitness: w_mod={args.w_mod}, w_sil={args.w_sil}, w_bal={args.w_bal}")
    print(f"Metric: {args.metric} | Graph: {args.graph_mode} | Backend: Louvain")
    print("-" * w)
    print()
    print(f"TABLE 1: Performance Comparison (mean +/- std over {n_seeds} seeds)")
    print("-" * w)
    print(generate_delta_table(base_summ, de_summ))
    print()
    has_labels = np.isfinite(base_summ["ARI"]["mean"])
    if has_labels:
        print("Note: External metrics (ARI, NMI, ACC) computed using ground-truth labels.")
    else:
        print("Note: No ground-truth labels available. External metrics show N/A.")
    print()

    # Complexity Analysis
    complexity_rows = compute_complexity_table(n_samples, n_features, args)
    print_complexity_table(complexity_rows)
    print()

    print(f"TABLE 2: LaTeX-ready performance table")
    print("-" * w)
    print(generate_latex_table(dataset_name, base_summ, de_summ))
    print()
    print(f"TABLE 3: LaTeX-ready complexity table")
    print("-" * w)
    print(generate_complexity_latex(complexity_rows, dataset_name))
    print("=" * w)


# ===========================================================================
# CROSS-DATASET COMPARISON CHARTS (Academic Style)
# ===========================================================================

def _academic_style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    """Apply consistent academic styling to axes."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


# Colors
_C_BASE = '#4c72b0'
_C_GWO = '#55a868'
_C_HYBRID = '#dd8452'
_C_QG = '#8c564b'
_C_ACCENT = '#c44e52'


def plot_cross_dataset_external(all_summaries: List[Dict], output_folder: str) -> None:
    """Fig: Grouped bar chart ARI / NMI / ACC across all datasets."""
    if not HAS_PLT:
        return
    import matplotlib.pyplot as plt

    names = [s["name"] for s in all_summaries]
    n = len(names)

    for metric in ["ARI", "NMI", "ACC"]:
        base_vals = [s["base_summ"][metric]["mean"] for s in all_summaries]
        base_errs = [s["base_summ"][metric]["std"] for s in all_summaries]
        de_vals = [s["de_summ"][metric]["mean"] for s in all_summaries]
        de_errs = [s["de_summ"][metric]["std"] for s in all_summaries]

        fig, ax = plt.subplots(figsize=(max(12, n * 1.4), 6))
        x = np.arange(n)
        w = 0.35
        ax.bar(x - w/2, base_vals, w, yerr=base_errs, capsize=4,
               label="Baseline", color=_C_BASE, edgecolor='white', linewidth=0.5)
        ax.bar(x + w/2, de_vals, w, yerr=de_errs, capsize=4,
               label="GWO-DE-NM Hybrid", color=_C_HYBRID, edgecolor='white', linewidth=0.5)

        # Value labels
        for i in range(n):
            ax.text(i - w/2, base_vals[i] + base_errs[i] + 0.015,
                    f"{base_vals[i]:.3f}", ha='center', va='bottom', fontsize=8, color=_C_BASE)
            ax.text(i + w/2, de_vals[i] + de_errs[i] + 0.015,
                    f"{de_vals[i]:.3f}", ha='center', va='bottom', fontsize=8, color=_C_HYBRID)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')
        _academic_style(ax, f"{metric} Comparison Across All Datasets", ylabel=metric)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        path = os.path.join(output_folder, f"cross_{metric.lower()}_comparison.png")
        plt.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"  Saved: {path}")


def plot_cross_dataset_heatmap(all_summaries: List[Dict], output_folder: str) -> None:
    """Fig: Heatmap of improvement (DEAC - Baseline), landscape layout.

    Design choices (IEEE-compliant):
      - Transposed: datasets on X-axis, metrics on Y-axis (landscape, matches
        the other cross-dataset panels in Fig 6).
      - Rows (metrics) ordered: ARI, NMI, ACC, SIL, MOD.
      - Columns (datasets) sorted by Delta-ARI descending so the visual story
        flows from biggest gains to zero-improvement cases.
      - Colorblind-safe diverging palette (RdBu_r): red = worse, blue = better.
      - Cell values annotated; diagonal bold gridlines between columns.
      - Title and colorbar label use the current "DEAC" naming.
    """
    if not HAS_PLT:
        return
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    names = [s["name"] for s in all_summaries]
    metrics = ["ARI", "NMI", "ACC", "SIL", "MOD"]
    n_ds = len(names)
    n_m = len(metrics)

    # Build delta matrix: rows = datasets, cols = metrics.
    delta = np.zeros((n_ds, n_m))
    for i, s in enumerate(all_summaries):
        for j, m in enumerate(metrics):
            bv = s["base_summ"][m]["mean"]
            dv = s["de_summ"][m]["mean"]
            delta[i, j] = dv - bv

    # Sort datasets by Delta-ARI descending (column 0).
    ari_order = np.argsort(-delta[:, 0])
    names_sorted = [names[i] for i in ari_order]
    delta_sorted = delta[ari_order]

    # Transpose: rows = metrics, cols = datasets (landscape).
    M = delta_sorted.T  # shape (n_m, n_ds)

    # Figure size scales with dataset count; keep aspect landscape.
    fig_w = max(8.0, 0.55 * n_ds + 1.8)
    fig_h = max(3.2, 0.55 * n_m + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = max(abs(M.min()), abs(M.max()), 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    # Colorblind-safe diverging palette (red = worse, blue = better).
    im = ax.imshow(M, cmap='RdBu_r', aspect='auto', norm=norm)
    # Flip so positive (better) shows as blue in standard viridis-like direction.
    # RdBu_r: low=blue, high=red -> we want high=blue (better), so use 'RdBu'.
    im.set_cmap('RdBu')

    # Axes
    ax.set_xticks(np.arange(n_ds))
    ax.set_xticklabels(names_sorted, fontsize=9, rotation=35, ha='right')
    ax.set_yticks(np.arange(n_m))
    ax.set_yticklabels(metrics, fontsize=10, fontweight='bold')

    # Thin gridlines between cells
    ax.set_xticks(np.arange(-0.5, n_ds, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_m, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8)
    ax.tick_params(which='minor', length=0)

    # Annotate cells
    for i in range(n_m):
        for j in range(n_ds):
            val = M[i, j]
            color = 'white' if abs(val) > vmax * 0.55 else 'black'
            ax.text(j, i, f"{val:+.3f}", ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color=color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.015, aspect=15,
                        orientation='vertical')
    cbar.set_label(r"$\Delta$ = DEAC $-$ Baseline", fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Multi-Metric Improvement: DEAC vs. Baseline (sorted by $\\Delta$ARI)",
                 fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel("Dataset (sorted by $\\Delta$ARI descending)",
                  fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_folder, "cross_improvement_heatmap.png")
    plt.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved: {path}")


def plot_cross_dataset_improvement_pct(all_summaries: List[Dict], output_folder: str) -> None:
    """Fig: Percentage improvement bar chart for ARI across datasets."""
    if not HAS_PLT:
        return
    import matplotlib.pyplot as plt

    names = [s["name"] for s in all_summaries]
    n = len(names)
    pct_improvements = []
    for s in all_summaries:
        bv = s["base_summ"]["ARI"]["mean"]
        dv = s["de_summ"]["ARI"]["mean"]
        if abs(bv) > 1e-9:
            pct = 100.0 * (dv - bv) / abs(bv)
        else:
            pct = 100.0 * dv if dv > 0 else 0.0
        pct_improvements.append(pct)

    colors = [_C_HYBRID if p > 0 else _C_QG for p in pct_improvements]

    fig, ax = plt.subplots(figsize=(max(10, n * 1.3), 6))
    x = np.arange(n)
    bars = ax.bar(x, pct_improvements, 0.6, color=colors, edgecolor='white', linewidth=0.5)

    for i, (bar, pct) in enumerate(zip(bars, pct_improvements)):
        y_pos = bar.get_height() + (1 if pct >= 0 else -3)
        ax.text(i, y_pos, f"{pct:+.1f}%", ha='center', va='bottom' if pct >= 0 else 'top',
                fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    _academic_style(ax, "ARI Improvement: Hybrid vs. Baseline (%)", ylabel="Improvement (%)")
    plt.tight_layout()
    path = os.path.join(output_folder, "cross_ari_improvement_pct.png")
    plt.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved: {path}")


def plot_cross_dataset_radar(all_summaries: List[Dict], output_folder: str) -> None:
    """Fig: Multi-dataset radar chart comparing Hybrid performance."""
    if not HAS_PLT:
        return
    import matplotlib.pyplot as plt

    metrics = ["ARI", "NMI", "ACC", "SIL", "MOD"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    cmap = plt.cm.get_cmap('tab10')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=dict(polar=True))

    for ax_idx, (ax, mode, label) in enumerate(zip(axes, ['base_summ', 'de_summ'],
                                                      ['Baseline', 'GWO-DE-NM Hybrid'])):
        for i, s in enumerate(all_summaries):
            vals = [max(0, s[mode][m]["mean"]) for m in metrics]
            vals += vals[:1]
            color = cmap(i / max(len(all_summaries) - 1, 1))
            ax.plot(angles, vals, 'o-', linewidth=1.8, label=s["name"], color=color, markersize=4)
            ax.fill(angles, vals, alpha=0.05, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_title(label, fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.suptitle("Cross-Dataset Performance Radar", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_folder, "cross_radar_all_datasets.png")
    plt.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved: {path}")


def plot_quality_gate_summary(all_summaries: List[Dict], output_folder: str) -> None:
    """Fig: QualityGate activation summary -- which datasets triggered the guard."""
    if not HAS_PLT:
        return
    import matplotlib.pyplot as plt

    names = [s["name"] for s in all_summaries]
    n = len(names)

    base_ari = [s["base_summ"]["ARI"]["mean"] for s in all_summaries]
    de_ari = [s["de_summ"]["ARI"]["mean"] for s in all_summaries]
    qg_used = [s.get("qg_used", False) for s in all_summaries]

    fig, ax = plt.subplots(figsize=(max(12, n * 1.4), 6))
    x = np.arange(n)
    w = 0.35

    ax.bar(x - w/2, base_ari, w, label="Baseline", color=_C_BASE, edgecolor='white')
    bar_colors = [_C_QG if g else _C_HYBRID for g in qg_used]
    ax.bar(x + w/2, de_ari, w, label="Hybrid", color=bar_colors, edgecolor='white')

    # Guard annotation
    for i, g in enumerate(qg_used):
        if g:
            ax.annotate('QG', xy=(i + w/2, de_ari[i]),
                        xytext=(i + w/2, de_ari[i] + 0.05),
                        ha='center', fontsize=8, fontweight='bold', color=_C_ACCENT,
                        arrowprops=dict(arrowstyle='->', color=_C_ACCENT, lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    _academic_style(ax, "3-Layer QualityGate Activation Summary", ylabel="ARI")
    ax.legend(fontsize=11)

    # Custom legend entries for guard
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_C_BASE, label='Baseline'),
        Patch(facecolor=_C_HYBRID, label='Hybrid (DE improved)'),
        Patch(facecolor=_C_QG, label='Hybrid (QualityGate activated)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
    plt.tight_layout()
    path = os.path.join(output_folder, "cross_qualitygate_summary.png")
    plt.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved: {path}")


def plot_dataset_characteristics(all_summaries: List[Dict], output_folder: str) -> None:
    """Fig: Dataset characteristics table as figure (n, d, classes)."""
    if not HAS_PLT:
        return
    import matplotlib.pyplot as plt

    names = [s["name"] for s in all_summaries]
    n_samples = [s["n"] for s in all_summaries]
    n_features = [s["d"] for s in all_summaries]
    n_ds = len(names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, n_ds * 0.5)))

    # Samples
    y = np.arange(n_ds)
    ax1.barh(y, n_samples, 0.6, color=_C_BASE, edgecolor='white')
    for i, v in enumerate(n_samples):
        ax1.text(v + max(n_samples) * 0.02, i, f"{v:,}", va='center', fontsize=10)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=11)
    _academic_style(ax1, "Number of Samples (n)", xlabel="Samples")
    ax1.invert_yaxis()

    # Features
    ax2.barh(y, n_features, 0.6, color=_C_HYBRID, edgecolor='white')
    for i, v in enumerate(n_features):
        ax2.text(v + max(n_features) * 0.02, i, f"{v:,}", va='center', fontsize=10)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=11)
    _academic_style(ax2, "Number of Features (d)", xlabel="Features")
    ax2.invert_yaxis()

    plt.suptitle("Benchmark Dataset Characteristics", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_folder, "cross_dataset_characteristics.png")
    plt.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved: {path}")


def plot_cross_all_metrics_grouped(all_summaries: List[Dict], output_folder: str) -> None:
    """Fig: Grand comparison -- all 5 metrics side by side for all datasets."""
    if not HAS_PLT:
        return
    import matplotlib.pyplot as plt

    metrics = ["ARI", "NMI", "ACC", "SIL", "MOD"]
    names = [s["name"] for s in all_summaries]
    n_ds = len(names)
    n_m = len(metrics)

    fig, axes = plt.subplots(1, n_m, figsize=(n_m * 3.5, max(6, n_ds * 0.55)), sharey=True)
    y = np.arange(n_ds)
    h = 0.35

    for j, (ax, metric) in enumerate(zip(axes, metrics)):
        base_vals = [s["base_summ"][metric]["mean"] for s in all_summaries]
        base_errs = [s["base_summ"][metric]["std"] for s in all_summaries]
        de_vals = [s["de_summ"][metric]["mean"] for s in all_summaries]
        de_errs = [s["de_summ"][metric]["std"] for s in all_summaries]

        ax.barh(y - h/2, base_vals, h, xerr=base_errs, capsize=3,
                label="Baseline" if j == 0 else "", color=_C_BASE, edgecolor='white')
        ax.barh(y + h/2, de_vals, h, xerr=de_errs, capsize=3,
                label="Hybrid" if j == 0 else "", color=_C_HYBRID, edgecolor='white')

        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_xlabel("Score", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        if j == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=10)
            ax.invert_yaxis()

    axes[0].legend(fontsize=10, loc='lower left')
    plt.suptitle("Baseline vs. Hybrid: All Metrics Across Datasets",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_folder, "cross_all_metrics_grouped.png")
    plt.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved: {path}")


def generate_cross_dataset_latex(all_summaries: List[Dict], output_folder: str) -> None:
    """Generate comprehensive LaTeX table for all datasets comparison."""
    metrics = ["ARI", "NMI", "ACC", "SIL", "MOD"]
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Cross-dataset performance comparison: Baseline vs.\ GWO--DE--NM Hybrid. Bold values indicate the better method per metric. QualityGate column indicates whether the 3-Layer QualityGate was activated.}",
        r"\label{tab:cross_dataset}",
        r"\begin{tabular}{l" + "cc" * len(metrics) + r"c}",
        r"\toprule",
    ]
    # Header row 1
    header1 = "Dataset"
    for m in metrics:
        header1 += f" & \\multicolumn{{2}}{{c}}{{{m}}}"
    header1 += " & QG \\\\"
    lines.append(header1)

    # Header row 2
    header2 = ""
    for m in metrics:
        header2 += " & Base & Hybrid"
    header2 += " & \\\\"
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}")
    lines.append(header2)
    lines.append(r"\midrule")

    for s in all_summaries:
        safe_name = s["name"].replace("_", "\\_")
        row = safe_name
        for m in metrics:
            bm = s["base_summ"][m]["mean"]
            dm = s["de_summ"][m]["mean"]
            b_str = f"{bm:.3f}" if np.isfinite(bm) else "N/A"
            d_str = f"{dm:.3f}" if np.isfinite(dm) else "N/A"
            if np.isfinite(bm) and np.isfinite(dm):
                if dm > bm:
                    d_str = f"\\textbf{{{d_str}}}"
                elif bm > dm:
                    b_str = f"\\textbf{{{b_str}}}"
                else:
                    b_str = f"\\textbf{{{b_str}}}"
                    d_str = f"\\textbf{{{d_str}}}"
            row += f" & {b_str} & {d_str}"
        guard = "Yes" if s.get("qg_used", False) else "No"
        row += f" & {guard} \\\\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    latex_str = "\n".join(lines)
    path = os.path.join(output_folder, "cross_dataset_comparison.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"  Saved LaTeX: {path}")


def generate_all_cross_dataset_figures(all_summaries: List[Dict], output_folder: str) -> int:
    """Generate all cross-dataset comparison figures. Returns figure count."""
    comp_dir = os.path.join(output_folder, "comparison_plots")
    os.makedirs(comp_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("GENERATING CROSS-DATASET COMPARISON FIGURES")
    print(f"{'='*80}")

    plot_dataset_characteristics(all_summaries, comp_dir)        # 1
    plot_cross_dataset_external(all_summaries, comp_dir)         # 2 (3 figures: ARI, NMI, ACC)
    plot_cross_dataset_heatmap(all_summaries, comp_dir)          # 3
    plot_cross_dataset_improvement_pct(all_summaries, comp_dir)  # 4
    plot_cross_dataset_radar(all_summaries, comp_dir)            # 5
    plot_quality_gate_summary(all_summaries, comp_dir)                  # 6
    plot_cross_all_metrics_grouped(all_summaries, comp_dir)      # 7
    generate_cross_dataset_latex(all_summaries, comp_dir)        # LaTeX table

    n_figs = 8  # 7 PNG + 1 LaTeX
    print(f"\n  Cross-dataset figures: {n_figs} files saved to {comp_dir}/")
    return n_figs


# ============================================================================
# 3-WAY COMPARISON FIGURES: Baseline vs LevGWO-only vs LevGWO+DE+NM
# ============================================================================

def plot_metric_comparison_bars_3way(base_summ, gwo_summ, de_summ, dataset_name, output_folder):
    """3-way horizontal bar chart for external metrics (ARI, NMI, ACC)."""
    if not HAS_PLT: return
    import matplotlib.pyplot as plt

    metrics = ["ARI", "NMI", "ACC"]
    base_vals = [base_summ[m]["mean"] for m in metrics]
    gwo_vals = [gwo_summ[m]["mean"] for m in metrics]
    de_vals = [de_summ[m]["mean"] for m in metrics]

    y = np.arange(len(metrics))
    h = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(y - h, base_vals, h, label="Baseline", color=_C_BASE, edgecolor='white')
    ax.barh(y, gwo_vals, h, label="LevGWO-only", color=_C_GWO, edgecolor='white')
    ax.barh(y + h, de_vals, h, label="LevGWO+DE+NM", color=_C_HYBRID, edgecolor='white')

    for i in range(len(metrics)):
        ax.text(base_vals[i] + 0.005, y[i] - h, f"{base_vals[i]:.3f}", va='center', fontsize=8, color=_C_BASE)
        ax.text(gwo_vals[i] + 0.005, y[i], f"{gwo_vals[i]:.3f}", va='center', fontsize=8, color=_C_GWO)
        ax.text(de_vals[i] + 0.005, y[i] + h, f"{de_vals[i]:.3f}", va='center', fontsize=8, color=_C_HYBRID)

    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=12)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title(f"External Metrics Comparison — {dataset_name}", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    fig_num = _next_fig_num()
    path = os.path.join(output_folder, f"external_metrics_3way_{dataset_name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved Figure {fig_num}: {path}")


def plot_internal_metrics_3way(base_summ, gwo_summ, de_summ, dataset_name, output_folder):
    """3-way bar chart for internal metrics (SIL, MOD, Balance)."""
    if not HAS_PLT: return
    import matplotlib.pyplot as plt

    metrics = ["SIL", "MOD", "bal01"]
    labels = ["Silhouette", "Modularity", "Balance"]

    def safe_val(summ, m):
        v = summ.get(m, {}).get("mean", 0.0)
        return v if np.isfinite(v) else 0.0

    base_vals = [safe_val(base_summ, m) for m in metrics]
    gwo_vals = [safe_val(gwo_summ, m) for m in metrics]
    de_vals = [safe_val(de_summ, m) for m in metrics]

    x = np.arange(len(metrics))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, base_vals, w, label="Baseline", color=_C_BASE, edgecolor='white')
    ax.bar(x, gwo_vals, w, label="LevGWO-only", color=_C_GWO, edgecolor='white')
    ax.bar(x + w, de_vals, w, label="LevGWO+DE+NM", color=_C_HYBRID, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Internal Metrics Comparison — {dataset_name}", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig_num = _next_fig_num()
    path = os.path.join(output_folder, f"internal_metrics_3way_{dataset_name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved Figure {fig_num}: {path}")


def plot_radar_chart_3way(base_summ, gwo_summ, de_summ, dataset_name, output_folder):
    """3-polygon radar chart for multi-metric comparison."""
    if not HAS_PLT: return
    import matplotlib.pyplot as plt

    metrics = ["ARI", "NMI", "ACC", "SIL", "MOD"]

    def safe_val(summ, m):
        v = summ.get(m, {}).get("mean", 0.0)
        return max(0.0, v) if np.isfinite(v) else 0.0

    base_vals = [safe_val(base_summ, m) for m in metrics]
    gwo_vals = [safe_val(gwo_summ, m) for m in metrics]
    de_vals = [safe_val(de_summ, m) for m in metrics]

    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    base_vals += base_vals[:1]
    gwo_vals += gwo_vals[:1]
    de_vals += de_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, base_vals, 'o-', linewidth=2, label='Baseline', color=_C_BASE)
    ax.fill(angles, base_vals, alpha=0.1, color=_C_BASE)
    ax.plot(angles, gwo_vals, 's-', linewidth=2, label='LevGWO-only', color=_C_GWO)
    ax.fill(angles, gwo_vals, alpha=0.15, color=_C_GWO)
    ax.plot(angles, de_vals, 'D-', linewidth=2, label='LevGWO+DE+NM', color=_C_HYBRID)
    ax.fill(angles, de_vals, alpha=0.15, color=_C_HYBRID)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_title(f"Radar Chart — {dataset_name}", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig_num = _next_fig_num()
    path = os.path.join(output_folder, f"radar_3way_{dataset_name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved Figure {fig_num}: {path}")


def plot_gwo_convergence(convergence_data, dataset_name, output_folder):
    """Plot GWO fitness convergence curve per seed."""
    if not HAS_PLT or not convergence_data: return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

    for idx, (conv, _) in enumerate(convergence_data):
        if conv is None: continue
        iters = list(range(len(conv)))
        fitness = [-v for v in conv]  # negate (lower is better → higher quality)
        seed = 1001 + idx
        ax.plot(iters, fitness, 'o-', linewidth=2, markersize=5,
                color=colors[idx % len(colors)], label=f'Seed {seed}')

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Fitness Score (higher = better)", fontsize=12)
    ax.set_title(f"LevGWO Convergence Curve — {dataset_name}", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig_num = _next_fig_num()
    path = os.path.join(output_folder, f"gwo_convergence_{dataset_name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved Figure {fig_num}: {path}")


def plot_levy_step_distribution(convergence_data, dataset_name, output_folder):
    """Histogram of Levy flight step magnitudes showing heavy-tail distribution."""
    if not HAS_PLT or not convergence_data: return
    import matplotlib.pyplot as plt

    all_steps = []
    for (_, steps) in convergence_data:
        if steps is not None:
            all_steps.extend(steps)

    if not all_steps: return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(all_steps, bins=80, density=True, alpha=0.7, color=_C_GWO,
            edgecolor='white', label='Observed Levy steps')

    # Add log-scale inset
    ax.set_xlabel("Step Magnitude |L|", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Levy Flight Step Distribution (β=1.5) — {dataset_name}",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add stats annotation
    mean_s = np.mean(all_steps)
    median_s = np.median(all_steps)
    max_s = np.max(all_steps)
    ax.annotate(f"Mean = {mean_s:.2f}\nMedian = {median_s:.2f}\nMax = {max_s:.1f}",
                xy=(0.75, 0.85), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig_num = _next_fig_num()
    path = os.path.join(output_folder, f"levy_distribution_{dataset_name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved Figure {fig_num}: {path}")


def plot_phase_contribution(base_summ, gwo_summ, de_summ, dataset_name, output_folder):
    """Waterfall chart showing contribution of each optimization phase."""
    if not HAS_PLT: return
    import matplotlib.pyplot as plt

    metrics = ["ARI", "NMI", "ACC", "SIL", "MOD"]

    def safe_val(summ, m):
        v = summ.get(m, {}).get("mean", 0.0)
        return v if np.isfinite(v) else 0.0

    base_vals = [safe_val(base_summ, m) for m in metrics]
    gwo_contrib = [safe_val(gwo_summ, m) - safe_val(base_summ, m) for m in metrics]
    de_contrib = [safe_val(de_summ, m) - safe_val(gwo_summ, m) for m in metrics]

    x = np.arange(len(metrics))
    w = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline bars
    ax.bar(x, base_vals, w, label="Baseline", color=_C_BASE, edgecolor='white')

    # GWO contribution (stacked on top of baseline)
    gwo_colors = [_C_GWO if v >= 0 else '#e74c3c' for v in gwo_contrib]
    ax.bar(x, gwo_contrib, w, bottom=base_vals, label="+ LevGWO Phase",
           color=gwo_colors, edgecolor='white', hatch='///')

    # DE+NM contribution (stacked on top of GWO)
    gwo_tops = [b + g for b, g in zip(base_vals, gwo_contrib)]
    de_colors = [_C_HYBRID if v >= 0 else '#e74c3c' for v in de_contrib]
    ax.bar(x, de_contrib, w, bottom=gwo_tops, label="+ DE+NM Phase",
           color=de_colors, edgecolor='white', hatch='...')

    # Value labels on top
    for i in range(len(metrics)):
        total = base_vals[i] + gwo_contrib[i] + de_contrib[i]
        ax.text(x[i], total + 0.01, f"{total:.3f}", ha='center', va='bottom',
                fontsize=9, fontweight='bold')
        if abs(gwo_contrib[i]) > 0.001:
            mid_gwo = base_vals[i] + gwo_contrib[i] / 2
            ax.text(x[i], mid_gwo, f"{gwo_contrib[i]:+.3f}", ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        if abs(de_contrib[i]) > 0.001:
            mid_de = gwo_tops[i] + de_contrib[i] / 2
            ax.text(x[i], mid_de, f"{de_contrib[i]:+.3f}", ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Phase Contribution Analysis — {dataset_name}", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    fig_num = _next_fig_num()
    path = os.path.join(output_folder, f"phase_contribution_{dataset_name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved Figure {fig_num}: {path}")


def plot_cluster_distribution_3way(baseline_runs, gwo_runs, opt_runs, dataset_name, output_folder):
    """3-way cluster count comparison per seed."""
    if not HAS_PLT: return
    import matplotlib.pyplot as plt

    seeds = [r.seed for r in baseline_runs]
    base_c = [r.C for r in baseline_runs]
    gwo_c = [r.C for r in gwo_runs]
    de_c = [r.C for r in opt_runs]

    x = np.arange(len(seeds))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, base_c, w, label="Baseline", color=_C_BASE, edgecolor='white')
    ax.bar(x, gwo_c, w, label="LevGWO-only", color=_C_GWO, edgecolor='white')
    ax.bar(x + w, de_c, w, label="LevGWO+DE+NM", color=_C_HYBRID, edgecolor='white')

    for i in range(len(seeds)):
        ax.text(x[i] - w, base_c[i] + 0.3, str(base_c[i]), ha='center', fontsize=9)
        ax.text(x[i], gwo_c[i] + 0.3, str(gwo_c[i]), ha='center', fontsize=9)
        ax.text(x[i] + w, de_c[i] + 0.3, str(de_c[i]), ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=11)
    ax.set_ylabel("Number of Clusters (C)", fontsize=12)
    ax.set_title(f"Cluster Count Comparison — {dataset_name}", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig_num = _next_fig_num()
    path = os.path.join(output_folder, f"cluster_dist_3way_{dataset_name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved Figure {fig_num}: {path}")


def plot_time_comparison_3way(baseline_runs, gwo_runs, opt_runs, dataset_name, output_folder):
    """3-way execution time comparison per seed."""
    if not HAS_PLT: return
    import matplotlib.pyplot as plt

    seeds = [r.seed for r in baseline_runs]
    base_t = [r.time_total_sec for r in baseline_runs]
    gwo_t = [r.time_total_sec for r in gwo_runs]
    de_t = [r.time_total_sec + r.time_opt_sec for r in opt_runs]

    x = np.arange(len(seeds))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, base_t, w, label="Baseline", color=_C_BASE, edgecolor='white')
    ax.bar(x, gwo_t, w, label="LevGWO-only", color=_C_GWO, edgecolor='white')
    ax.bar(x + w, de_t, w, label="LevGWO+DE+NM (total)", color=_C_HYBRID, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=11)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title(f"Execution Time Comparison — {dataset_name}", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig_num = _next_fig_num()
    path = os.path.join(output_folder, f"time_3way_{dataset_name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved Figure {fig_num}: {path}")


def generate_latex_table_3way(dataset_name, base_summ, gwo_summ, de_summ):
    """Generate LaTeX table with 3-column comparison."""
    metrics = ["ARI", "NMI", "ACC", "SIL", "MOD"]

    def fmt(summ, m):
        v = summ.get(m, {})
        mean = v.get("mean", 0.0)
        std = v.get("std", 0.0)
        if not np.isfinite(mean): return "N/A"
        return f"${mean:.3f} \\pm {std:.3f}$"

    def fmt_bold(summ, m, is_best):
        v = summ.get(m, {})
        mean = v.get("mean", 0.0)
        std = v.get("std", 0.0)
        if not np.isfinite(mean): return "N/A"
        if is_best:
            return f"$\\mathbf{{{mean:.3f} \\pm {std:.3f}}}$"
        return f"${mean:.3f} \\pm {std:.3f}$"

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{3-Way comparison on \\textit{{{dataset_name}}}: Baseline vs.\\ LevGWO vs.\\ LevGWO+DE+NM.}}",
        f"\\label{{tab:{dataset_name.lower()}_3way}}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Metric & Baseline & LevGWO-only & LevGWO+DE+NM \\\\",
        "\\midrule",
    ]

    for m in metrics:
        vals = [
            base_summ.get(m, {}).get("mean", float("nan")),
            gwo_summ.get(m, {}).get("mean", float("nan")),
            de_summ.get(m, {}).get("mean", float("nan")),
        ]
        finite_vals = [v for v in vals if np.isfinite(v)]
        best_val = max(finite_vals) if finite_vals else None

        b_best = best_val is not None and np.isfinite(vals[0]) and abs(vals[0] - best_val) < 1e-6
        g_best = best_val is not None and np.isfinite(vals[1]) and abs(vals[1] - best_val) < 1e-6
        d_best = best_val is not None and np.isfinite(vals[2]) and abs(vals[2] - best_val) < 1e-6

        lines.append(f"{m} & {fmt_bold(base_summ, m, b_best)} & "
                     f"{fmt_bold(gwo_summ, m, g_best)} & "
                     f"{fmt_bold(de_summ, m, d_best)} \\\\")

    # Add C row
    for label, key in [("$C$", "C"), ("Time (s)", "time_total_sec")]:
        lines.append(f"{label} & {fmt(base_summ, key)} & {fmt(gwo_summ, key)} & {fmt(de_summ, key)} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def generate_all_figures_3way(
    dataset_name, baseline_runs, gwo_runs, opt_runs,
    base_summ, gwo_summ, de_summ, output_folder,
    convergence_data=None,
):
    """Generate all 3-way academic figures. Returns total figure count."""
    plots_dir = os.path.join(output_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\nGenerating 3-way academic figures for {dataset_name}...")
    start_count = FIGURE_COUNTER["n"]

    # Fig 1: 3-way cluster distribution
    plot_cluster_distribution_3way(baseline_runs, gwo_runs, opt_runs, dataset_name, plots_dir)

    # Fig 2: 3-way external metrics
    plot_metric_comparison_bars_3way(base_summ, gwo_summ, de_summ, dataset_name, plots_dir)

    # Fig 3: 3-way internal metrics
    plot_internal_metrics_3way(base_summ, gwo_summ, de_summ, dataset_name, plots_dir)

    # Fig 4: 3-way radar chart
    plot_radar_chart_3way(base_summ, gwo_summ, de_summ, dataset_name, plots_dir)

    # Fig 5: 3-way time comparison
    plot_time_comparison_3way(baseline_runs, gwo_runs, opt_runs, dataset_name, plots_dir)

    # Fig 6: Phase contribution waterfall
    plot_phase_contribution(base_summ, gwo_summ, de_summ, dataset_name, plots_dir)

    # Fig 7: GWO convergence curve
    if convergence_data:
        plot_gwo_convergence(convergence_data, dataset_name, plots_dir)

    # Fig 8: Levy step distribution
    if convergence_data:
        plot_levy_step_distribution(convergence_data, dataset_name, plots_dir)

    # Fig 9: 3-way LaTeX table
    latex_3way = generate_latex_table_3way(dataset_name, base_summ, gwo_summ, de_summ)
    latex_path = os.path.join(output_folder, f"{dataset_name}_3way_table.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_3way)
    print(f"  Saved LaTeX: {latex_path}")

    n_figs = FIGURE_COUNTER["n"] - start_count
    print(f"  Total 3-way figures generated: {n_figs}")
    return n_figs
