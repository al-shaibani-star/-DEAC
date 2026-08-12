# -*- coding: utf-8 -*-
"""
Benchmark clustering methods for comparison.

Two comparison modes:
  Table 1 (Fair):        All methods use k discovered by Hybrid (same k for all)
  Table 2 (Upper Bound): Benchmarks use true_k (advantage — for reference only)

Methods: KMeans, Spectral, Agglomerative, GMM, HDBSCAN
"""
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.io.representation import make_representation
from src.io.utils import best_map_acc


@dataclass
class BenchmarkResult:
    """Result from a single benchmark method."""
    method: str
    ARI: float
    NMI: float
    ACC: float
    SIL: float
    C: int
    time_sec: float
    mode: str = "fair"  # "fair" or "upper_bound"


def _safe_silhouette(Z, labels, metric="cosine") -> float:
    from sklearn.metrics import silhouette_score
    C = len(np.unique(labels))
    if C < 2 or C >= len(labels):
        return float("nan")
    try:
        return float(silhouette_score(Z, labels, metric=metric))
    except:
        return float("nan")


def _evaluate_labels(Z, y_true, y_pred, metric="cosine") -> Tuple[float, float, float, float, int]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    C = int(len(np.unique(y_pred)))
    if C < 2:
        return 0.0, 0.0, 0.0, float("nan"), C
    if y_true is not None:
        ARI = float(adjusted_rand_score(y_true, y_pred))
        NMI = float(normalized_mutual_info_score(y_true, y_pred))
        ACC = float(best_map_acc(y_true, y_pred))
    else:
        ARI = NMI = ACC = float("nan")
    SIL = _safe_silhouette(Z, y_pred, metric=metric)
    return ARI, NMI, ACC, SIL, C


def run_kmeans(Z, y_true, n_clusters, seed, metric="cosine") -> BenchmarkResult:
    from sklearn.cluster import KMeans
    t0 = time.time()
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=300)
    labels = km.fit_predict(Z)
    t = time.time() - t0
    ARI, NMI, ACC, SIL, C = _evaluate_labels(Z, y_true, labels, metric)
    return BenchmarkResult("KMeans", ARI, NMI, ACC, SIL, C, t)


def run_spectral(Z, y_true, n_clusters, seed, metric="cosine") -> BenchmarkResult:
    from sklearn.cluster import SpectralClustering
    t0 = time.time()
    n = Z.shape[0]
    if n > 5000:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=5000, replace=False)
        Z_sub, y_sub = Z[idx], y_true[idx] if y_true is not None else None
    else:
        Z_sub, y_sub = Z, y_true
    try:
        sc = SpectralClustering(n_clusters=n_clusters, random_state=seed,
                                affinity="nearest_neighbors", n_neighbors=10,
                                assign_labels="kmeans", n_init=5)
        labels = sc.fit_predict(Z_sub)
        t = time.time() - t0
        ARI, NMI, ACC, SIL, C = _evaluate_labels(Z_sub, y_sub, labels, metric)
    except:
        t = time.time() - t0
        ARI = NMI = ACC = SIL = 0.0
        C = 0
    return BenchmarkResult("Spectral", ARI, NMI, ACC, SIL, C, t)


def run_agglomerative(Z, y_true, n_clusters, metric="cosine") -> BenchmarkResult:
    from sklearn.cluster import AgglomerativeClustering
    t0 = time.time()
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = agg.fit_predict(Z)
    t = time.time() - t0
    ARI, NMI, ACC, SIL, C = _evaluate_labels(Z, y_true, labels, metric)
    return BenchmarkResult("Agglomerative", ARI, NMI, ACC, SIL, C, t)


def run_gmm(Z, y_true, n_clusters, seed, metric="cosine") -> BenchmarkResult:
    from sklearn.mixture import GaussianMixture
    t0 = time.time()
    try:
        gmm = GaussianMixture(n_components=n_clusters, random_state=seed,
                               covariance_type="full", max_iter=200, n_init=3)
        labels = gmm.fit_predict(Z)
        t = time.time() - t0
        ARI, NMI, ACC, SIL, C = _evaluate_labels(Z, y_true, labels, metric)
    except:
        t = time.time() - t0
        ARI = NMI = ACC = SIL = 0.0
        C = 0
    return BenchmarkResult("GMM", ARI, NMI, ACC, SIL, C, t)


def run_hdbscan(Z, y_true, seed, metric="cosine") -> BenchmarkResult:
    t0 = time.time()
    try:
        from sklearn.cluster import HDBSCAN as HDBSCAN_cls
        hdb = HDBSCAN_cls(min_cluster_size=15, min_samples=5, metric="euclidean")
        labels = hdb.fit_predict(Z)
        noise_mask = (labels == -1)
        if noise_mask.all():
            t = time.time() - t0
            return BenchmarkResult("HDBSCAN", 0.0, 0.0, 0.0, float("nan"), 0, t)
        if noise_mask.any():
            from sklearn.neighbors import NearestNeighbors
            good_idx = np.where(~noise_mask)[0]
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(Z[good_idx])
            _, nn_idx = nn.kneighbors(Z[noise_mask])
            labels[noise_mask] = labels[good_idx[nn_idx.ravel()]]
        t = time.time() - t0
        ARI, NMI, ACC, SIL, C = _evaluate_labels(Z, y_true, labels, metric)
    except:
        t = time.time() - t0
        ARI = NMI = ACC = SIL = 0.0
        C = 0
    return BenchmarkResult("HDBSCAN", ARI, NMI, ACC, SIL, C, t)


def run_benchmarks_fair(X, y, seed, dr_dim, metric, hybrid_k):
    """Table 1: FAIR comparison — all methods use k discovered by Hybrid."""
    Z = make_representation(X, seed=seed, dr_dim=dr_dim)
    k = max(2, int(hybrid_k))
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results.append(run_kmeans(Z, y, k, seed, metric))
        results.append(run_spectral(Z, y, k, seed, metric))
        results.append(run_agglomerative(Z, y, k, metric))
        results.append(run_gmm(Z, y, k, seed, metric))
        results.append(run_hdbscan(Z, y, seed, metric))
    for r in results:
        r.mode = "fair"
    return results


def run_benchmarks_upper(X, y, seed, dr_dim, metric):
    """Table 2: UPPER BOUND — benchmarks use true_k (unfair advantage)."""
    Z = make_representation(X, seed=seed, dr_dim=dr_dim)
    true_k = int(len(np.unique(y))) if y is not None else 10
    true_k = max(2, true_k)
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results.append(run_kmeans(Z, y, true_k, seed, metric))
        results.append(run_spectral(Z, y, true_k, seed, metric))
        results.append(run_agglomerative(Z, y, true_k, metric))
        results.append(run_gmm(Z, y, true_k, seed, metric))
        results.append(run_hdbscan(Z, y, seed, metric))
    for r in results:
        r.mode = "upper_bound"
    return results


def summarize_benchmarks(all_seed_results: List[List[BenchmarkResult]]) -> Dict[str, Dict[str, float]]:
    """Summarize benchmark results across seeds."""
    if not all_seed_results or not all_seed_results[0]:
        return {}
    methods = [r.method for r in all_seed_results[0]]
    summary = {}
    for m_idx, method in enumerate(methods):
        aris = [seed_res[m_idx].ARI for seed_res in all_seed_results]
        nmis = [seed_res[m_idx].NMI for seed_res in all_seed_results]
        accs = [seed_res[m_idx].ACC for seed_res in all_seed_results]
        sils = [seed_res[m_idx].SIL for seed_res in all_seed_results]
        cs = [seed_res[m_idx].C for seed_res in all_seed_results]
        times = [seed_res[m_idx].time_sec for seed_res in all_seed_results]
        summary[method] = {
            "ARI_mean": float(np.nanmean(aris)), "ARI_std": float(np.nanstd(aris)),
            "NMI_mean": float(np.nanmean(nmis)), "NMI_std": float(np.nanstd(nmis)),
            "ACC_mean": float(np.nanmean(accs)), "ACC_std": float(np.nanstd(accs)),
            "SIL_mean": float(np.nanmean(sils)), "SIL_std": float(np.nanstd(sils)),
            "C_mean": float(np.mean(cs)), "C_std": float(np.std(cs)),
            "time_mean": float(np.mean(times)), "time_std": float(np.std(times)),
        }
    return summary


def print_benchmark_table(bench_summary: Dict, ds_name: str, mode: str = "fair"):
    """Print benchmark comparison table."""
    label = "FAIR (same k as Hybrid)" if mode == "fair" else "UPPER BOUND (true k)"
    print(f"\n  BENCHMARKS — {label} ({ds_name})")
    print(f"  {'Method':<16}| {'ARI':>12} | {'NMI':>12} | {'ACC':>12} | {'SIL':>12} | {'C':>6} | {'Time':>8}")
    print("  " + "-" * 82)
    for method, s in bench_summary.items():
        ari_str = f"{s['ARI_mean']:.3f}\u00b1{s['ARI_std']:.3f}"
        nmi_str = f"{s['NMI_mean']:.3f}\u00b1{s['NMI_std']:.3f}"
        acc_str = f"{s['ACC_mean']:.3f}\u00b1{s['ACC_std']:.3f}"
        sil_str = f"{s['SIL_mean']:.3f}\u00b1{s['SIL_std']:.3f}" if np.isfinite(s['SIL_mean']) else "N/A"
        c_str = f"{s['C_mean']:.0f}"
        t_str = f"{s['time_mean']:.2f}s"
        print(f"  {method:<16}| {ari_str:>12} | {nmi_str:>12} | {acc_str:>12} | {sil_str:>12} | {c_str:>6} | {t_str:>8}")


def generate_benchmark_latex(ds_name, bench_fair, bench_upper, base_summ, gwo_summ, de_summ):
    """Generate LaTeX with TWO tables: Fair + Upper Bound."""
    lines = []

    # Table 1: Fair comparison
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering\small")
    lines.append(f"\\caption{{Fair comparison on \\textit{{{ds_name}}}: all methods use $k$ discovered by GWO--DE--NM.}}")
    lines.append(f"\\label{{tab:fair_{ds_name.lower()}}}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & ARI & NMI & ACC & $C$ \\")
    lines.append(r"\midrule")

    # Collect all ARI means to find best
    all_aris = {}
    for m, s in bench_fair.items():
        all_aris[m] = s['ARI_mean']
    all_aris["Baseline"] = base_summ["ARI"]["mean"]
    all_aris["GWO-only"] = gwo_summ["ARI"]["mean"]
    all_aris["GWO-DE-NM"] = de_summ["ARI"]["mean"]
    best_method = max(all_aris, key=all_aris.get)

    for method, s in bench_fair.items():
        bold = method == best_method
        ari = f"$\\mathbf{{{s['ARI_mean']:.3f} \\pm {s['ARI_std']:.3f}}}$" if bold else f"${s['ARI_mean']:.3f} \\pm {s['ARI_std']:.3f}$"
        nmi = f"${s['NMI_mean']:.3f} \\pm {s['NMI_std']:.3f}$"
        acc = f"${s['ACC_mean']:.3f} \\pm {s['ACC_std']:.3f}$"
        c = f"${s['C_mean']:.0f}$"
        lines.append(f"{method} & {ari} & {nmi} & {acc} & {c} \\\\")

    lines.append(r"\midrule")
    for label, summ, key in [("Baseline", base_summ, "Baseline"),
                              ("GWO-only", gwo_summ, "GWO-only"),
                              ("GWO--DE--NM", de_summ, "GWO-DE-NM")]:
        am, astd = summ["ARI"]["mean"], summ["ARI"]["std"]
        bold = key == best_method
        ari = f"$\\mathbf{{{am:.3f} \\pm {astd:.3f}}}$" if bold else f"${am:.3f} \\pm {astd:.3f}$"
        nmi = f"${summ['NMI']['mean']:.3f} \\pm {summ['NMI']['std']:.3f}$"
        acc = f"${summ['ACC']['mean']:.3f} \\pm {summ['ACC']['std']:.3f}$"
        c = f"${summ['C']['mean']:.1f}$"
        lines.append(f"{label} & {ari} & {nmi} & {acc} & {c} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # Table 2: Upper Bound
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering\small")
    lines.append(f"\\caption{{Upper bound comparison on \\textit{{{ds_name}}}: benchmarks use true $k$ (unfair advantage).}}")
    lines.append(f"\\label{{tab:upper_{ds_name.lower()}}}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & ARI & NMI & ACC & $C$ \\")
    lines.append(r"\midrule")

    for method, s in bench_upper.items():
        ari = f"${s['ARI_mean']:.3f} \\pm {s['ARI_std']:.3f}$"
        nmi = f"${s['NMI_mean']:.3f} \\pm {s['NMI_std']:.3f}$"
        acc = f"${s['ACC_mean']:.3f} \\pm {s['ACC_std']:.3f}$"
        c = f"${s['C_mean']:.0f}$"
        lines.append(f"{method} (true $k$) & {ari} & {nmi} & {acc} & {c} \\\\")

    lines.append(r"\midrule")
    am, astd = de_summ["ARI"]["mean"], de_summ["ARI"]["std"]
    ari = f"$\\mathbf{{{am:.3f} \\pm {astd:.3f}}}$"
    nmi = f"$\\mathbf{{{de_summ['NMI']['mean']:.3f} \\pm {de_summ['NMI']['std']:.3f}}}$"
    acc = f"$\\mathbf{{{de_summ['ACC']['mean']:.3f} \\pm {de_summ['ACC']['std']:.3f}}}$"
    c = f"${de_summ['C']['mean']:.1f}$"
    lines.append(f"GWO--DE--NM (auto $k$) & {ari} & {nmi} & {acc} & {c} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def plot_benchmark_comparison(ds_name, bench_fair, bench_upper,
                               base_summ, gwo_summ, de_summ, output_folder):
    """Generate benchmark comparison bar charts (Fair + Upper Bound)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(output_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── Fair comparison chart ──
    methods_fair = list(bench_fair.keys()) + ["Baseline", "GWO-only", "Hybrid"]
    aris_fair = [bench_fair[m]["ARI_mean"] for m in bench_fair]
    aris_fair += [base_summ["ARI"]["mean"], gwo_summ["ARI"]["mean"], de_summ["ARI"]["mean"]]
    stds_fair = [bench_fair[m]["ARI_std"] for m in bench_fair]
    stds_fair += [base_summ["ARI"]["std"], gwo_summ["ARI"]["std"], de_summ["ARI"]["std"]]

    n = len(methods_fair)
    colors = ["#95A5A6"] * len(bench_fair) + ["#3498DB", "#E67E22", "#2C3E50"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n)
    bars = ax.bar(x, aris_fair, yerr=stds_fair, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    best_idx = int(np.argmax(aris_fair))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods_fair, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("ARI", fontsize=12)
    ax.set_title(f"{ds_name} — Fair Comparison (all methods use same k)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, (v, s) in enumerate(zip(aris_fair, stds_fair)):
        if np.isfinite(v):
            ax.text(i, v + s + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"bench_fair_ari_{ds_name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: bench_fair_ari_{ds_name}.png")

    # ── Fair vs Upper Bound side-by-side ──
    methods_bench = list(bench_fair.keys())
    fair_aris = [bench_fair[m]["ARI_mean"] for m in methods_bench]
    upper_aris = [bench_upper[m]["ARI_mean"] for m in methods_bench]
    hybrid_ari = de_summ["ARI"]["mean"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(methods_bench))
    w = 0.35
    ax.bar(x - w/2, fair_aris, w, label="Fair (Hybrid k)", color="#3498DB", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, upper_aris, w, label="Upper Bound (true k)", color="#E74C3C", edgecolor="black", linewidth=0.5)
    ax.axhline(y=hybrid_ari, color="#2C3E50", linewidth=2, linestyle="--", label=f"Hybrid ARI={hybrid_ari:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_bench, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("ARI", fontsize=12)
    ax.set_title(f"{ds_name} — Fair vs Upper Bound Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for i, (f_v, u_v) in enumerate(zip(fair_aris, upper_aris)):
        ax.text(i - w/2, f_v + 0.01, f"{f_v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w/2, u_v + 0.01, f"{u_v:.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"bench_fair_vs_upper_{ds_name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: bench_fair_vs_upper_{ds_name}.png")

    # ── All metrics grouped (Fair) ──
    nmis_fair = [bench_fair[m]["NMI_mean"] for m in bench_fair]
    nmis_fair += [base_summ["NMI"]["mean"], gwo_summ["NMI"]["mean"], de_summ["NMI"]["mean"]]
    accs_fair = [bench_fair[m]["ACC_mean"] for m in bench_fair]
    accs_fair += [base_summ["ACC"]["mean"], gwo_summ["ACC"]["mean"], de_summ["ACC"]["mean"]]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(methods_fair))
    w = 0.25
    ax.bar(x - w, aris_fair, w, label="ARI", color="#3498DB", edgecolor="black", linewidth=0.3)
    ax.bar(x, nmis_fair, w, label="NMI", color="#2ECC71", edgecolor="black", linewidth=0.3)
    ax.bar(x + w, accs_fair, w, label="ACC", color="#E74C3C", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(methods_fair, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{ds_name} — All Metrics Fair Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"bench_all_metrics_{ds_name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: bench_all_metrics_{ds_name}.png")
