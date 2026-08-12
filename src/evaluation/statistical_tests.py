# -*- coding: utf-8 -*-
"""
Statistical significance tests for clustering comparison.

Implements:
  1. Wilcoxon Signed-Rank Test (pairwise)
  2. Friedman Test + Nemenyi post-hoc (multi-method)
  3. Critical Difference Diagram
"""
import numpy as np
from typing import Dict, List, Tuple


def wilcoxon_test(scores_a: List[float], scores_b: List[float],
                   name_a: str = "A", name_b: str = "B") -> dict:
    """Wilcoxon signed-rank test between two methods.

    H0: No significant difference between methods.
    """
    from scipy.stats import wilcoxon

    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = a - b

    # Remove zeros (ties)
    nonzero = diff[diff != 0]
    if len(nonzero) < 3:
        return {
            "test": "Wilcoxon",
            "method_a": name_a, "method_b": name_b,
            "statistic": float("nan"), "p_value": 1.0,
            "significant_005": False, "significant_001": False,
            "winner": "tie", "mean_diff": float(np.mean(diff)),
            "note": "Too few non-zero differences for Wilcoxon test",
        }

    try:
        stat, p = wilcoxon(a, b, alternative="two-sided")
        winner = name_a if np.mean(diff) > 0 else name_b
        return {
            "test": "Wilcoxon",
            "method_a": name_a, "method_b": name_b,
            "statistic": float(stat), "p_value": float(p),
            "significant_005": p < 0.05,
            "significant_001": p < 0.01,
            "winner": winner if p < 0.05 else "no_significant_diff",
            "mean_diff": float(np.mean(diff)),
        }
    except Exception as e:
        return {
            "test": "Wilcoxon", "method_a": name_a, "method_b": name_b,
            "statistic": float("nan"), "p_value": 1.0,
            "significant_005": False, "significant_001": False,
            "winner": "error", "mean_diff": float(np.mean(diff)),
            "note": str(e),
        }


def friedman_test(all_scores: Dict[str, List[float]]) -> dict:
    """Friedman test for comparing multiple methods.

    H0: All methods perform equally.
    """
    from scipy.stats import friedmanchisquare

    methods = list(all_scores.keys())
    n_methods = len(methods)
    n_datasets = len(all_scores[methods[0]])

    if n_methods < 3:
        return {"test": "Friedman", "note": "Need >= 3 methods"}

    # Build score matrix
    arrays = [np.array(all_scores[m]) for m in methods]

    try:
        stat, p = friedmanchisquare(*arrays)

        # Compute average ranks
        score_matrix = np.column_stack(arrays)
        ranks = np.zeros_like(score_matrix)
        for i in range(n_datasets):
            row = score_matrix[i]
            # Higher ARI = better = lower rank
            order = np.argsort(-row)
            for rank_val, idx in enumerate(order):
                ranks[i, idx] = rank_val + 1
        avg_ranks = ranks.mean(axis=0)

        rank_dict = {methods[i]: float(avg_ranks[i]) for i in range(n_methods)}

        return {
            "test": "Friedman",
            "statistic": float(stat), "p_value": float(p),
            "significant_005": p < 0.05,
            "n_methods": n_methods, "n_datasets": n_datasets,
            "avg_ranks": rank_dict,
            "best_method": min(rank_dict, key=rank_dict.get),
        }
    except Exception as e:
        return {"test": "Friedman", "note": str(e)}


def nemenyi_cd(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Compute Nemenyi critical difference.

    CD = q_alpha * sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    """
    # q_alpha values for alpha=0.05 (from Demsar 2006, Table 5)
    q_values = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q = q_values.get(n_methods, 3.0)
    cd = q * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets))
    return float(cd)


def plot_critical_difference(avg_ranks: Dict[str, float], n_datasets: int,
                              output_path: str, alpha: float = 0.05):
    """Generate Critical Difference diagram (Demsar 2006)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = list(avg_ranks.keys())
    ranks = [avg_ranks[m] for m in methods]
    n_methods = len(methods)

    # Sort by rank
    sorted_idx = np.argsort(ranks)
    methods = [methods[i] for i in sorted_idx]
    ranks = [ranks[i] for i in sorted_idx]

    cd = nemenyi_cd(n_methods, n_datasets, alpha)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(0, 1.5)

    # Draw axis
    ax.hlines(1.0, 0.5, n_methods + 0.5, colors='black', linewidth=1)
    for i in range(1, n_methods + 1):
        ax.vlines(i, 0.95, 1.05, colors='black', linewidth=1)
        ax.text(i, 0.85, str(i), ha='center', fontsize=9)

    # CD bar
    ax.hlines(1.3, 1, 1 + cd, colors='red', linewidth=2)
    ax.text(1 + cd / 2, 1.35, f"CD={cd:.2f}", ha='center', fontsize=8, color='red')

    # Place methods
    top_methods = methods[:len(methods) // 2]
    bottom_methods = methods[len(methods) // 2:]

    for i, (m, r) in enumerate(zip(methods, ranks)):
        y = 0.65 if i < len(methods) // 2 else 0.35
        ax.plot(r, 1.0, 'ko', markersize=6)
        ax.vlines(r, y + 0.05, 1.0, colors='gray', linewidth=0.5, linestyles='dashed')
        ax.text(r, y, m, ha='center', fontsize=8, fontweight='bold')

    # Connect methods not significantly different
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            if abs(ranks[i] - ranks[j]) < cd:
                y_line = 1.05 + 0.03 * (j - i)
                ax.hlines(y_line, ranks[i], ranks[j], colors='blue',
                          linewidth=2, alpha=0.5)

    ax.set_title(f"Critical Difference Diagram (Nemenyi, \u03B1={alpha})",
                 fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_all_statistical_tests(results: List[dict], output_dir: str) -> dict:
    """Run all statistical tests on experiment results.

    Parameters
    ----------
    results : list of dict
        Each dict has 'name', 'base_ari_list', 'amr_ari_list',
        and benchmark ARI lists.
    output_dir : str
        Where to save plots.

    Returns
    -------
    dict
        All test results.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Collect per-dataset mean ARIs for each method
    datasets = [r['name'] for r in results]
    n_ds = len(datasets)

    method_scores = {
        "AMR-DE": [r.get('amr_ari', 0) for r in results],
        "Baseline": [r.get('base_ari', 0) for r in results],
    }

    # Add benchmarks if available
    for bench in ["KMeans", "Spectral", "Agglomerative", "GMM", "HDBSCAN"]:
        key = f"bench_fair_{bench}_ari"
        if all(key in r for r in results):
            method_scores[bench] = [r[key] for r in results]

    # 1. Wilcoxon: AMR-DE vs each method
    wilcoxon_results = []
    amr_scores = method_scores["AMR-DE"]
    for method, scores in method_scores.items():
        if method == "AMR-DE":
            continue
        w = wilcoxon_test(amr_scores, scores, "AMR-DE", method)
        wilcoxon_results.append(w)
        p_str = f"p={w['p_value']:.4f}" if not np.isnan(w['p_value']) else "N/A"
        sig = "*" if w.get('significant_005') else ""
        print(f"  Wilcoxon AMR-DE vs {method}: {p_str} {sig} "
              f"(winner={w['winner']}, mean_diff={w['mean_diff']:+.3f})")

    # 2. Friedman test (if >= 3 methods)
    friedman_result = None
    if len(method_scores) >= 3:
        friedman_result = friedman_test(method_scores)
        if 'p_value' in friedman_result:
            print(f"\n  Friedman test: chi2={friedman_result['statistic']:.3f}, "
                  f"p={friedman_result['p_value']:.4f}")
            if friedman_result.get('significant_005'):
                print("  -> Significant difference exists among methods (p<0.05)")
            print(f"  Average ranks: {friedman_result.get('avg_ranks', {})}")
            print(f"  Best method: {friedman_result.get('best_method', '?')}")

            # 3. Critical Difference Diagram
            if friedman_result.get('avg_ranks'):
                cd_path = os.path.join(output_dir, "critical_difference_diagram.png")
                plot_critical_difference(
                    friedman_result['avg_ranks'], n_ds, cd_path)
                print(f"  Saved: {cd_path}")

    return {
        "wilcoxon": wilcoxon_results,
        "friedman": friedman_result,
        "method_scores": method_scores,
        "datasets": datasets,
    }


def generate_stats_latex(wilcoxon_results: List[dict], friedman_result: dict,
                          output_path: str):
    """Generate LaTeX table for statistical tests."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical significance tests: AMR-DE vs. benchmark methods.}",
        r"\label{tab:statistical_tests}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Comparison & Statistic & $p$-value & Significant & Winner \\",
        r"\midrule",
    ]
    for w in wilcoxon_results:
        method = w['method_b']
        stat = f"{w['statistic']:.1f}" if not np.isnan(w['statistic']) else "N/A"
        p = f"{w['p_value']:.4f}" if not np.isnan(w['p_value']) else "N/A"
        sig = r"$\checkmark$" if w.get('significant_005') else "---"
        winner = w.get('winner', '---')
        lines.append(f"AMR-DE vs. {method} & {stat} & {p} & {sig} & {winner} \\\\")

    if friedman_result and 'p_value' in friedman_result:
        lines.append(r"\midrule")
        stat = f"{friedman_result['statistic']:.2f}"
        p = f"{friedman_result['p_value']:.4f}"
        sig = r"$\checkmark$" if friedman_result.get('significant_005') else "---"
        best = friedman_result.get('best_method', '---')
        lines.append(f"Friedman (all) & {stat} & {p} & {sig} & {best} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
