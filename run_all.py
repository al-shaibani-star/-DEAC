
import json
import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    SEEDS, SUBSAMPLE,
    W_MOD, W_SIL, W_BAL, SIL_EVERY, SIL_SAMPLE,
    METRIC, GRAPH_MODE,
    BASELINE_K, BASELINE_DR_DIM, BASELINE_PRUNE_Q, BASELINE_RESOLUTION,
    OUTPUT_BASE,
    AMR_DE_MAXITER, AMR_DE_POPSIZE, AMR_N_RESTARTS,
)

# ---- Dataset Configuration (14 datasets) ----
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

DATASETS = [
    # --- Original 9 datasets ---
    {"name": "MFeat",        "path": os.path.join(DATA_DIR, "mfeat_full.csv"),    "max_n": 0},
    {"name": "USPS",         "path": os.path.join(DATA_DIR, "USPS.csv"),           "max_n": 0},
    {"name": "Optdigits",    "path": os.path.join(DATA_DIR, "Optdigits.csv"),     "max_n": 0},
    {"name": "CNAE9",        "path": os.path.join(DATA_DIR, "CNAE9.csv"),         "max_n": 0},
    {"name": "COIL20",       "path": os.path.join(DATA_DIR, "COIL20.csv"),        "max_n": 0},
    {"name": "Segmentation", "path": os.path.join(DATA_DIR, "Segmentation.csv"),  "max_n": 0},
    {"name": "ISOLET",       "path": os.path.join(DATA_DIR, "isolet_full.csv"),   "max_n": 0},
    {"name": "Olivetti",     "path": os.path.join(DATA_DIR, "olivetti_full.csv"), "max_n": 0},
    {"name": "HAR",          "path": os.path.join(DATA_DIR, "har_full.csv"),      "max_n": 0},
    # --- New datasets (for statistical power) ---
    {"name": "PenDigits",    "path": os.path.join(DATA_DIR, "PenDigits.csv"),     "max_n": 0},
    {"name": "Letter",       "path": os.path.join(DATA_DIR, "Letter.csv"),        "max_n": 10000},
    {"name": "Satimage",     "path": os.path.join(DATA_DIR, "Satimage.csv"),      "max_n": 0},
    {"name": "Wine",         "path": os.path.join(DATA_DIR, "Wine.csv"),          "max_n": 0},
    {"name": "Glass",        "path": os.path.join(DATA_DIR, "Glass.csv"),         "max_n": 0},
]


# ============================================================
# Cache System
# ============================================================
CACHE_DIR = os.path.join(OUTPUT_BASE, ".cache")


def _cache_path(ds_name: str) -> str:
    """Get cache file path for a dataset."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{ds_name}.json")


def _save_cache(ds_name: str, result: dict):
    """Save dataset result to cache (JSON-safe only)."""
    cache = {}
    for k, v in result.items():
        if k in ("base_summ", "amr_summ", "de_summ", "gwo_summ", "bench_summ"):
            cache[k] = v
        elif isinstance(v, (int, float, str, bool)):
            cache[k] = v
    cache["_completed"] = True
    with open(_cache_path(ds_name), "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"  [CACHE] Saved: {ds_name}")


def _load_cache(ds_name: str):
    """Load cached result if exists and complete. Returns dict or None."""
    path = _cache_path(ds_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("_completed"):
            return cache
    except Exception:
        pass
    return None


def run_single_dataset(dataset_cfg: dict) -> dict:
    """Run one dataset through the AMR-DE pipeline with QualityGate."""
    from src.io.data_loading import load_csv
    from src.core.pipeline import run_pipeline_once
    from src.core.fast_amr_de import run_fast_amr_de as run_amr_de
    from src.evaluation.guard import apply_quality_gate
    from src.evaluation.scoring import RunResult
    from src.visualization.reporting import (
        summarize_results, print_summary_table,
        print_academic_summary, generate_latex_table,
        generate_all_figures, reset_figure_counter,
        save_latex_table, compute_complexity_table, generate_complexity_latex,
        generate_all_figures_3way,
    )

    ds_name = dataset_cfg["name"]
    ds_path = dataset_cfg["path"]
    max_n = dataset_cfg.get("max_n", 0)

    output_folder = os.path.join(OUTPUT_BASE, ds_name)
    os.makedirs(output_folder, exist_ok=True)

    # Load data
    X, y, file_name = load_csv(ds_path, label_col=-1)

    # Subsample if needed
    if max_n > 0 and X.shape[0] > max_n:
        from sklearn.model_selection import train_test_split
        n_orig = X.shape[0]
        if y is not None:
            _, X, _, y = train_test_split(X, y, test_size=max_n, stratify=y, random_state=42)
        else:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_orig, size=max_n, replace=False)
            X = X[idx]
        print(f"  [max-n] Subsampled {n_orig} -> {X.shape[0]}")

    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])

    # Adaptive baseline parameters
    adaptive_dr_dim = max(2, min(BASELINE_DR_DIM, n_features - 1))
    adaptive_k = BASELINE_K
    if n_samples < 500:
        adaptive_k = min(BASELINE_K, max(5, n_samples // 10))
    adaptive_k = min(adaptive_k, n_samples - 1)
    adaptive_subsample = min(SUBSAMPLE, n_samples)

    print(f"  n={n_samples} | d={n_features} | labels={'YES' if y is not None else 'NO'}")
    print(f"  Baseline(UMAP): k={adaptive_k} d={adaptive_dr_dim} q={BASELINE_PRUNE_Q} r={BASELINE_RESOLUTION}")
    if adaptive_dr_dim != BASELINE_DR_DIM or adaptive_k != BASELINE_K:
        print(f"  [ADAPTIVE] d': {BASELINE_DR_DIM}->{adaptive_dr_dim} | k: {BASELINE_K}->{adaptive_k}")
    print(f"  AMR-DE: maxiter={AMR_DE_MAXITER} popsize={AMR_DE_POPSIZE} restarts={AMR_N_RESTARTS} subsample={adaptive_subsample}")

    # Args object for complexity table
    class Args:
        pass
    args = Args()
    args.k = adaptive_k
    args.dr_dim = adaptive_dr_dim
    args.de_maxiter = AMR_DE_MAXITER
    args.de_popsize = AMR_DE_POPSIZE
    args.subsample = SUBSAMPLE
    args.sil_sample = SIL_SAMPLE
    args.sil_every = SIL_EVERY
    args.w_mod = W_MOD
    args.w_sil = W_SIL
    args.w_bal = W_BAL
    args.metric = METRIC
    args.graph_mode = GRAPH_MODE

    baseline_runs = []
    amr_runs = []           # AMR-DE hybrid results
    convergence_data = []   # placeholder for compatibility
    all_bench_fair = []
    all_bench_upper = []

    for s in range(1, SEEDS + 1):
        seed = 1000 + s
        print(f"  Seed {seed}...", end=" ", flush=True)

        # -- Baseline (UMAP) --
        b = run_pipeline_once(
            X=X, y=y, seed=seed, metric=METRIC,
            k=adaptive_k, dr_dim=adaptive_dr_dim,
            prune_q=BASELINE_PRUNE_Q, resolution=BASELINE_RESOLUTION,
            w_mod=W_MOD, w_sil=W_SIL, w_bal=W_BAL,
            graph_mode=GRAPH_MODE, use_umap=True,
        )
        baseline_runs.append(b)

        # -- AMR-DE Optimization (8D search) --
        amr_result = run_amr_de(
            X=X, y=y, seed=seed, metric=METRIC,
            subsample=adaptive_subsample,
            de_maxiter=AMR_DE_MAXITER, de_popsize=AMR_DE_POPSIZE,
            n_restarts=AMR_N_RESTARTS,
            sil_every=SIL_EVERY, sil_sample=SIL_SAMPLE,
            graph_mode=GRAPH_MODE, min_deg_guard=3, verbose=0,
            max_dr_dim=n_features - 1, max_k=n_samples - 1,
        )

        # -- Evaluate AMR-DE result using the selected representation --
        o = run_pipeline_once(
            X=X, y=y, seed=seed, metric=METRIC,
            k=amr_result.best_k, dr_dim=amr_result.best_d,
            prune_q=amr_result.best_q, resolution=amr_result.best_r,
            w_mod=W_MOD, w_sil=W_SIL, w_bal=W_BAL,
            graph_mode=GRAPH_MODE,
            rep_type=amr_result.best_rep,
            n_neighbors=amr_result.best_n_neighbors,
            min_dist=amr_result.best_min_dist,
            perplexity=amr_result.best_perplexity,
        )
        o.mode = "optimized"
        o.time_opt_sec = float(amr_result.opt_time)

        # -- QualityGate: Hybrid >= Baseline always --
        use_baseline = apply_quality_gate(b, o, n_samples=n_samples)

        if use_baseline:
            o = RunResult(
                seed=b.seed, mode="optimized(=base)",
                k=b.k, dr_dim=b.dr_dim, prune_q=b.prune_q, resolution=b.resolution,
                C=b.C, edges=b.edges,
                n_components=b.n_components, comp_min=b.comp_min,
                comp_med=b.comp_med, comp_max=b.comp_max,
                degenerate=b.degenerate,
                ARI=b.ARI, NMI=b.NMI, ACC=b.ACC,
                SIL=b.SIL, MOD=b.MOD,
                bal01=b.bal01, n_singletons=b.n_singletons, n_tiny=b.n_tiny,
                internal_score=b.internal_score, internal_pen=b.internal_pen,
                time_total_sec=b.time_total_sec, time_opt_sec=float(amr_result.opt_time),
            )

        amr_runs.append(o)
        qg_tag = " [QG]" if use_baseline else ""
        best_rep = amr_result.best_rep
        print(f"BASE ARI={b.ARI:.3f} | AMR-DE ARI={o.ARI:.3f} | "
              f"rep={best_rep} | k={amr_result.best_k} d={amr_result.best_d} "
              f"q={amr_result.best_q:.2f} r={amr_result.best_r:.2f}{qg_tag}")

        # -- Benchmarks (Fair: same k as Hybrid) --
        from src.evaluation.benchmarks import run_benchmarks_fair, run_benchmarks_upper
        hybrid_k = o.C
        bench_fair = run_benchmarks_fair(X, y, seed=seed, dr_dim=adaptive_dr_dim, metric=METRIC, hybrid_k=hybrid_k)
        all_bench_fair.append(bench_fair)
        bench_str = " | ".join([f"{r.method}={r.ARI:.3f}" for r in bench_fair])
        print(f"    Fair(k={hybrid_k}): {bench_str}")

        # -- Benchmarks (Upper Bound: true k) --
        bench_upper = run_benchmarks_upper(X, y, seed=seed, dr_dim=adaptive_dr_dim, metric=METRIC)
        all_bench_upper.append(bench_upper)
        bench_str_ub = " | ".join([f"{r.method}={r.ARI:.3f}" for r in bench_upper])
        print(f"    Upper(true_k): {bench_str_ub}")

        # Placeholder convergence data for compatibility with plotting
        convergence_data.append(([], []))

    # -- Summary --
    print()
    base_summ = summarize_results(baseline_runs)
    amr_summ = summarize_results(amr_runs)
    print_summary_table(f"  BASELINE-UMAP ({ds_name})", base_summ)
    print()
    print_summary_table(f"  AMR-DE HYBRID ({ds_name})", amr_summ)

    # -- Benchmarks Summary (Fair + Upper Bound) --
    from src.evaluation.benchmarks import (
        summarize_benchmarks, print_benchmark_table,
        generate_benchmark_latex, plot_benchmark_comparison,
    )
    bench_fair_summ = summarize_benchmarks(all_bench_fair)
    bench_upper_summ = summarize_benchmarks(all_bench_upper)
    print_benchmark_table(bench_fair_summ, ds_name, mode="fair")
    print_benchmark_table(bench_upper_summ, ds_name, mode="upper_bound")

    # Academic output
    print()
    print_academic_summary(
        dataset_name=ds_name, n_samples=n_samples, n_features=n_features,
        n_seeds=SEEDS, base_summ=base_summ, de_summ=amr_summ, args=args,
    )

    # -- Generate figures --
    # Use 3-way figures with amr_runs serving as both gwo and opt for compatibility
    reset_figure_counter()
    try:
        n_figs = generate_all_figures_3way(
            dataset_name=ds_name,
            baseline_runs=baseline_runs,
            gwo_runs=amr_runs,
            opt_runs=amr_runs,
            base_summ=base_summ,
            gwo_summ=amr_summ,
            de_summ=amr_summ,
            output_folder=output_folder,
            convergence_data=convergence_data,
        )
    except Exception as e:
        print(f"  [WARN] 3-way figures failed: {e}")
        n_figs = 0

    # -- Benchmark figures + LaTeX --
    print(f"\n  Generating benchmark comparison figures for {ds_name}...")
    try:
        plot_benchmark_comparison(ds_name, bench_fair_summ, bench_upper_summ,
                                  base_summ, amr_summ, amr_summ, output_folder)
        n_figs += 3
    except Exception as e:
        print(f"  [WARN] Benchmark plots failed: {e}")

    # -- Academic Plots --
    try:
        from src.visualization.academic_plots import generate_academic_plots
        n_academic = generate_academic_plots(
            X, y, baseline_runs, amr_runs, amr_runs,
            base_summ, amr_summ, amr_summ,
            convergence_data, ds_name, output_folder,
        )
        n_figs += n_academic
    except Exception as e:
        print(f"  [WARN] Academic plots failed: {e}")

    # Save LaTeX
    latex_str = generate_latex_table(ds_name, base_summ, amr_summ)
    latex_str += "\n\n" + generate_benchmark_latex(ds_name, bench_fair_summ, bench_upper_summ,
                                                    base_summ, amr_summ, amr_summ)
    complexity_rows = compute_complexity_table(n_samples, n_features, args)
    latex_str += "\n\n" + generate_complexity_latex(complexity_rows, ds_name)
    latex_path = os.path.join(output_folder, f"{ds_name}_tables.tex")
    save_latex_table(latex_str, latex_path)
    print(f"  Saved LaTeX: {latex_path}")

    return {
        "name": ds_name, "n": n_samples, "d": n_features,
        "base_ari": base_summ["ARI"]["mean"],
        "amr_ari": amr_summ["ARI"]["mean"],
        "base_nmi": base_summ["NMI"]["mean"],
        "amr_nmi": amr_summ["NMI"]["mean"],
        # Backwards-compatible keys for cross-dataset plotting functions
        "de_ari": amr_summ["ARI"]["mean"],
        "gwo_ari": amr_summ["ARI"]["mean"],
        "de_nmi": amr_summ["NMI"]["mean"],
        "gwo_nmi": amr_summ["NMI"]["mean"],
        "n_figs": n_figs,
        "qg_used": any("=base" in r.mode for r in amr_runs),
        "base_summ": base_summ,
        "amr_summ": amr_summ,
        # Backwards-compatible keys
        "de_summ": amr_summ,
        "gwo_summ": amr_summ,
        "bench_fair_summ": bench_fair_summ, "bench_upper_summ": bench_upper_summ,
    }


def main():
    print("=" * 90)
    print("AMR-GraphClust: Adaptive Multi-Representation Graph Clustering")
    print("=" * 90)
    print(f"Datasets: {len(DATASETS)} | Seeds: {SEEDS}")
    print(f"Optimizer: Fast AMR-DE (maxiter={AMR_DE_MAXITER}, pop={AMR_DE_POPSIZE}, restarts={AMR_N_RESTARTS})")
    print(f"QualityGate: Hybrid >= Baseline (UMAP) always")
    print(f"Weights: W_MOD={W_MOD}, W_SIL={W_SIL}, W_BAL={W_BAL}")
    print(f"Output: {OUTPUT_BASE}/")
    print("=" * 90)

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    all_results = []
    total_figs = 0
    t_start = time.time()

    for i, ds in enumerate(DATASETS, 1):
        print(f"\n{'=' * 90}")
        print(f"[{i}/{len(DATASETS)}] Dataset: {ds['name']}")
        print(f"{'=' * 90}")

        # Check cache first
        cached = _load_cache(ds['name'])
        if cached:
            print(f"  [CACHE] Found cached result -- skipping computation")
            all_results.append(cached)
            total_figs += cached.get("n_figs", 0)
            print(f"\n  >> {ds['name']}: Base={cached['base_ari']:.3f} | "
                  f"AMR-DE={cached['amr_ari']:.3f} | [CACHED]")
            continue

        try:
            result = run_single_dataset(ds)
            all_results.append(result)
            total_figs += result["n_figs"]
            # Save to cache
            _save_cache(ds['name'], result)
            print(f"\n  >> {ds['name']}: Base={result['base_ari']:.3f} | "
                  f"AMR-DE={result['amr_ari']:.3f} | "
                  f"Figures: {result['n_figs']}")
        except Exception as e:
            print(f"\n  >> ERROR on {ds['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Final Summary
    total_time = time.time() - t_start
    print(f"\n{'=' * 90}")
    print("FINAL SUMMARY -- AMR-GraphClust Results")
    print(f"{'=' * 90}")
    print(f"{'Dataset':<14}| {'n':>7} | {'Base':>7} | {'AMR-DE':>7} | "
          f"{'KMeans':>7} | {'Spect':>7} | {'Agg':>7} | {'GMM':>7} | {'HDBSC':>7}")
    print("-" * 100)
    for r in all_results:
        bs = r.get("bench_fair_summ", r.get("bench_summ", {}))
        km = bs.get("KMeans", {}).get("ARI_mean", 0)
        sp = bs.get("Spectral", {}).get("ARI_mean", 0)
        ag = bs.get("Agglomerative", {}).get("ARI_mean", 0)
        gm = bs.get("GMM", {}).get("ARI_mean", 0)
        hd = bs.get("HDBSCAN", {}).get("ARI_mean", 0)
        amr_ari = r.get("amr_ari", r.get("de_ari", 0))
        print(f"{r['name']:<14}| {r['n']:>7} | {r['base_ari']:>7.3f} | {amr_ari:>7.3f} | "
              f"{km:>7.3f} | {sp:>7.3f} | {ag:>7.3f} | {gm:>7.3f} | {hd:>7.3f}")
    print("-" * 100)

    # Cross-Dataset Comparison Figures
    if len(all_results) >= 2:
        try:
            from src.visualization.reporting import generate_all_cross_dataset_figures
            n_cross = generate_all_cross_dataset_figures(all_results, OUTPUT_BASE)
            total_figs += n_cross
        except Exception as e:
            print(f"  [WARN] Cross-dataset figures failed: {e}")

        try:
            from src.visualization.academic_plots import generate_cross_academic_plots
            n_academic_cross = generate_cross_academic_plots(all_results, OUTPUT_BASE)
            total_figs += n_academic_cross
        except Exception as e:
            print(f"  [WARN] Cross-academic plots failed: {e}")

    print(f"\nTotal figures: {total_figs} | Total time: {total_time:.1f}s")
    print(f"Output saved in: {os.path.abspath(OUTPUT_BASE)}/")
    print("=" * 90)


if __name__ == "__main__":
    main()
