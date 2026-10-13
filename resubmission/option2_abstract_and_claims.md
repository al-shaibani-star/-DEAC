# Option 2 — Reframe around the negative result: prepared abstract & claims

**Decision locked:** the paper's primary contribution becomes the **internal–external
misalignment finding** and the **oracle–deployable gap**, demonstrated rigorously.
The method (EMR-CGC ensemble + QualityGate) is the vehicle/evidence, not the hero.
All numbers below are from the **validated canonical run** (5 seeds, full data).

**Status:** DRAFT for your approval. Nothing in the manuscript is changed yet.

---

## Canonical numbers these claims rest on

| Quantity | Value |
|---|---|
| Baseline (UMAP+Louvain) | 0.580 |
| AMR-DE (optimizes internal metrics) | **0.563** (below baseline) |
| EMR-CGC (label-free ensemble) | **0.631** |
| Deployable QualityGate | 0.623 |
| Oracle (label-selected upper bound) | 0.646 |
| Oracle gain recovered — gate / EMR-CGC / best learned selector | 65% / 77% / 64% |
| Deployable vs comparators (Holm) | beats only Spectral, HDBSCAN; tied with KMeans, GMM, Agg, DEC, IDEC, DCN; trails **N2D (0.642)** |
| Silhouette–ARI rank correlation (Phase-0) | ≈ 0.08 |

---

## 1. New abstract (full replacement)

> Graph-based clustering of high-dimensional data without a known cluster count
> conventionally returns the partition that maximizes an **internal** quality score
> such as modularity or silhouette. We show this standard practice is unsafe. Across
> twelve benchmarks, metaheuristically optimizing internal objectives *degrades*
> external agreement: a differential-evolution engine that maximizes modularity and
> silhouette attains mean ARI 0.563, **below** an unoptimized UMAP+Louvain baseline
> (0.580), because internal and external quality are only weakly correlated
> (silhouette–ARI rank correlation ≈ 0.08). We make this misalignment the object of
> study rather than a footnote. Treating a label-aware oracle that selects the
> best engine per dataset as an upper bound (mean ARI 0.646), we ask how much of it
> any **label-free** rule can recover: a three-layer QualityGate reaches 0.623 (65%
> of the oracle's gain over the baseline), and across learned, stability-, and
> consensus-based selectors the gap never closes (best 64%) — evidence that it is
> **structural**, not a tuning artifact. The ensemble the gate arbitrates (EMR-CGC,
> 0.631) is itself a competitive label-free method that infers the cluster count:
> after Holm correction it is statistically indistinguishable from strong classical
> baselines (K-Means, GMM, Agglomerative) and the DEC/IDEC/DCN deep methods, and it
> trails only N2D (0.642) — so weak clustering is not the bottleneck; the inability
> to *recognize* a good partition without labels is. We release all per-seed labels,
> internal scores, and gate decisions for verification.

*(~215 words. Removes the false "outperforms all baselines / 13.1% / matches N2D"
oracle claims; leads with the diagnostic result.)*

---

## 2. New contributions list (replaces the current four)

1. **A rigorous internal–external misalignment result (primary).** On twelve
   benchmarks, optimizing internal objectives with a metaheuristic *reduces* external
   accuracy (DE engine 0.563 vs baseline 0.580 ARI); silhouette barely tracks ARI
   (rank correlation ≈ 0.08). This reframes "which partition scores best internally"
   as the wrong question.
2. **A quantified oracle–deployable gap.** We define a label-aware oracle (0.646) as
   an upper bound and measure how far label-free selection reaches: the QualityGate
   recovers 65% of its gain over the baseline; across learned/stability/consensus
   selectors the best is 64%. The gap resists every standard label-free approach,
   indicating it is structural.
3. **A competitive label-free, cluster-count-inferring ensemble (EMR-CGC).** A
   multi-representation SNN–Leiden ensemble (0.631) that is statistically on par with
   strong classical and deep baselines without being given C — showing the limitation
   is recognition, not clustering power.
4. **Full reproducibility artifacts.** Per-seed labels, internal scores, gate
   decisions, and every figure/table/test script, released to reconcile all outputs.

The **QualityGate** is demoted from "the contribution" to a studied object: a
conservative label-free arbitrator that trades a small amount of mean ARI
(0.631 → 0.623) for protection against catastrophic single-engine failures, with
documented cases where it helps and where it hurts (e.g., the Wine regression).

---

## 3. Before → After for the key claim sentences

**(A) Headline / main-results sentence**
- ❌ OLD: "DEAC lifts mean ARI by 13.1% over the baseline and significantly
  outperforms five classical methods on 9 of 12 datasets."
- ✅ NEW: "The label-aware oracle improves mean ARI from 0.580 to 0.646 (+11%), but
  this upper bound is unavailable at deployment; the label-free QualityGate recovers
  65% of that gain (0.623), and the ensemble alone (EMR-CGC) recovers 77% (0.631)."

**(B) Deep-clustering comparison**
- ❌ OLD: "DEAC attains the highest mean ARI (0.649 vs 0.642 for N2D) and the most
  wins; parity-without-supervision."
- ✅ NEW: "Even the label-aware oracle only *matches* N2D (0.646 vs 0.642); the
  deployable method (EMR-CGC 0.631, gate 0.623) trails N2D, losing on 7 of 12
  datasets, though the difference is not significant."

**(C) Statistical-significance claim**
- ❌ OLD: "After Holm–Bonferroni, DEAC significantly outperforms all five classical
  methods and DCN; it is tied with N2D."
- ✅ NEW: "After Holm–Bonferroni over the nine comparisons, the deployable method
  significantly exceeds only the two weakest baselines (Spectral, HDBSCAN;
  p_Holm = 0.004). It is statistically indistinguishable from K-Means, GMM,
  Agglomerative, DEC, IDEC, and DCN, and trails N2D. We therefore claim competitive,
  not superior, performance — the paper's contribution is diagnostic."

**(D) Cluster-count claim**
- ❌ OLD: "DEAC attains the lowest cluster-count MAE (0.25) and highest exact-match
  (10/12)." (These were the oracle.)
- ✅ NEW: "The deployable gate's cluster-count error is substantial (MAE ≈ 5.8,
  exact 2/12): recognizing the right granularity without labels is itself part of the
  misalignment, not a solved sub-problem."

**(E) Conclusion opening**
- ❌ OLD: "We presented DEAC … which significantly outperforms classical methods and
  matches the strongest deep-clustering method."
- ✅ NEW: "We presented evidence that internal clustering metrics are an unreliable
  proxy for external accuracy, and a characterization of how far label-free selection
  can close the resulting oracle–deployable gap. A competitive label-free ensemble
  (EMR-CGC) shows the bottleneck is recognition, not clustering power; closing the
  residual gap plausibly requires supervision or a learned notion of external quality."

---

## 4. Downstream edits implied (for the next step, after you approve)

- Replace hardcoded `main_results`, `multi_metric`, `ablation` tables with the
  canonical-generated versions (gate = DEAC; oracle as a labeled diagnostic row).
- Swap in the regenerated `ccount`, `gate_oracle`, and new `stats` (gate-based)
  tables from `paper/generated_canonical/`.
- Regenerate the cross-dataset figures from the canonical source (removes ARI>1 /
  HAR / Letter; fixes Olivetti bar; separates oracle vs gate).
- Update §Discussion "Why Arbitration" to state the gate is a net −0.008 ARI vs
  EMR-CGC on average (safety, not accuracy).
- Optional title change to foreground the thesis, e.g.
  *"When Internal Metrics Mislead: The Oracle–Deployable Gap in Automatic Graph-Based
  Clustering."*

---

*Prepared from validated canonical data. On approval, I apply (A)–(E) to the LaTeX,
wire in the canonical tables/figures, regenerate stats on the gate, and recompile.*
