# Response to Reviewers

**Journal:** International Journal of Intelligent Engineering and Systems (IJIES)
**Paper ID:** 20266428
**Title:** DEAC: A Defensive Ensemble Framework for Automatic Graph-Based Clustering of High-Dimensional Data

We thank both reviewers for their careful and constructive reviews. The common, central point — that our headline comparisons relied on a *label-selected oracle* while the paper's actual product is a *label-free* rule — was well taken, and it drove a substantial revision. We have **re-centered the entire evaluation on the deployable, label-free QualityGate**, regenerated every table and figure from a **single canonical run** (five seeds, full data), and reframed the paper around its genuine contribution: the internal–external misalignment finding and the oracle–deployable gap. We now claim *competitive, count-free* performance rather than superiority. All per-seed results and scripts are released.

## Summary of major changes
1. The **deployable gate is now the primary evaluated method** everywhere (main results, comparisons, ablations, significance); the oracle appears only as a clearly labeled diagnostic upper bound.
2. **One canonical source** (5 seeds, full data) produces all numbers; a full-precision per-seed matrix and all scripts are released (GitHub).
3. **New analyses:** gate-constant sensitivity + a development/evaluation leakage test; controlled single-variable ablations; two recent baselines (DPC-AKNN, PRCut); a TOST equivalence test vs. N2D; and a robustness study.
4. **Honest repositioning** of the abstract, contributions, results, and conclusion.

Below, each comment is answered with the change made and where it appears.

---

## Reviewer 1

**R1.1 — Figure 6 showed ARI>1, a GROUSE-MH legend, HAR/Letter datasets, and mismatched sample counts.**
These were stale assets from an earlier experimental iteration, inadvertently retained. All cross-dataset figures have been **regenerated from the final 12-benchmark canonical outputs**; ARI is in [0,1] throughout, only the twelve benchmarks appear, sample counts match Table (Datasets), and the panels show the actual per-dataset ARI, gate activation, and improvement. (Fig. "Main experimental results".)

**R1.2 — Inconsistent seeds (42 vs 1001–1005); report the gate and all comparators over the same seeds with dispersion; the 13.1% is the oracle figure.**
A single protocol is now used: **five fixed seeds (1001–1005), full data, for every method**, with dispersion. The "seed 42" reference is removed. The abstract is restated around the **deployable** numbers; the oracle is reported separately and explicitly labeled an upper bound. (Secs. "Evaluation Protocol", "Main Results".)

**R1.3 — §3.6 gave the ISOLET baseline as C=8 while Table 10 gave 24; Figure 9 counts disagreed.**
Corrected. The cluster-count table is regenerated from the canonical run, and the rationale now uses a **genuine, canonical-backed under-segmentation case — Olivetti, baseline C≈5 vs. 40 true classes** — instead of the erroneous ISOLET figure. The count table and figures now come from the same run. (Sec. "Design Rationale"; Table "cluster counts".)

**R1.4 — Recompute win summaries from unrounded results; state the tie convention; the Friedman rank is inconsistent.**
We release the **full-precision per-seed ARI matrix** (`canonical_matrix.csv`). All win counts and significance are recomputed from it; a method is credited a win only on a strict row-maximum. The earlier oracle-based Friedman rank is removed; significance is now dataset-level Wilcoxon with Holm–Bonferroni on the deployable gate. (Sec. "Statistical Significance"; Table "stats".)

**R1.5 — Figure 7 caption ("9 of 12") contradicted best-on-all-12; some bars (Olivetti baseline ≈0.34 vs 0.076) were wrong; separate wins/ties and oracle from gate.**
The caption is corrected (no "9 of 12" claim), the figures are regenerated with correct values (Olivetti baseline ≈0.06), and the main results table now **separates the three engines, the deployable gate, and the oracle** in distinct columns. (Fig. "metric bars"; Table "Main Results".)

**R1.6 — Figure 10 trajectories reach ~generation 13 though G=8; reconcile with the P·G·R budget and E≈200.**
The DE configuration is now stated consistently across the text and the figures as **G=8 generations, population P=20 (the LHS-initialised population over the four-dimensional search), R=2 restarts**. The trajectory text and caption state that the horizontal axis **concatenates the R=2 restarts** (hence it exceeds generation 8) before Nelder–Mead refinement, and the evaluation budget is reconciled as E = R·G·P ≈ 320 (= 2·8·20) in the complexity table, replacing the earlier "E≈200". The AMR-DE schematic (Fig. 3) now shows the same P=20. (Secs. "Evaluation Protocol", "Optimization Dynamics"; Table "complexity"; Fig. 3.)

**R1.7 — Table 13 labels t-SNE O(n²) yet the total sub-quadratic; provide measured timing and peak memory.**
The complexity table is corrected: t-SNE uses the Barnes–Hut approximation (output dimension ≤3), which is O(n log n); the sub-quadratic total then holds, with the O(nk²) SNN term dominating. We add **measured end-to-end wall-clock and peak memory** for the full three-engine pipeline: Wine (n=178) 29 s / 0.6 GB, MFeat (n=2000) 102 s / 0.7 GB, USPS (n=9298) 200 s / 1.8 GB. (Sec. "Computational Complexity"; Table "complexity".)

**R1.8 — Specify the executable algorithm; "hybrid candidate H"; r as pruning vs resolution.**
The parameter-definition conflict is resolved in **both the text and the schematics**: Figures 3 and 4 now label **q as the pruning quantile and r as the Leiden resolution** (the earlier "r = pruning ratio" / "q = quantile level" labels are removed), matching the released code (`fast_amr_de.py`, where `vec[2]` is the pruning quantile and `vec[3]` is the resolution). The "hybrid candidate H" is now defined explicitly where the gate is introduced: **B is the baseline partition and H is whichever engine output (AMR-DE or EMR-CGC) scores higher internally; the gate accepts H only if it clears all three layers, else falls back to B.** The gate description matches the **implemented deployable form** — among ρ-eligible engines, select the highest internal score (0.40 MOD + 0.35 SIL + 0.25 BAL) — with the Δs and regression checks as conservative safeguards that rarely bind (sensitivity analysis). The exact score normalisations, search bounds, integer rounding, graph bridging, and small-community merging are in the **released reference implementation**. (Secs. "QualityGate", "Design Rationale"; Figs. 3–4; code repository.)

**R1.9 — Archive machine-readable artifacts to reconcile outputs.**
We release, from a single canonical run: a **full-precision per-seed matrix** of internal scores, cluster counts, and gate decisions (`canonical_matrix.csv`); all per-seed result files; and every script that generates each figure, table, and statistical test. The claim in the manuscript is stated to match exactly what is released. (Code & data availability statement; repository.)

---

## Reviewer 2

**R2.1 — Make the frozen deployable rule the primary evaluated algorithm; rewrite the abstract and conclusion around it.**
Done, as the central revision. The deployable gate is the "DEAC" column in the main results, the classical and deep comparisons, the ablations, and the significance tests; the oracle is a labeled diagnostic upper bound only. The abstract and conclusion are rewritten around the deployable results (gate mean ARI 0.623, recovering 65% of the oracle's gain over the baseline; EMR-CGC 0.631). We state plainly where the deployable method underperforms. (Throughout.)

**R2.2 — Justify the gate constants; disclose any label influence; dev/eval split; sensitivity; failure cases.**
We add a sensitivity analysis and a dataset-level development/evaluation split. Mean ARI is **flat across the rho floor (2–6: 0.617–0.624) and c (25–150: 0.624)**, and across the full weight simplex stays in [0.571, 0.627] with the default within 0.003 of the best. Crucially, **tuning the weights on a 6-dataset development split and evaluating on the disjoint six recovers only 40.8% of the oracle's gain vs. 64.9% for the fixed default** (−24.2 points) — fitting the constants to data generalizes *worse*, so they carry no label-dependent advantage. Per-dataset failure cases (the Wine regression; the gate below EMR-CGC in aggregate) are reported. (Sec. "Design Rationale"; Table "sensitivity".)

**R2.3 — Isolate the ensemble/SNN/detector contributions with controlled, equal-budget substitutions; a best-internal-score selector and gate-layer removals; the DE engine's deployable value.**
Two controlled studies are added. (i) A **single-variable** ablation at matched budget: PCA→UMAP +0.251 mean ARI (representation dominates), Louvain→Leiden −0.003, KNN→SNN −0.010 — so the gain is not attributable to Leiden or SNN per se (SNN helps only on high-dimensional data). (ii) A **selection-rule** ablation: removing the ρ guard leaves the gate unchanged; "always-EMR-CGC" (0.631) in fact exceeds the gate (0.624); dropping AMR-DE from the gate costs only +0.003. We now state that the gate is protective, not accuracy-maximizing, and that AMR-DE does not earn its cost deployably. EMR-CGC is described as selecting the single best-scoring partition (not a consensus). (Secs. "Ablation Study", "Why Arbitration"; Tables "controlled", "selection".)

**R2.4 — Add recent methods (DPC-AKNN, PRCut); separate known-C from unknown-C; common-representation vs native pipelines.**
Both recent methods are added, each **given the true cluster count** (our gate/EMR-CGC infer it). **DPC-AKNN** [23] is faithfully reimplemented from the paper (validated: ARI 1.0 on two-moons/circles), native raw-feature geodesic pipeline → mean ARI 0.420. **PRCut** (AISTATS 2025) is run with the **authors' released implementation** on the **same UMAP representation** as our engines (a controlled common-representation comparison) → mean ARI 0.631. The deployable gate (0.623) and EMR-CGC (0.631) are on par with PRCut (0.631), close to N2D (0.642), and well above DPC-AKNN (0.420) — while inferring C. (Sec. "Comparison with Recent Methods"; Table "recent".)

**R2.5 — p=0.622 is not equality; report effect sizes, uncertainty, an equivalence margin; dataset-level vs per-dataset significance; robustness.**
(i) The "tied with N2D" wording is replaced by a quantified statement: gate − N2D = −0.019 mean ARI (95% CI [−0.079, +0.040]; Wilcoxon p=0.73; Cohen's d_z=−0.18), and a **TOST equivalence test does not establish equivalence even at a ±0.05 margin (p=0.17)** — we claim neither superiority nor equivalence. (ii) All significance statements are now explicitly dataset-level Wilcoxon/Holm; per-dataset win counts are reported as wins, not significance. (iii) A **robustness study** on controlled synthetic data shows the method is robust to class imbalance (ARI 0.49–0.60 up to 30×), improves with separation and with larger n (0.32→0.79 as n:1000→10000), and degrades sharply only under many irrelevant dimensions — stated as an explicit scope limitation. (Secs. "Statistical Significance", "Robustness", "Limitations"; Table "robustness".)

---

## Additional Comments (Editorial)

**A1 — Tables and figures must be cited, provided with legends, and numbered serially.**
Every figure and table is now **cited in the running text** (Fig. 1–11 and Table 1–19 are each referenced where they are discussed), carries a **legend**, and is **numbered serially**. Figure captions appear centred below the figure and table titles centred above the table, in the "Fig. N." / "Table N." form. In-text reference citations, which were missing from the Word export, are restored and numbered ([1]–[34]).

---

## Formatting

The manuscript has been reformatted to the official **IJIES template (Ver. 2026.10.1, `IJIES_Format.docx`)**: A4 two-column layout within the prescribed print area; Times New Roman throughout (14 pt title, 11 pt author, 11 pt body, 10 pt abstract/keywords/captions, 12/11/10 pt headings); numbered first-order headings with a period; the required **Conflicts of Interest** and **Acknowledgments** statements; and the IJIES reference style. No figure, table, or equation extends beyond the print area.

---

We believe these revisions address every point and make the paper's evidence internally consistent and reproducible. We thank the reviewers again for feedback that materially improved the work.
