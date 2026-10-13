# Response to Reviewers & Revision Plan
### IJIES Paper 20266428 — *DEAC: A Defensive Ensemble Framework for Automatic Graph-Based Clustering of High-Dimensional Data*

**Status:** Rejected, resubmission invited after one month. This document is the **revision strategy and draft point-by-point response** agreed *before* executing the revisions. Each response states (a) our reading of the comment, (b) what we will change, and (c) which phase of the plan delivers it. The response letter will be finalized once the work is done, replacing every "we will" with "we have".

---

## 0. Strategic summary (read this first)

Both reviewers praise the **scientific contribution** — the finding that better internal validity can accompany worse external agreement, and the explicit oracle-vs-deployable separation. Neither rejects the idea. They reject the **evidence**, for one dominant reason:

> **The paper's headline results are computed on a label-selected *oracle*, but the paper's actual product is a *label-free* gate. An oracle that picks the best engine using ground-truth ARI does not validate an automatic method.**

Everything else (inconsistent figures, seed confusion, win/tie counts, complexity) is secondary to this. **The revision re-centers the entire evaluation on the frozen, label-free QualityGate.**

**Honesty note — the cost of doing this.** Re-centering on the deployable gate will *lower the headline numbers* and *change some conclusions*:

| Claim (current, oracle) | Claim (after re-centering, deployable gate) |
|---|---|
| Mean ARI **0.649** | Mean ARI **~0.630** (gate), baseline 0.582 |
| "Best/tied on 12/12 datasets" | Best on **6/12** (gate agrees with oracle on 6) |
| "Matches N2D (statistically tied)" | Gate **sits modestly below N2D** — parity claim does **not** survive for the deployable method |
| "+0.005 from QualityGate arbitration" | Gate (0.630) is actually **below EMR-CGC alone (0.644)** on the full data |

The honest, defensible story becomes: *a label-free, cluster-count-inferring method that is competitive with classical auto-C baselines, recovers ~75% of the oracle's gain, and has a documented, structural gap to supervised-count deep methods.* This is a **weaker but truthful** story. **Decision point for the authors:** whether this cleared-up story meets IJIES's bar as-is, or whether we also need a *stronger deployable selector* (new research — our Phase-1 investigation already found metric-reweighting/stability/consensus do **not** close the gap; only a modest RandomForest reaches 64%).

---

## 1. ⚠️ Version-control finding (must fix before anything else)

The submitted `DEAC finally.docx` is **not** the clean manuscript. It contains **stale artifacts from an earlier experimental iteration** that are the source of several Reviewer-1 complaints:

- Figure 6(b): **ARI = 1.05 / 1.03** (impossible; ARI ≤ 1) — a mislabeled axis or placeholder from a non-final run.
- A **"GROUSE-MH"** legend and datasets **HAR** and **Letter** that are **not** in the final 12-benchmark pool.
- Panel (a) labeling MFeat/CNAE-9 as **1,010** samples (correct: 2,000 and 1,080).
- **"seed 42"** in one location vs. "1001–1005" elsewhere.
- A duplicated/garbled abstract sentence.

**Action:** resubmit from the maintained LaTeX source, delete every stale asset, and attach a **figure/table provenance manifest** (script → output file → figure/table number) so the editor can verify every number. This single step resolves or de-risks R1.1, and parts of R1.2, R1.5.

---

## 2. Response to Reviewer 1 (9 points)

> **R1.1 — Figure 6 impossible ARI values, GROUSE-MH legend, HAR/Letter, Set1–Set15, wrong sample counts.**

**Response.** These panels are stale assets retained in error (see §1); ARI > 1 is not attainable and reflects a placeholder axis. **Action:** regenerate Figures 6(a–d) exclusively from the final 12-benchmark output files; remove GROUSE/HAR/Letter entirely; relabel panel (c) with the actual per-dataset gate activations (not Set1–Set15); verify every sample count against Table 3. Provenance manifest attached (Phase 0, R1.9).

> **R1.2 — Seed protocol is inconsistent (42 vs 1001–1005); report the deployable gate and all comparators over the same seeds with dispersion; the 13.1% is the oracle figure.**

**Response.** We adopt a **single protocol: five seeds 1001–1005, mean ± std, for every method including the deployable gate.** The "seed 42" reference is removed. Oracle results are reported **only** as a separately labeled diagnostic upper bound. The abstract's +13.1% is re-stated as the **deployable-gate** improvement (with the oracle figure given alongside, explicitly labeled). (Phase 1.)

> **R1.3 — §3.6 gives ISOLET baseline C=8; Table 10 gives 24; Figure 9 counts (17/19/11) disagree with Table 10 (20/38/10).**

**Response.** The "C=8" in the rationale was an illustrative value from a non-final subsample; the governing number is Table 10's. **Action:** correct §3.6 to the final baseline count, regenerate Figure 9 from the **same run and seed** as Table 10 so counts match exactly, and annotate any subsampling (the qualitative/count figures use a stratified 2,000-point sample — this will be stated, or the figures regenerated on full data). (Phase 0.)

> **R1.4 — Win/tie counts and Friedman ranks are not reproducible from the rounded tables; supply the full-precision matrix and a stated tie convention.**

**Response.** We will release the **unrounded per-seed ARI matrix** and adopt an explicit tie rule: a method is credited a *win* only on a strict maximum; ties are reported in a separate column. All win counts, and Friedman average ranks + Nemenyi CD, are recomputed from unrounded scores using standard **mid-rank** tie handling (this resolves the 1.58 vs ~1.458 discrepancy, which arose from rounding plus tie handling). (Phase 1.)

> **R1.5 — Figure 7 caption ("9 of 12") contradicts best-on-all-12; some bars (Olivetti baseline ≈0.34 vs 0.076) are wrong; separate wins from ties and oracle from gate.**

**Response.** The "9 of 12" conflated the multi-metric lead with the oracle's by-construction 12/12. **Action:** regenerate Figure 7 from correct values (Olivetti baseline = 0.076), and in all result plots **distinguish three quantities explicitly**: strict wins, ties, and oracle-vs-gate. (Phase 1.)

> **R1.6 — Figure 10 trajectories reach ~generation 13 though G=8; reconcile with the P·G·R budget and Table 13's E≈200.**

**Response.** The x-axis aggregated **R=2 restarts plus Nelder–Mead refinement**, not one DE run; axes will be relabeled to show restart/refinement boundaries. Budget accounting: nominal P·G·R = 20·8·2 = **320** candidate evaluations; we will report the **exact** evaluation count from logs and either correct "E≈200" or define it precisely (e.g., distinct evaluations after caching/early-stopping). (Phase 0 for relabeling; Phase 3 for exact counts.)

> **R1.7 — Table 13 gives t-SNE O(n²) but labels Total DEAC sub-quadratic; provide real timing and peak memory.**

**Response.** Correct — with t-SNE in the representation set, the honest worst case is **O(n²)**, not sub-quadratic. **Action:** either (a) correct the Total row to O(n²) and state when the sub-quadratic path applies (t-SNE disabled), or (b) replace exact t-SNE with an approximate variant and say so. We will add **measured end-to-end wall-clock and peak memory per dataset** on specified hardware, acknowledging that caching does not remove a representation's one-time cost. (Phase 0 fix; Phase 3 measurements.)

> **R1.8 — Specify the executable algorithm: candidate set / "hybrid H", normalization reference sets, invalid SIL/DB handling, penalties, bounds/rounding, bridging, small-community merging; how d′ is optimized under a "fixed representation"; and the r = pruning-ratio vs resolution contradiction.**

**Response.** We will add a complete specification: the gate's candidate set is **{Baseline, AMR-DE, EMR-CGC}** — there is no separate "hybrid H" (the term was a leftover and will be removed); all `[0,1]` normalizations with their reference sets; invalid-silhouette / single-cluster handling; the exact penalty terms; search bounds and integer rounding for (k, d′, q, r); graph bridging for disconnected components; and the small-community merge threshold. We will clarify that **RepScan fixes the representation *method*, while Stage-2 DE still optimizes the output dimension d′** (resolving "fixed representation"). The parameter **r is the Leiden resolution**; figures mislabeling it a "pruning ratio" (that role belongs to q) will be corrected. (Phase 0 text; Phase 3 final spec + pseudocode audit.)

> **R1.9 — Archive machine-readable artifacts: loaders, retained indices, settings, configs, seeds, per-seed labels, internal scores, gate decisions, and figure/table/test scripts.**

**Response.** We will publish a **tagged, DOI-archived release** (Zenodo snapshot of the GitHub repo) containing exact dataset loaders and retained sample indices, preprocessing/representation settings, all engine and baseline configurations, seeds, per-seed labels, per-engine internal scores, gate decisions, and the script that produces **each** figure, table, and statistical test. (Phase 3.)

---

## 3. Response to Reviewer 2 (5 points)

> **R2.1 — Make the frozen deployable rule the primary evaluated algorithm; compare it directly with independent alternatives under matched conditions; rewrite abstract and conclusion around those results. A central experimental redesign.**

**Response (the core revision).** We agree and will restructure accordingly:
- The **"DEAC" column in every main table, comparator table (classical + deep), ablation, and significance test becomes the label-free gate's output**, over 5 seeds with dispersion.
- The **oracle** appears once, clearly labeled as a diagnostic upper bound, not as "DEAC".
- The **abstract and conclusion are rewritten** around the deployable numbers (gate mean ARI ≈ 0.630; 75% oracle-gain recovery; honest standing vs. N2D).
- We will state plainly where the deployable method **under-performs** (it is not strictly monotone over the baseline; it trails N2D). (Phase 1 — backbone.)

> **R2.2 — The gate is an empirical heuristic; justify floor=4, c=50, weights, 0.06; disclose whether labels/ARI on these 12 datasets influenced them; freeze then test on new data or use a dev/eval split; add sensitivity analyses and per-dataset failure cases.**

**Response (DONE).** The constants' derivations are stated in §III-C (the floor of 4 follows from baselines under-segmenting by up to $\sim$4$\times$; $c{=}50$ fixes the floor/$\sqrt{n}$ crossover at $n{\approx}800$ via the $\sqrt{n/2}$ rule; the weights are the ensemble-scoring weights). We now add a **sensitivity analysis** (new Table~\ref{tab:sensitivity}) and a **dataset-level development/evaluation split** that settle the leakage question empirically:
- **Insensitive to the rho guard and weights:** mean ARI is flat across the rho floor (2–6: 0.617–0.624) and $c$ (25–150: 0.624), and over the full weight simplex stays in [0.571, 0.627] with the default within 0.003 of the best attainable; only a degenerate modularity-only weighting fails.
- **No leakage:** tuning the weights on 6 development datasets and evaluating on the disjoint 6 recovers only **40.8%** of the oracle's gain vs **64.9%** for the fixed default (**−24.2 points**). Fitting the constants to data generalizes *worse* than leaving them fixed, so the defaults carry no label-dependent advantage.
- **Failure cases** are reported honestly in §V: the gate is not monotone over the baseline (Wine regression, −0.011 ARI) and sits below EMR-CGC in aggregate (0.623 vs 0.631).

Reproduced by `run_sensitivity_driver.py` + `analyze_sensitivity.py` (60 units, 5 seeds). *Note:* the deployable gate selects by the internal score (0.40 MOD + 0.35 SIL + 0.25 BAL) under the rho guard; the $\Delta s{=}0.06$ layer belongs to the fuller three-layer gate described in §III-D. **(Completed.)**

> **R2.3 — Ensemble-diversity contribution is not isolated; PCA→UMAP→EMR changes many factors at once; §5.3 changes KNN+Louvain into SNN+Leiden together. Use controlled single-variable substitutions at equal budget, a simple best-internal-score selector, and gate-layer removals. Establish whether the DE engine's small oracle contribution is a real deployable benefit.**

**Response (d,e DONE; a,b,c in progress).**
- *(d) selection rule & (e) gate-layer ablation — done* (new Table~\ref{tab:selection}): over the three engines, the over-fragmentation guard barely binds (removing it leaves the gate unchanged at 0.624); the trivial **"always-EMR-CGC" policy (0.631) exceeds the gate (0.624)**; and dropping AMR-DE from the gate costs only **+0.003**. So the gate is **protective, not accuracy-maximizing**, and AMR-DE does **not** earn its cost deployably — stated plainly in §V.
- We confirm the manuscript states EMR-CGC selects the **single best-scoring** partition (not a consensus).
- *(a,b,c) controlled single-variable substitutions — DONE* (new Table~\ref{tab:controlled}, equal budget $k{=}35$, $r{=}0.25$): changing one factor at a time, the **representation dominates** (PCA→UMAP **+0.251** mean ARI), while the **detector** (Louvain→Leiden **−0.003**) and the **edge weighting** (KNN→SNN **−0.010**) barely move the mean. This directly answers the reviewer's point that PCA→UMAP→EMR and §5.3's KNN+Louvain→SNN+Leiden conflated factors: isolated, Leiden and SNN contribute almost nothing on average (SNN's benefit is confined to high-dimensional data), so the ensemble gain cannot be attributed to SNN weighting alone.

Reproduced by `analyze_selection_ablation.py`, `run_ablation_driver.py`, `analyze_ablation.py`. **(R2.3 fully completed.)**

> **R2.4 — Positioning limited to classical and DEC/IDEC/DCN/N2D; add recent methods (DPC-AKNN [23], PRCut 2025); separate known-C from unknown-C; report both common-representation comparisons and native pipelines rather than treating untuned implementations as ceilings.**

**Response (DONE).** Added both recent methods (new Table~\ref{tab:recent}), both **given the true $C$** (known-C), vs our label-free gate/EMR-CGC which infer it:
- **DPC-AKNN [23]** (2026): faithfully reimplemented from the paper (`src/evaluation/dpc_aknn.py`; validated — ARI 1.0 on two-moons/circles), native raw-feature geodesic pipeline, stratified cap 2000 ($O(n^3)$). Mean ARI **0.420**.
- **PRCut (2025)**: run with the **authors' released implementation** (`ayghri/prcut`; their exact model + ratio-cut loss) on the **same UMAP representation** as our engines (controlled common-representation comparison). Mean ARI **0.631**.
- Result: the deployable gate (0.623) and EMR-CGC (0.631) are **on par with PRCut (0.631)**, close to N2D (0.642), and **well above DPC-AKNN (0.420)** — while inferring $C$ rather than being told it. We claim competitive-without-the-count, not superiority.

Reproduced by `run_dpcaknn.py`, `run_prcut_ours.py`, `analyze_recent.py`. **(R2.4 completed.)**

> **R2.5 — p=0.622 does not establish equality; report effect sizes + uncertainty and an equivalence margin; distinguish dataset-level from per-dataset significance; evaluate robustness (imbalance, density, irrelevant dimensions, larger n).**

**Response (a,b DONE; c in progress).**
- *(a) DONE:* "statistically tied with N2D" is replaced by a quantified statement — gate $-$ N2D $= -0.019$ mean ARI (95\% CI $[-0.079, +0.040]$; Wilcoxon $p{=}0.73$; Cohen's $d_z{=}-0.18$) — plus a **TOST** equivalence test that does **not** establish equivalence even at a $\pm0.05$ margin ($p{=}0.17$). We therefore claim **neither superiority nor equivalence**, only a small, statistically unresolved difference. (`analyze_tost.py`.)
- *(b) DONE:* reworded every "significantly outperforms on $N$ of 12" statement; all significance claims are now explicitly **dataset-level** Wilcoxon/Holm (Table~\ref{tab:stats}), and per-dataset win counts are reported separately as wins, not significance. The stale "leads on 9 of 12" figure caption is fixed.
- *(c) DONE:* robustness on controlled synthetic data (new Table~\ref{tab:robustness}, 14 conditions × 3 seeds, ground truth known). The gate is **robust to class imbalance** (ARI 0.49–0.60 up to 30×), **improves with larger $n$** (0.32 at n=1k → 0.79 at n=10k) and with separation (→0.97), and has one clear **failure mode — irrelevant dimensions** (ARI 0.52 → ~0 as 50–500 noise features are added; every engine fails, not just the gate). We disclose this as an explicit scope condition (new Limitation (v)). Via `run_robustness_driver.py` + `analyze_robustness.py`.

**(R2.5 fully completed.)**

---

## 4. Phased revision plan

| Phase | Goal | Comments addressed | New experiments? | Rough effort |
|---|---|---|---|---|
| **0. Consistency pass** | Resubmit from LaTeX; delete stale assets; fix contradictions reproducible from existing outputs | R1.1, R1.3, R1.5(values), R1.6(labels), R1.7(table), R1.8(text + r fix) | No | ~2–3 days |
| **1. Deployable re-centering** | Gate becomes the primary method everywhere; 5-seed ± std; full-precision matrix; corrected stats + equivalence test; rewrite abstract/conclusion | **R2.1**, R1.2, R1.4, R2.5(a,b), R1.5(oracle/gate split) | Yes (re-run all comparisons on the gate) | ~1–2 weeks |
| **2. Gate validation & scope** | Dev/eval split + sensitivity + failure cases; controlled single-variable ablations; add DPC-AKNN & PRCut; common-representation vs native; robustness | R2.2, R2.3, R2.4, R2.5(c) | Yes (several) | ~2–3 weeks |
| **3. Artifacts & finalize** | Zenodo release; exact budget/complexity + timing/memory; final pseudocode spec; finalize response letter | R1.6(counts), R1.7(timing), R1.8(spec), R1.9 | Measurements only | ~3–4 days |

**Critical path:** Phase 1 (R2.1) gates everything — the new numbers it produces feed Phases 2–3 and the final response letter. **It must be done first and its results must be accepted honestly before we commit to resubmission.**

---

## 5. Open decisions for the authors

1. **Does the honest deployable story (gate ≈ 0.630, trails N2D) meet IJIES's bar?** If not, we must either (a) target a lower-tier venue, or (b) invest in a *stronger deployable selector* as new research (our Phase-1 work suggests this is hard — the gap is structural).
2. ~~**Were the gate constants ever tuned using the 12 benchmarks' labels/ARI?**~~ **ANSWERED (R2.2 done):** empirically no — the sensitivity + dev/eval analysis shows tuning the weights on held-out datasets *lowers* evaluation recovery (40.8% vs 64.9% default, −24.2 pts), so the fixed defaults carry no label-dependent advantage. Table~\ref{tab:sensitivity} is wired into the manuscript (§III-C).
3. **Authorship/affiliation** on the resubmission (currently Hussein Ali Ismael, University of Diyala) — confirm before finalizing.

---

*Prepared as the agreed strategy document. On your go-ahead, Phase 0 can begin immediately; Phase 1 is the decisive step.*
