# Theoretical Convergence Analysis of GWO-DE-NM Hybrid

---

## Theorem 1: Global Convergence of GWO-DE-NM

**Statement:**
Let f: S -> R be the composite fitness function defined on bounded search space
S = [lb, ub]^D subset R^D. The GWO-DE-NM hybrid algorithm generates a sequence
{theta_t} that converges to the global optimum theta* with probability 1, i.e.,

    P(lim_{t->inf} f(theta_t) = f(theta*)) = 1

**Proof sketch:**

The proof proceeds in three parts, corresponding to the three phases.

---

### Part I: GWO with Levy Flights (Phase 1) — Global Exploration Guarantee

**Lemma 1.1 (Levy Flight Reachability):**
For any point theta_target in S and any epsilon > 0, there exists a finite
number of iterations T_0 such that for t >= T_0:

    P(||x_i^(t) - theta_target|| < epsilon) > 0  for at least one wolf i

*Proof:*
The Levy flight step L is drawn from a Levy stable distribution with
characteristic exponent beta = 1.5. The probability density function
satisfies:

    p(|L| > s) ~ s^{-beta}  as s -> inf

This heavy-tailed property ensures that:
1. P(|L| > R) > 0 for any R > 0 (unbounded support)
2. After clipping to [lb, ub], every point in S is reachable with non-zero
   probability from any starting position
3. The levy_weight w_L = 0.5(1 - t/T) + 0.01 > 0 for all t, ensuring
   Levy flights are always active

Therefore, the GWO with Levy flights satisfies the **global search condition**:
every point in S can be reached with positive probability.

**Lemma 1.2 (Monotonic Best-So-Far):**
The alpha wolf fitness f(alpha^(t)) is monotonically non-increasing:

    f(alpha^(t+1)) <= f(alpha^(t))  for all t

*Proof:*
By construction, alpha^(t) is updated only if a better solution is found.
The ranking step (line 6 of Algorithm 2) ensures that alpha always holds
the best-known position. Since fitness evaluations are deterministic for
fixed parameters, the alpha fitness sequence is monotonically non-increasing.

**Theorem 1.1 (GWO-Levy Convergence):**
Combining Lemma 1.1 and 1.2, the GWO with Levy flights converges to the
global optimum with probability 1 as T -> inf. This follows from the
classical result that a stochastic search algorithm converges globally if:
(a) every point is reachable with positive probability (Lemma 1.1), and
(b) the best solution is never lost (Lemma 1.2).

Reference: Solis & Wets (1981), "Minimization by Random Search Techniques,"
Mathematics of Operations Research, 6(1), 19-30.

---

### Part II: Differential Evolution (Phase 2) — Refinement Guarantee

**Lemma 2.1 (DE Population Improvement):**
The DE algorithm with greedy selection ensures:

    f(x_i^(g+1)) <= f(x_i^(g))  for all i, g

*Proof:*
The greedy selection rule (Algorithm 4, line 8) replaces x_i with u_i
only if f(u_i) < f(x_i). Therefore, each individual's fitness is
monotonically non-increasing across generations.

**Lemma 2.2 (Warm-Start Advantage):**
The warm-started DE population P_0 = {H_1,...,H_5, r_1,...,r_{43}} satisfies:

    min_{x in P_0} f(x) <= f(alpha_GWO)

since H_1 = alpha_GWO is included in the initial population.

**Lemma 2.3 (Elitist Guarantee):**
The elitist mechanism (Algorithm 4, lines 12-13) ensures:

    f(theta*_de) <= f(alpha_GWO)

*Proof:*
After DE completes, the final solution is:
    theta*_de = argmin(f(best_DE), f(alpha_GWO))
This guarantees that DE never produces a result worse than GWO's best.

**Theorem 2.1 (DE Convergence):**
Under standard conditions (bounded search space, continuous fitness),
DE with mutation, crossover, and greedy selection converges to the
global optimum as G -> inf.

Reference: Zaharie (2002), "Critical Values for the Control Parameters
of Differential Evolution Algorithms," Proc. MENDEL, 62-67.

---

### Part III: Nelder-Mead (Phase 3) — Local Convergence

**Theorem 3.1 (NM Local Convergence):**
For a continuous function f on R^D, the Nelder-Mead simplex method
converges to a local minimum from any starting simplex in a
neighborhood of a strict local minimum.

*Proof:*
The NM algorithm performs a sequence of operations (reflect, expand,
contract, shrink) that systematically reduce the simplex diameter.
For strictly convex functions, convergence to the unique minimum
is guaranteed (Lagarias et al., 1998).

For non-convex functions (our case), NM converges to a stationary
point. Since Phase 1 and Phase 2 already identified a near-optimal
region, NM refines within this basin of attraction.

**Safety Guarantee (Algorithm 5, lines 7-8):**

    f(theta*) <= f(theta*_de)

NM's result is accepted only if it improves upon DE's result.

Reference: Lagarias, J.C., Reeds, J.A., Wright, M.H., & Wright, P.E.
(1998), "Convergence Properties of the Nelder-Mead Simplex Method in
Low Dimensions," SIAM Journal on Optimization, 9(1), 112-147.

---

### Combined Convergence (Main Theorem)

**Proof of Theorem 1:**

By the chain of inequalities from each phase:

    f(theta*_NM) <= f(theta*_DE) <= f(alpha_GWO)

And by Theorem 1.1, as the GWO iterations T -> inf:

    f(alpha_GWO) -> f(theta*)  with probability 1

Therefore:

    f(theta*_NM) -> f(theta*)  with probability 1

The convergence rate is characterized by:
- Phase 1 (GWO): O(T * W) fitness evaluations for global exploration
- Phase 2 (DE): O(G * P) fitness evaluations for population-based refinement
- Phase 3 (NM): O(M) fitness evaluations for local polishing

Total: O(T*W + G*P + M) = O(12*12 + 30*48 + 50) = O(1634) evaluations

**QualityGate does not affect convergence** — it only selects between
baseline and hybrid results, ensuring the final output is never worse
than the baseline (a monotonicity guarantee on the output quality).

---

## Theorem 2: Computational Complexity

**Statement:**
The total computational complexity of the GWO-DE-NM hybrid is:

    O(E * F_cost + n * d * d')

where:
- E = total fitness evaluations = O(W*T + G*P*D + M)
- F_cost = cost of single fitness evaluation = O(n' * k * d' + n' * log(n'))
- n * d * d' = PCA cost (dominant for large datasets)

**Breakdown:**

| Component | Complexity | With defaults |
|-----------|-----------|---------------|
| PCA | O(n * d * d') | O(n * d * 40) |
| GWO (Phase 1) | O(W * T * F_cost) | O(144 * F_cost) |
| DE (Phase 2) | O(G * P * F_cost) | O(1440 * F_cost) |
| NM (Phase 3) | O(M * F_cost) | O(50 * F_cost) |
| F_cost (single eval) | O(n' * k * d' + n' * log(n')) | O(800 * 80 * 100) |
| QualityGate | O(1) | O(1) |
| **Total** | **O(E * F_cost + n * d * d')** | |

**Comparison with alternatives:**

| Method | Complexity | Notes |
|--------|-----------|-------|
| KMeans | O(n * k * d * I) | I = iterations |
| Spectral | O(n^2 * d + n^3) | Eigendecomposition |
| HDBSCAN | O(n^2) | Minimum spanning tree |
| **GWO-DE-NM** | **O(E * n' * k * d')** | **Subsampled** |

Key advantage: GWO-DE-NM uses subsampling (n' << n), making F_cost
independent of dataset size for large n. This gives effective
complexity O(E * n' * k * d' + n * d * d'), which scales
**linearly** with n (dominated by PCA), compared to O(n^2) or
O(n^3) for Spectral clustering.

---

## Theorem 3: QualityGate Correctness

**Statement:**
The 3-Layer Adaptive QualityGate guarantees:

    ARI(output) >= min(ARI(baseline), ARI(hybrid))

when the QualityGate score s(.) is a monotone function of clustering quality.

**Proof:**
The QualityGate selects between baseline and hybrid based on internal
quality metrics (MOD, SIL, BAL). If s(baseline) > s(hybrid), it
returns baseline, preserving at least baseline quality. If
s(hybrid) >= s(baseline), it returns hybrid.

The adaptive C-ratio threshold rho = max(4.0, sqrt(n/50)) ensures:
- Small datasets (n=400): strict threshold (rho=4.0) prevents
  over-fragmentation
- Large datasets (n=10000): permissive threshold (rho=14.1) allows
  discovery of more clusters when appropriate

**Limitation:** The guarantee assumes s(.) correlates with ARI, which
is not always perfect for unsupervised metrics. This is a fundamental
challenge in unsupervised clustering — addressed in Section 4.4.
