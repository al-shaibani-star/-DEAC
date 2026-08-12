# -*- coding: utf-8 -*-
"""
Phase 1: learned label-free selectors, evaluated Leave-One-Dataset-Out (no leak).

Uses the per-engine internal metrics saved by phase0_diagnostic.py
(results_phase0/) as features and ARI as the (training-only) target. For each
held-out dataset, a selector trained on the other 11 predicts a score per engine
and picks the argmax; we measure how often it picks the oracle engine and the
ARI it thereby achieves. Compared against the current fixed-weight gate and the
oracle upper bound.
"""
import json, glob, os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
ORDER = ["Wine","Glass","Olivetti","CNAE-9","COIL-20","Optdigits","MFeat",
         "Segmentation","ISOLET","Satimage","USPS","PenDigits"]
ENGS = ["Baseline","AMR-DE","EMR-CGC"]
FEATS = ["MOD","SIL","BAL","CH","DB","C"]

D = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(os.path.join(ROOT,"results_phase0","*.json"))}

def feat(rec):    # feature matrix (3 engines x |FEATS|) + ARI vector for one dataset
    X = np.array([[rec["engines"][e][f] for f in FEATS] for e in ENGS], float)
    a = np.array([rec["engines"][e]["ARI"] for e in ENGS], float)
    return X, a

data = {n: feat(D[n]) for n in ORDER}
baseline = np.array([data[n][1][0] for n in ORDER])          # Baseline ARI per dataset
oracle   = np.array([data[n][1].max() for n in ORDER])       # oracle (max ARI)

def evaluate(pick_fn, name):
    picks, agree = [], 0
    for i, n in enumerate(ORDER):
        Xte, ate = data[n]
        j = pick_fn(n, Xte)                                   # engine index chosen
        picks.append(ate[j])
        if j == int(np.argmax(ate)): agree += 1
    picks = np.array(picks)
    rec = 100*(picks.mean()-baseline.mean())/(oracle.mean()-baseline.mean())
    print(f"{name:<26} agree={agree:>2}/12  deployable ARI={picks.mean():.3f}  recover={rec:.0f}%")
    return picks.mean(), agree

# --- current fixed-weight gate (0.40 MOD + 0.35 SIL + 0.25 BAL) ---
def cur(n, X):
    s = 0.40*X[:,0] + 0.35*X[:,1] + 0.25*X[:,2]
    return int(np.argmax(s))

# --- LODO linear selector (learned weights over all 6 features) ---
def make_lodo(model_ctor):
    def pick(n, Xte):
        Xtr = np.vstack([data[m][0] for m in ORDER if m != n])
        ytr = np.concatenate([data[m][1] for m in ORDER if m != n])
        model = model_ctor().fit(Xtr, ytr)
        return int(np.argmax(model.predict(Xte)))
    return pick

print("Phase 1 — label-free selectors (LODO), 12 datasets\n" + "-"*64)
evaluate(lambda n,X: 0, "Baseline-only (reference)")
evaluate(cur, "Current gate (fixed w)")
evaluate(make_lodo(lambda: LinearRegression()), "Linear selector (LODO)")
evaluate(make_lodo(lambda: RandomForestRegressor(n_estimators=200, max_depth=3, random_state=0)), "RandomForest selector (LODO)")
evaluate(lambda n,X: int(np.argmax(data[n][1])), "Oracle (upper bound)")

# --- which features does the linear model rely on (fit on all 12) ---
Xall = np.vstack([data[n][0] for n in ORDER]); yall = np.concatenate([data[n][1] for n in ORDER])
lr = LinearRegression().fit(Xall, yall)
print("\nLinear ARI-model coefficients (fit on all 12):")
for f, c in zip(FEATS, lr.coef_):
    print(f"  {f:<4} {c:+.3f}")
