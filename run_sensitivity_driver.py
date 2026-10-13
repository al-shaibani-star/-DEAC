# -*- coding: utf-8 -*-
"""Driver for R2.2 capture: one fresh subprocess per (dataset, seed) -- the same
isolation that keeps EMR-CGC from collapsing. Resumable. Prints a final DONE line."""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_sensitivity")
RUN = os.path.join(HERE, "run_sensitivity.py")
DS = ["Wine", "Glass", "Olivetti", "CNAE-9", "COIL-20", "Optdigits", "MFeat",
      "Segmentation", "ISOLET", "Satimage", "USPS", "PenDigits"]
SEEDS = [1001, 1002, 1003, 1004, 1005]
UNIT_TIMEOUT = 1800


def main():
    os.makedirs(OUT, exist_ok=True)
    units = [(d, s) for d in DS for s in SEEDS]
    t0 = time.time(); ok = skip = fail = 0
    for i, (n, s) in enumerate(units, 1):
        ck = os.path.join(OUT, f"{n}_s{s}.json")
        if os.path.exists(ck):
            skip += 1; print(f"[{i}/{len(units)}] [cache] {n} s{s}", flush=True); continue
        print(f"[{i}/{len(units)}] {n} s{s} ...", flush=True)
        try:
            r = subprocess.run([sys.executable, RUN, n, str(s)], cwd=HERE,
                               timeout=UNIT_TIMEOUT, capture_output=True, text=True)
            if os.path.exists(ck):
                ok += 1; print(f"        -> {(r.stdout.strip().splitlines() or ['?'])[-1]}", flush=True)
            else:
                fail += 1; print(f"        -> FAILED: {(r.stderr.strip().splitlines() or ['?'])[-1]}", flush=True)
        except subprocess.TimeoutExpired:
            fail += 1; print(f"        -> TIMEOUT {UNIT_TIMEOUT}s", flush=True)
    nck = len([f for f in os.listdir(OUT) if f.endswith('.json')]) if os.path.isdir(OUT) else 0
    print(f"\nDONE in {time.time()-t0:.0f}s. ok={ok} skip={skip} fail={fail} | {nck}/60 checkpoints", flush=True)


if __name__ == "__main__":
    main()
