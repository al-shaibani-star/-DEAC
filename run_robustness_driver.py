# -*- coding: utf-8 -*-
"""Driver for R2.5(c) robustness: one fresh subprocess per (condition, seed).
Resumable. 14 conditions x 3 seeds = 42 units. Prints a final DONE line."""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_robustness")
RUN = os.path.join(HERE, "run_robustness.py")
NCOND = 14
SEEDS = [1001, 1002, 1003]
UNIT_TIMEOUT = 1800


def main():
    os.makedirs(OUT, exist_ok=True)
    units = [(i, s) for i in range(NCOND) for s in SEEDS]
    t0 = time.time(); ok = skip = fail = 0
    for n, (i, s) in enumerate(units, 1):
        ck = os.path.join(OUT, f"c{i}_s{s}.json")
        if os.path.exists(ck):
            skip += 1; print(f"[{n}/{len(units)}] [cache] c{i} s{s}", flush=True); continue
        print(f"[{n}/{len(units)}] c{i} s{s} ...", flush=True)
        try:
            r = subprocess.run([sys.executable, RUN, str(i), str(s)], cwd=HERE,
                               timeout=UNIT_TIMEOUT, capture_output=True, text=True)
            if os.path.exists(ck):
                ok += 1; print(f"        -> {(r.stdout.strip().splitlines() or ['?'])[-1]}", flush=True)
            else:
                fail += 1; print(f"        -> FAILED: {(r.stderr.strip().splitlines() or ['?'])[-1]}", flush=True)
        except subprocess.TimeoutExpired:
            fail += 1; print(f"        -> TIMEOUT {UNIT_TIMEOUT}s", flush=True)
    nck = len([f for f in os.listdir(OUT) if f.endswith('.json')]) if os.path.isdir(OUT) else 0
    print(f"\nDONE in {time.time()-t0:.0f}s. ok={ok} skip={skip} fail={fail} | {nck}/{len(units)} checkpoints", flush=True)


if __name__ == "__main__":
    main()
