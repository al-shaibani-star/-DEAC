# -*- coding: utf-8 -*-
"""
Driver for the canonical run: executes each (dataset, seed) in its OWN fresh
Python subprocess (python run_canonical.py <name> <seed>). This isolation is the
fix for EMR-CGC collapsing to a single cluster after long sequential in-process
runs -- every unit gets a clean interpreter, which always produces valid results.

Resumable: skips units whose checkpoint already exists. Prints a final DONE line.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results_canonical")
RUNNER = os.path.join(HERE, "run_canonical.py")
DATASETS = ["Wine", "Glass", "Olivetti", "CNAE-9", "COIL-20", "Optdigits", "MFeat",
            "Segmentation", "ISOLET", "Satimage", "USPS", "PenDigits"]
SEEDS = [1001, 1002, 1003, 1004, 1005]
UNIT_TIMEOUT = 1800  # seconds per (dataset, seed)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    units = [(d, s) for d in DATASETS for s in SEEDS]
    t0 = time.time()
    ok = fail = skip = 0
    for i, (name, seed) in enumerate(units, 1):
        ck = os.path.join(OUT_DIR, f"{name}_s{seed}.json")
        if os.path.exists(ck):
            skip += 1
            print(f"[{i}/{len(units)}] [cache] {name} s{seed}", flush=True)
            continue
        print(f"[{i}/{len(units)}] running {name} s{seed} ...", flush=True)
        try:
            r = subprocess.run([sys.executable, RUNNER, name, str(seed)],
                               cwd=HERE, timeout=UNIT_TIMEOUT,
                               capture_output=True, text=True)
            tail = (r.stdout.strip().splitlines() or ["<no stdout>"])[-1]
            if os.path.exists(ck):
                ok += 1
                print(f"        -> {tail}", flush=True)
            else:
                fail += 1
                err = (r.stderr.strip().splitlines() or ["<no stderr>"])[-1]
                print(f"        -> FAILED (no checkpoint). stdout:{tail} | stderr:{err}", flush=True)
        except subprocess.TimeoutExpired:
            fail += 1
            print(f"        -> TIMEOUT after {UNIT_TIMEOUT}s", flush=True)
        except Exception as e:
            fail += 1
            print(f"        -> ERROR: {e}", flush=True)
    n = len(os.listdir(OUT_DIR)) if os.path.isdir(OUT_DIR) else 0
    print(f"\nDONE in {time.time()-t0:.0f}s. ok={ok} skip={skip} fail={fail} | "
          f"{n} checkpoints in results_canonical/", flush=True)


if __name__ == "__main__":
    main()
