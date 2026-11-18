import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "seed_results.csv"

CONFIGS = {
    "bilstm_ph": ["--model", "bilstm", "--features", "features_pose_hands.npz"],
    "lstm_ph": ["--model", "lstm", "--features", "features_pose_hands.npz"],
    "bilstm_face10": ["--model", "bilstm", "--features", "features_pose_hands_face10.npz"],
}
FIELDS = ["config", "seed", "train", "val", "top1", "top3", "f1"]


def load_done():
    if not RESULTS.exists():
        return []
    with open(RESULTS) as f:
        return list(csv.DictReader(f))


def summarize(rows):
    print("\n=== mean ± std over seeds (test set) ===")
    for cfg in CONFIGS:
        r = [x for x in rows if x["config"] == cfg]
        if not r:
            continue
        a = np.array([float(x["top1"]) for x in r]) * 100
        f = np.array([float(x["f1"]) for x in r])
        sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
        print(f"{cfg:14s} n={len(r)}  top-1 {a.mean():.1f} ± {sd(a):.1f} "
              f"(min {a.min():.1f}, max {a.max():.1f})  macro-F1 {f.mean():.3f} ± {sd(f):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    rows = load_done()
    done = {(r["config"], int(r["seed"])) for r in rows}
    new_file = not RESULTS.exists()

    for cfg in args.configs:
        for seed in args.seeds:
            if (cfg, seed) in done:
                continue
            print(f"running {cfg} seed {seed} ...", flush=True)
            cmd = [sys.executable, "train.py", *CONFIGS[cfg],
                   "--seed", str(seed), "--name", f"{cfg}_s{seed}"]
            out = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True).stdout
            m1 = re.search(r"train acc ([\d.]+) \| val acc ([\d.]+)", out)
            m2 = re.search(r"TEST top-1 ([\d.]+) \| top-3 ([\d.]+) \| macro-F1 ([\d.]+)", out)
            if not (m1 and m2):
                print("  could not parse output, last lines:\n" + "\n".join(out.splitlines()[-5:]))
                continue
            row = dict(zip(FIELDS, [cfg, seed, *m1.groups(), *m2.groups()]))
            with open(RESULTS, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new_file:
                    w.writeheader()
                    new_file = False
                w.writerow(row)
            rows.append(row)
            print(f"  top-1 {row['top1']}  macro-F1 {row['f1']}", flush=True)

    summarize(rows)