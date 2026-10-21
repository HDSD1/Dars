"""
Run multi-seed experiments without overwriting existing single-seed results.

Each seed is saved under:
  results/<tag>_seed<seed>/<algo>.npz

The script also writes a compact CSV summary to:
  results/<tag>_multiseed_summary.csv
"""

import argparse
import csv
import os
import subprocess
import sys

import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(ROOT, "results")


def parse_list(text, cast=str):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def run_one(algo, episodes, tag, seed, hotspot=False, nowct=False):
    seed_tag = f"{tag}_seed{seed}"
    cmd = [
        sys.executable, "train_unified.py",
        "--algo", algo,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--tag", seed_tag,
    ]
    if hotspot:
        cmd.append("--hotspot")
    if nowct:
        cmd.append("--nowct")

    print("\n" + "=" * 72)
    print(f"Running {algo} | seed={seed} | tag={seed_tag} | episodes={episodes}")
    print("=" * 72)
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def load_metric(tag, algo, seed):
    path = os.path.join(RESULT_DIR, f"{tag}_seed{seed}", f"{algo}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    row = {
        "algo": algo,
        "seed": seed,
        "episodes": int(data["n_episodes"]),
        "train_reward_last50": float(np.mean(data["ep_reward"][-50:])),
        "train_rho_last50": float(np.mean(data["avg_rho"][-50:])),
        "train_delay_last50": float(np.mean(data["avg_delay"][-50:])),
        "train_survival_last50": float(np.mean(data["survival_steps"][-50:])),
        "total_time": float(data["total_time"]),
    }
    if "eval_rho" in data and len(data["eval_rho"]) > 0:
        row.update({
            "eval_reward_mean": float(np.mean(data["eval_reward"])),
            "eval_reward_std": float(np.std(data["eval_reward"])),
            "eval_rho_mean": float(np.mean(data["eval_rho"])),
            "eval_rho_std": float(np.std(data["eval_rho"])),
            "eval_survival_mean": float(np.mean(data["eval_surv"])),
            "eval_survival_std": float(np.std(data["eval_surv"])),
        })
    return row


def write_summary(tag, algos, seeds):
    rows = []
    for algo in algos:
        for seed in seeds:
            row = load_metric(tag, algo, seed)
            if row is not None:
                rows.append(row)

    if not rows:
        print("No finished seed results found; summary skipped.")
        return

    path = os.path.join(RESULT_DIR, f"{tag}_multiseed_summary.csv")
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-seed summary: {path}")

    print("\nMulti-seed aggregate:")
    for algo in algos:
        algo_rows = [r for r in rows if r["algo"] == algo]
        if not algo_rows:
            continue
        metric = "eval_rho_mean" if "eval_rho_mean" in algo_rows[0] else "train_rho_last50"
        vals = np.array([r[metric] for r in algo_rows], dtype=float)
        print(f"  {algo:18s} {metric}: {vals.mean():.4f} +/- {vals.std(ddof=0):.4f} ({len(vals)} seeds)")


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed UAV-MEC experiments.")
    parser.add_argument("--algos", type=str, required=True,
                        help="Comma-separated algorithm names.")
    parser.add_argument("--seeds", type=str, default="42,43,44",
                        help="Comma-separated seeds. Default: 42,43,44.")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--tag", type=str, default="new_v11_ms")
    parser.add_argument("--hotspot", action="store_true")
    parser.add_argument("--nowct", action="store_true")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only regenerate the CSV/aggregate from existing results.")
    args = parser.parse_args()

    algos = parse_list(args.algos, str)
    seeds = parse_list(args.seeds, int)

    if not args.summary_only:
        failed = []
        for algo in algos:
            for seed in seeds:
                ok = run_one(algo, args.episodes, args.tag, seed,
                             hotspot=args.hotspot, nowct=args.nowct)
                if not ok:
                    failed.append((algo, seed))
        if failed:
            print(f"\nFailed runs: {failed}")

    write_summary(args.tag, algos, seeds)


if __name__ == "__main__":
    main()
