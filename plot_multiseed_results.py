"""
Plot multi-seed comparison figures from completed runs.

Expected input layout:
  results/<tag>_seed<seed>/<algo>.npz

Example:
  python plot_multiseed_results.py --tag new_v11_ms --seeds 42,43,44
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


ALGO_STYLES = {
    "per_sac_cvx": {
        "label": "DARS (Ours)",
        "color": "#D62728",
        "ls": "-",
        "marker": "*",
        "lw": 2.8,
        "zorder": 10,
    },
    "esac_per_cvx": {
        "label": "ESAC-PER",
        "color": "#1F77B4",
        "ls": "--",
        "marker": "p",
        "lw": 2.2,
        "zorder": 8,
    },
    "tp_drl": {
        "label": "TP-DRL",
        "color": "#7B1FA2",
        "ls": "--",
        "marker": "X",
        "lw": 2.0,
        "zorder": 7,
    },
    "sac_cvx": {
        "label": "SAC",
        "color": "#2CA02C",
        "ls": "--",
        "marker": "o",
        "lw": 2.0,
        "zorder": 6,
    },
    "random": {
        "label": "Random+CVX",
        "color": "#BCBD22",
        "ls": ":",
        "marker": "+",
        "lw": 1.8,
        "zorder": 3,
    },
    "td3_cvx": {
        "label": "TD3",
        "color": "#9467BD",
        "ls": "-.",
        "marker": "s",
        "lw": 2.0,
        "zorder": 5,
    },
    "maddpg_cvx": {
        "label": "MADDPG",
        "color": "#FF7F0E",
        "ls": "-.",
        "marker": "^",
        "lw": 2.0,
        "zorder": 4,
    },
}


METRICS = [
    ("ep_reward", "Episode Reward", ""),
    ("avg_rho", r"Delay Reduction Ratio $\rho$", r"Delay Reduction Ratio ($\rho$)"),
    ("avg_delay", "Delay (s)", "Average Active-user Delay"),
    ("survival_steps", "Survival Steps", "Survival Steps"),
]


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.unicode_minus": False,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
})


def parse_list(text, cast=str):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def style_for(algo):
    return ALGO_STYLES.get(algo, {
        "label": algo,
        "color": "#555555",
        "ls": "-",
        "marker": "d",
        "lw": 2.0,
        "zorder": 1,
    })


def smooth_1d(data, window):
    if window <= 1 or len(data) < window:
        return np.asarray(data), np.arange(len(data))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(data, kernel, mode="valid"), np.arange(window - 1, len(data))


def smooth_stack(values, window):
    smoothed = []
    x_ref = None
    for row in values:
        s, x = smooth_1d(row, window)
        smoothed.append(s)
        if x_ref is None:
            x_ref = x
    return np.vstack(smoothed), x_ref


def discover_algos(tag, seeds, result_dir):
    common = None
    for seed in seeds:
        seed_dir = os.path.join(result_dir, f"{tag}_seed{seed}")
        if not os.path.isdir(seed_dir):
            raise FileNotFoundError(f"Missing seed directory: {seed_dir}")
        algos = {
            os.path.splitext(name)[0]
            for name in os.listdir(seed_dir)
            if name.endswith(".npz")
        }
        common = algos if common is None else common & algos
    return sorted(common, key=lambda a: style_for(a).get("zorder", 1), reverse=True)


def load_multiseed(tag, seeds, algos, result_dir):
    grouped = {}
    for algo in algos:
        runs = []
        for seed in seeds:
            path = os.path.join(result_dir, f"{tag}_seed{seed}", f"{algo}.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing result: {path}")
            data = np.load(path, allow_pickle=True)
            if "ep_reward" not in data:
                raise ValueError(f"{path} is not a training result npz")
            runs.append({key: data[key] for key in data.files})
        grouped[algo] = runs
    return grouped


def metric_array(runs, key):
    arrays = [np.asarray(run[key], dtype=float) for run in runs]
    min_len = min(len(a) for a in arrays)
    return np.vstack([a[:min_len] for a in arrays])


def add_markers(ax, x, y, style):
    every = max(1, len(x) // 8)
    ax.plot(
        x[::every],
        y[::every],
        linestyle="none",
        marker=style["marker"],
        markersize=style.get("ms", 7),
        color=style["color"],
        markeredgecolor="white",
        markeredgewidth=0.5,
        zorder=style.get("zorder", 1) + 1,
    )


def plot_training(grouped, save_dir, tag, window):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (key, ylabel, title) in zip(axes, METRICS):
        for algo, runs in sorted(grouped.items(), key=lambda item: style_for(item[0]).get("zorder", 1)):
            style = style_for(algo)
            values = metric_array(runs, key)
            smoothed, x = smooth_stack(values, window)
            mean = smoothed.mean(axis=0)
            std = smoothed.std(axis=0)
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=style["color"],
                alpha=0.13,
                linewidth=0,
                zorder=style.get("zorder", 1) - 0.2,
            )
            ax.plot(
                x,
                mean,
                color=style["color"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                label=style["label"],
                zorder=style.get("zorder", 1),
            )
            add_markers(ax, x, mean, style)

        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc="best", framealpha=0.92, edgecolor="gray", fontsize=9)

    plt.tight_layout(pad=2.0)
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(save_dir, f"{tag}_multiseed_training_curves.{ext}"))
    plt.close()


def plot_convergence(grouped, save_dir, tag, window):
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo, runs in sorted(grouped.items(), key=lambda item: style_for(item[0]).get("zorder", 1)):
        style = style_for(algo)
        values = metric_array(runs, "avg_rho")
        smoothed, x = smooth_stack(values, window)
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.13, linewidth=0)
        ax.plot(
            x,
            mean,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            label=style["label"],
            zorder=style.get("zorder", 1),
        )
        add_markers(ax, x, mean, style)

    ax.set_xlabel("Episode")
    ax.set_ylabel(r"Smoothed $\rho$")
    ax.set_title("Multi-seed Convergence Comparison")
    ax.grid(True)
    ax.legend(framealpha=0.92, edgecolor="gray", fontsize=10)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(save_dir, f"{tag}_multiseed_convergence.{ext}"))
    plt.close()


def final_metric_from_run(run, train_key, eval_key=None, last_n=50):
    if eval_key and eval_key in run and len(run[eval_key]) > 0:
        return float(np.mean(run[eval_key]))
    return float(np.mean(np.asarray(run[train_key], dtype=float)[-last_n:]))


def plot_final_bars(grouped, save_dir, tag):
    bar_metrics = [
        ("ep_reward", "eval_reward", "Reward", "Evaluation Reward"),
        ("avg_rho", "eval_rho", r"$\rho$", r"Evaluation Delay Reduction Ratio"),
        ("avg_delay", None, "Delay (s)", "Final Training Delay"),
        ("survival_steps", "eval_surv", "Steps", "Evaluation Survival Steps"),
    ]
    algos = sorted(grouped.keys(), key=lambda algo: style_for(algo).get("zorder", 1), reverse=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2))
    for ax, (train_key, eval_key, ylabel, title) in zip(axes, bar_metrics):
        labels, means, stds, colors = [], [], [], []
        for algo in algos:
            style = style_for(algo)
            vals = np.array([final_metric_from_run(run, train_key, eval_key) for run in grouped[algo]], dtype=float)
            labels.append(style["label"])
            means.append(vals.mean())
            stds.append(vals.std())
            colors.append(style["color"])

        bars = ax.bar(
            np.arange(len(labels)),
            means,
            yerr=stds,
            capsize=4,
            color=colors,
            alpha=0.86,
            edgecolor="black",
            linewidth=0.7,
        )
        offset = max(stds) * 0.25 if max(stds) > 0 else max(abs(np.array(means))) * 0.015
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(save_dir, f"{tag}_multiseed_final_performance.{ext}"))
    plt.close()


def write_summary(grouped, save_dir, tag):
    rows = []
    for algo, runs in grouped.items():
        for run in runs:
            seed = int(run["seed"]) if "seed" in run else -1
            rows.append({
                "algo": algo,
                "label": style_for(algo)["label"],
                "seed": seed,
                "train_reward_last50": final_metric_from_run(run, "ep_reward"),
                "train_rho_last50": final_metric_from_run(run, "avg_rho"),
                "train_delay_last50": final_metric_from_run(run, "avg_delay"),
                "train_survival_last50": final_metric_from_run(run, "survival_steps"),
                "eval_reward_mean": final_metric_from_run(run, "ep_reward", "eval_reward"),
                "eval_rho_mean": final_metric_from_run(run, "avg_rho", "eval_rho"),
                "eval_survival_mean": final_metric_from_run(run, "survival_steps", "eval_surv"),
                "total_time_min": float(run["total_time"]) / 60.0,
            })

    path = os.path.join(save_dir, f"{tag}_multiseed_plot_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    agg_path = os.path.join(save_dir, f"{tag}_multiseed_aggregate.csv")
    metric_keys = [
        "train_reward_last50",
        "train_rho_last50",
        "train_delay_last50",
        "train_survival_last50",
        "eval_reward_mean",
        "eval_rho_mean",
        "eval_survival_mean",
    ]
    agg_rows = []
    for algo in sorted(grouped.keys(), key=lambda name: style_for(name).get("zorder", 1), reverse=True):
        algo_rows = [row for row in rows if row["algo"] == algo]
        out = {"algo": algo, "label": style_for(algo)["label"], "seeds": len(algo_rows)}
        for key in metric_keys:
            vals = np.array([row[key] for row in algo_rows], dtype=float)
            out[f"{key}_mean"] = vals.mean()
            out[f"{key}_std"] = vals.std()
        agg_rows.append(out)

    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)

    print("\nMulti-seed aggregate (eval rho):")
    for row in agg_rows:
        print(
            f"  {row['label']:<16s} "
            f"{row['eval_rho_mean_mean']:.4f} +/- {row['eval_rho_mean_std']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Plot multi-seed training and evaluation results.")
    parser.add_argument("--tag", type=str, default="new_v11_ms")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--algos", type=str, default="",
                        help="Comma-separated algos. Default: common algos found in all seed dirs.")
    parser.add_argument("--dir", type=str, default=RESULT_DIR)
    parser.add_argument("--out", type=str, default="",
                        help="Output directory. Default: results/<tag>_multiseed_plots")
    parser.add_argument("--smooth", type=int, default=100)
    args = parser.parse_args()

    seeds = parse_list(args.seeds, int)
    algos = parse_list(args.algos, str) if args.algos else discover_algos(args.tag, seeds, args.dir)
    if not algos:
        raise RuntimeError("No algorithms found.")

    save_dir = args.out or os.path.join(args.dir, f"{args.tag}_multiseed_plots")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading tag={args.tag}, seeds={seeds}")
    print(f"Algorithms: {', '.join(algos)}")
    grouped = load_multiseed(args.tag, seeds, algos, args.dir)

    plot_training(grouped, save_dir, args.tag, args.smooth)
    plot_convergence(grouped, save_dir, args.tag, args.smooth)
    plot_final_bars(grouped, save_dir, args.tag)
    write_summary(grouped, save_dir, args.tag)

    print(f"\nSaved figures and tables to: {save_dir}")


if __name__ == "__main__":
    main()
