import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "results", "sensitivity", "raw_data_HOT")
OUT_DIR = os.path.join(ROOT, "results", "sensitivity")

ALGO_ORDER = [
    "per_sac_cvx",
    "sac_cvx",
    "td3_cvx",
    "maddpg_cvx",
    "esac_per_cvx",
]

ALGO_STYLES = {
    "per_sac_cvx": {"label": "DARS (Ours)", "color": "#D62728"},
    "sac_cvx": {"label": "SAC", "color": "#2CA02C"},
    "td3_cvx": {"label": "TD3", "color": "#1D3557"},
    "maddpg_cvx": {"label": "MADDPG", "color": "#FF7F0E"},
    "esac_per_cvx": {"label": "ESAC-PER", "color": "#1F77B4"},
}

METRICS = [
    ("eval_reward", "Reward", "Evaluation Reward"),
    ("eval_rho", r"Delay Reduction Ratio $\rho$", r"Evaluation Delay Reduction Ratio"),
    ("avg_delay", "Delay (s)", "Average Delay (Last 50 Episodes)"),
]

Y_AXIS_PADDING = {
    "eval_reward": (0.10, 0.12),
    "eval_rho": (0.10, 0.12),
    "avg_delay": (0.10, 0.12),
}


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.unicode_minus": False,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_results():
    """加载所有 HOT raw_data 文件, 返回逐种子记录列表."""
    pattern = re.compile(r"HOT_([0-9.]+)_seed(\d+)_(.+)\.npz$")
    rows = []
    for name in sorted(os.listdir(RAW_DIR)):
        match = pattern.match(name)
        if not match:
            continue
        hot = float(match.group(1))
        seed = int(match.group(2))
        algo = match.group(3)
        path = os.path.join(RAW_DIR, name)
        data = np.load(path, allow_pickle=True)

        rows.append({
            "hot": hot,
            "seed": seed,
            "algo": algo,
            "reward_mean": float(np.mean(data["eval_reward"])),
            "reward_std": float(np.std(data["eval_reward"])),
            "rho_mean": float(np.mean(data["eval_rho"])),
            "rho_std": float(np.std(data["eval_rho"])),
            "delay_mean": float(np.mean(data["avg_delay"][-50:])),
            "delay_std": float(np.std(data["avg_delay"][-50:])),
            "episodes": int(data["n_episodes"]) if "n_episodes" in data else len(data["ep_reward"]),
        })
    return rows


def aggregate_multiseed(rows):
    """将逐种子记录聚合为 (hot, algo) → {mean, std} 字典."""
    from collections import defaultdict
    buckets = defaultdict(lambda: {"reward": [], "rho": [], "delay": []})
    for row in rows:
        key = (row["hot"], row["algo"])
        buckets[key]["reward"].append(row["reward_mean"])
        buckets[key]["rho"].append(row["rho_mean"])
        buckets[key]["delay"].append(row["delay_mean"])

    agg = {}
    for (hot, algo), vals in buckets.items():
        agg[(hot, algo)] = {
            "reward_mean": np.mean(vals["reward"]),
            "reward_std": np.std(vals["reward"]),
            "rho_mean": np.mean(vals["rho"]),
            "rho_std": np.std(vals["rho"]),
            "delay_mean": np.mean(vals["delay"]),
            "delay_std": np.std(vals["delay"]),
            "n_seeds": len(vals["reward"]),
        }
    return agg


def write_summary(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "summary_HOT_bar.csv")
    fields = [
        "hot", "seed", "algo", "label", "episodes",
        "reward_mean", "reward_std", "rho_mean", "rho_std", "delay_mean", "delay_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["hot"], ALGO_ORDER.index(r["algo"]) if r["algo"] in ALGO_ORDER else 99, r["seed"])):
            out = dict(row)
            out["label"] = ALGO_STYLES.get(row["algo"], {}).get("label", row["algo"])
            writer.writerow(out)
    return path


def values_for(agg, hot_values, algo, metric):
    """提取 (均值列表, 标准差列表)."""
    key_map = {
        "eval_reward": ("reward_mean", "reward_std"),
        "eval_rho": ("rho_mean", "rho_std"),
        "avg_delay": ("delay_mean", "delay_std"),
    }
    mean_key, std_key = key_map[metric]
    means = [agg[(hot, algo)][mean_key] for hot in hot_values]
    stds = [agg[(hot, algo)][std_key] for hot in hot_values]
    return means, stds


def set_zoomed_ylim(ax, values, metric):
    y_max = max(values)
    if metric == "eval_reward":
        ax.set_ylim(300, y_max + (y_max - 300) * 0.15)
    elif metric == "eval_rho":
        ax.set_ylim(0.2, y_max + (y_max - 0.2) * 0.15)
    else:
        y_min = min(values)
        y_range = y_max - y_min
        if y_range <= 0:
            y_range = max(abs(y_max), 1.0) * 0.1
        lower_pad, upper_pad = Y_AXIS_PADDING[metric]
        ax.set_ylim(max(0, y_min - y_range * lower_pad), y_max + y_range * upper_pad)


def plot_grouped_bars(agg):
    hot_values = sorted({h for (h, _) in agg.keys()})
    available_algos = [algo for algo in ALGO_ORDER if any((h, algo) in agg for h in hot_values)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    x = np.arange(len(hot_values))
    width = 0.78 / len(available_algos)
    offsets = (np.arange(len(available_algos)) - (len(available_algos) - 1) / 2) * width

    for plot_idx, (ax, (metric, ylabel, title)) in enumerate(zip(axes, METRICS)):
        all_y = []
        for idx, algo in enumerate(available_algos):
            style = ALGO_STYLES.get(algo, {"label": algo, "color": "#555555"})
            y_means, y_stds = values_for(agg, hot_values, algo, metric)
            all_y.extend(y_means)
            ax.bar(
                x + offsets[idx],
                y_means,
                width=width,
                label=style["label"],
                color=style["color"],
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
            )

        set_zoomed_ylim(ax, all_y, metric)

        ax.set_xlabel(r"Hotspot Arrival Probability ($p_{hot}$)")
        ax.set_ylabel(ylabel)
        # ax.set_title(f"({chr(97 + plot_idx)}) {title}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.1f}" for v in hot_values])
        ax.grid(True, axis="y", linestyle="--", alpha=0.32)

    axes[0].legend(loc="best", framealpha=0.92, edgecolor="gray")
    plt.tight_layout()

    outputs = []
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"fig_hot_bar.{ext}")
        plt.savefig(path)
        outputs.append(path)
    plt.close()
    return outputs


def plot_single_metric(agg, metric, filename):
    hot_values = sorted({h for (h, _) in agg.keys()})
    available_algos = [algo for algo in ALGO_ORDER if any((h, algo) in agg for h in hot_values)]
    ylabel = next(item[1] for item in METRICS if item[0] == metric)
    title = next(item[2] for item in METRICS if item[0] == metric)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    x = np.arange(len(hot_values))
    width = 0.78 / len(available_algos)
    offsets = (np.arange(len(available_algos)) - (len(available_algos) - 1) / 2) * width

    all_y = []
    for idx, algo in enumerate(available_algos):
        style = ALGO_STYLES.get(algo, {"label": algo, "color": "#555555"})
        y_means, y_stds = values_for(agg, hot_values, algo, metric)
        all_y.extend(y_means)
        ax.bar(
            x + offsets[idx],
            y_means,
            width=width,
            label=style["label"],
            color=style["color"],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )

    set_zoomed_ylim(ax, all_y, metric)

    ax.set_xlabel(r"Hotspot Arrival Probability ($p_{hot}$)")
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in hot_values])
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.92, edgecolor="gray")
    plt.tight_layout()

    outputs = []
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{filename}.{ext}")
        plt.savefig(path)
        outputs.append(path)
    plt.close()
    return outputs


def main():
    rows = load_results()
    if not rows:
        raise RuntimeError(f"No HOT result files found in {RAW_DIR}")

    # 多种子聚合
    agg = aggregate_multiseed(rows)

    # 打印聚合信息
    seeds_info = {k: v["n_seeds"] for k, v in agg.items()}
    print(f"Loaded {len(rows)} records -> {len(agg)} (hot, algo) groups")
    seed_counts = set(seeds_info.values())
    print(f"Seeds per group: {seed_counts}")

    summary_path = write_summary(rows)
    outputs = plot_grouped_bars(agg)
    outputs += plot_single_metric(agg, "eval_rho", "fig_hot_bar_rho")
    outputs += plot_single_metric(agg, "eval_reward", "fig_hot_bar_reward")
    outputs += plot_single_metric(agg, "avg_delay", "fig_hot_bar_delay")

    print(f"Saved summary: {summary_path}")
    for path in outputs:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
