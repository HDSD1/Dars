import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "results", "ablation")

ALGO_CONFIG = {
    "DARS (Full)": {
        "dir_prefix": "new_v13_ms_seed",
        "file_name": "per_sac_cvx.npz",
        "color": "#D62728"
    },
    "w/o TDA": {
        "dir_prefix": "new_v13_ms_ablation_seed",
        "file_name": "no_atten.npz",
        "color": "#1F77B4"
    },
    "w/o Dual-Priority": {
        "dir_prefix": "new_v13_ms_ablation_seed",
        "file_name": "per_sac_cvx_no_surprise.npz",
        "color": "#2CA02C"
    }
}

ALGO_ORDER = ["DARS (Full)", "w/o TDA", "w/o Dual-Priority"]
SEEDS = [42, 43, 44]
WINDOW = 500
ZOOM_LAST_EPISODES = 2000

METRICS = [
    ("ep_reward", "Reward", "Training Reward over Episodes"),
    ("avg_rho", r"Delay Reduction Ratio $\rho$", r"Delay Reduction Ratio over Episodes"),
    ("avg_delay", "Delay (s)", "Average Delay over Episodes")
]

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
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def moving_average(data, window_size=100):
    """Smooth the curve while preserving the original length."""
    if window_size <= 1:
        return np.asarray(data, dtype=float)
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    return np.convolve(np.asarray(data, dtype=float), kernel, mode="same")

def load_curve_data():
    """Load full training curves for the ablation study."""
    curves = {}
    for algo_label in ALGO_ORDER:
        config = ALGO_CONFIG[algo_label]
        seeds_data = {"ep_reward": [], "avg_rho": [], "avg_delay": []}

        for seed in SEEDS:
            dir_name = f"{config['dir_prefix']}{seed}"
            path = os.path.join(RESULTS_DIR, dir_name, config['file_name'])

            if not os.path.exists(path):
                continue

            data = np.load(path, allow_pickle=True)
            seeds_data["ep_reward"].append(data["ep_reward"])
            seeds_data["avg_rho"].append(data["avg_rho"])
            seeds_data["avg_delay"].append(data["avg_delay"])

        if len(seeds_data["ep_reward"]) > 0:
            min_len = min(len(arr) for arr in seeds_data["ep_reward"])
            for m in ["ep_reward", "avg_rho", "avg_delay"]:
                seeds_data[m] = np.array([arr[:min_len] for arr in seeds_data[m]])
            curves[algo_label] = seeds_data
    return curves

def get_smoothed_mean_std(curves, algo, metric_key, window):
    raw_data = curves[algo][metric_key]
    mean_vals = np.mean(raw_data, axis=0)
    std_vals = np.std(raw_data, axis=0)
    smooth_mean = moving_average(mean_vals, window)
    smooth_std = moving_average(std_vals, window)
    x = np.arange(len(smooth_mean))
    return x, smooth_mean, smooth_std

def add_zoom_inset(ax, curves, metric_key, window):
    """
    Place the zoomed inset at the upper-right corner and make it smaller
    so it does not cover the main body of the curves.
    """
    axins = inset_axes(
        ax,
        width="34%",
        height="30%",
        loc="upper right",
        borderpad=0.9,
    )
    axins.set_facecolor("white")
    axins.patch.set_alpha(0.95)

    x2 = None
    for algo in ALGO_ORDER:
        if algo not in curves:
            continue
        color = ALGO_CONFIG[algo]["color"]
        x, smooth_mean, _ = get_smoothed_mean_std(curves, algo, metric_key, window)
        axins.plot(x, smooth_mean, color=color, linewidth=1.9)
        x2 = len(x) - 1 if len(x) > 0 else 0

    if x2 is None:
        return

    x1 = max(0, x2 - ZOOM_LAST_EPISODES)
    axins.set_xlim(x1, x2)

    y_vals_in_range = []
    for algo in ALGO_ORDER:
        if algo not in curves:
            continue
        _, smooth_mean, _ = get_smoothed_mean_std(curves, algo, metric_key, window)
        y_vals_in_range.extend(smooth_mean[x1:x2 + 1])

    if y_vals_in_range:
        y1 = float(np.min(y_vals_in_range))
        y2 = float(np.max(y_vals_in_range))
        y_margin = (y2 - y1) * 0.15
        if y_margin == 0:
            y_margin = max(abs(y1), 1.0) * 0.05
        axins.set_ylim(y1 - y_margin, y2 + y_margin)

    axins.tick_params(labelsize=8)
    axins.grid(True, linestyle=":", alpha=0.5)

    mark_inset(
        ax, axins,
        loc1=2, loc2=4,
        fc="none", ec="gray",
        alpha=0.55, linestyle="--"
    )

def plot_ablation_curves(curves):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (metric_key, ylabel, title) in zip(axes, METRICS):
        for algo in ALGO_ORDER:
            if algo not in curves:
                continue

            color = ALGO_CONFIG[algo]["color"]
            x, smooth_mean, smooth_std = get_smoothed_mean_std(curves, algo, metric_key, WINDOW)

            ax.plot(x, smooth_mean, label=algo, color=color, linewidth=1.8, alpha=0.9)
            ax.fill_between(
                x,
                smooth_mean - smooth_std,
                smooth_mean + smooth_std,
                color=color,
                alpha=0.15,
                edgecolor="none"
            )

        ax.set_xlabel("Training Episodes")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)

        add_zoom_inset(ax, curves, metric_key, WINDOW)

    axes[0].legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "fig_ablation_curves")
    plt.savefig(f"{out_path}.png")
    plt.savefig(f"{out_path}.pdf")
    print(f"Saved {out_path}.png and .pdf")
    plt.close()

    # Generate separate plots for each metric
    for metric_key, ylabel, title in METRICS:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        for algo in ALGO_ORDER:
            if algo not in curves:
                continue

            color = ALGO_CONFIG[algo]["color"]
            x, smooth_mean, smooth_std = get_smoothed_mean_std(curves, algo, metric_key, WINDOW)

            ax.plot(x, smooth_mean, label=algo, color=color, linewidth=1.8, alpha=0.9)
            ax.fill_between(
                x,
                smooth_mean - smooth_std,
                smooth_mean + smooth_std,
                color=color,
                alpha=0.15,
                edgecolor="none"
            )

        ax.set_xlabel("Training Episodes")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)

        add_zoom_inset(ax, curves, metric_key, WINDOW)

        ax.legend(loc="upper left", framealpha=0.9)
        plt.tight_layout()

        single_path = os.path.join(OUT_DIR, f"fig_ablation_{metric_key}")
        plt.savefig(f"{single_path}.png")
        plt.savefig(f"{single_path}.pdf")
        plt.close()
        print(f"Saved {single_path}.png and .pdf")

if __name__ == "__main__":
    curves_data = load_curve_data()
    if curves_data:
        plot_ablation_curves(curves_data)
    else:
        print("No data found to plot!")
