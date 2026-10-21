import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "results", "ablation")

# 定义三个版本的获取路径规则
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

METRICS = [
    ("eval_reward", "Reward", "Evaluation Reward"),
    ("eval_rho", r"Delay Reduction Ratio $\rho$", r"Evaluation Delay Reduction Ratio"),
    ("avg_delay", "Delay (s)", "Average Delay (Last 50 Episodes)")
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
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def load_ablation_data():
    """读取消融实验最后阶段的数据并求 3 个种子的均值"""
    agg = {}
    for algo_label in ALGO_ORDER:
        config = ALGO_CONFIG[algo_label]
        algo_data = {"eval_reward": [], "eval_rho": [], "avg_delay": []}
        
        for seed in SEEDS:
            dir_name = f"{config['dir_prefix']}{seed}"
            path = os.path.join(RESULTS_DIR, dir_name, config['file_name'])
            
            if not os.path.exists(path):
                print(f"Warning: {path} not found.")
                continue
            
            data = np.load(path, allow_pickle=True)
            # 取最后 50 轮的平均值代表最终稳态性能
            algo_data["eval_reward"].append(np.mean(data["eval_reward"][-50:]))
            algo_data["eval_rho"].append(np.mean(data["eval_rho"][-50:]))
            algo_data["avg_delay"].append(np.mean(data["avg_delay"][-50:]))
            
        if len(algo_data["eval_reward"]) > 0:
            agg[algo_label] = {
                "eval_reward": np.mean(algo_data["eval_reward"]),
                "eval_rho": np.mean(algo_data["eval_rho"]),
                "avg_delay": np.mean(algo_data["avg_delay"])
            }
    return agg

def plot_ablation_bars(agg):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    x_positions = np.arange(len(ALGO_ORDER))
    width = 0.5
    
    for ax, (metric_key, ylabel, title) in zip(axes, METRICS):
        values = [agg[algo][metric_key] for algo in ALGO_ORDER if algo in agg]
        labels = [algo for algo in ALGO_ORDER if algo in agg]
        colors = [ALGO_CONFIG[algo]["color"] for algo in ALGO_ORDER if algo in agg]
        
        bars = ax.bar(
            x_positions[:len(values)], 
            values, 
            width=width, 
            color=colors, 
            edgecolor="black", 
            linewidth=0.8,
            alpha=0.9
        )
        
        # 在柱子上标注具体数值
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                yval + (max(values)*0.01 if metric_key != "avg_delay" else -max(values)*0.05), 
                f"{yval:.3f}" if metric_key == "eval_rho" else f"{yval:.1f}", 
                ha='center', 
                va='bottom' if metric_key != "avg_delay" else 'top',
                fontweight='bold',
                fontsize=11
            )
            
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x_positions[:len(values)])
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        
        # 调整 Y 轴范围让差距更明显
        y_min, y_max = min(values), max(values)
        padding = (y_max - y_min) * 0.4
        
        if metric_key == "eval_reward":
            ax.set_ylim(max(0, y_min - padding), y_max + padding)
        elif metric_key == "eval_rho":
            ax.set_ylim(max(0, y_min - padding), y_max + padding)
        else:
            # 延迟是越低越好，稍微留点空间
            ax.set_ylim(max(0, y_min - padding), y_max + padding)
            
    plt.tight_layout()
    
    out_path = os.path.join(OUT_DIR, "fig_ablation_bar")
    plt.savefig(f"{out_path}.png")
    plt.savefig(f"{out_path}.pdf")
    print(f"Saved {out_path}.png and .pdf")
    
    plt.close()

if __name__ == "__main__":
    agg_data = load_ablation_data()
    if agg_data:
        plot_ablation_bars(agg_data)
    else:
        print("No data found to plot!")
