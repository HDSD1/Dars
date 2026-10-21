import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "results", "ablation")

ALGO_CONFIG = {
    "DARS (Full)": {
        "dir_prefix": "new_v13_ms_seed",
        "file_name": "per_sac_cvx.npz",
        "color": "#D62728"  # 红色
    },
    "w/o TDA": {
        "dir_prefix": "new_v13_ms_ablation_seed",
        "file_name": "no_atten.npz",
        "color": "#1F77B4"  # 蓝色
    },
    "w/o Dual-Priority": {
        "dir_prefix": "new_v13_ms_ablation_seed",
        "file_name": "per_sac_cvx_no_surprise.npz",
        "color": "#2CA02C"  # 绿色
    }
}

ALGO_ORDER = ["DARS (Full)", "w/o TDA", "w/o Dual-Priority"]
SEEDS = [42, 43, 44]

METRICS = [
    ("eval_reward", "Reward", "Evaluation Reward Distribution"),
    ("eval_rho", r"Delay Reduction Ratio $\rho$", r"Delay Reduction Ratio Distribution"),
    ("avg_delay", "Delay (s)", "Average Delay Distribution")
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

def load_ablation_distributions():
    """读取消融实验最后50轮的数据，汇总3个种子形成分布 (150个数据点)"""
    distributions = {}
    for algo_label in ALGO_ORDER:
        config = ALGO_CONFIG[algo_label]
        algo_data = {"eval_reward": [], "eval_rho": [], "avg_delay": []}
        
        for seed in SEEDS:
            dir_name = f"{config['dir_prefix']}{seed}"
            path = os.path.join(RESULTS_DIR, dir_name, config['file_name'])
            
            if not os.path.exists(path):
                continue
            
            data = np.load(path, allow_pickle=True)
            # 提取最后 50 轮
            algo_data["eval_reward"].extend(data["eval_reward"][-50:])
            algo_data["eval_rho"].extend(data["eval_rho"][-50:])
            algo_data["avg_delay"].extend(data["avg_delay"][-50:])
            
        if len(algo_data["eval_reward"]) > 0:
            distributions[algo_label] = {
                "eval_reward": np.array(algo_data["eval_reward"]),
                "eval_rho": np.array(algo_data["eval_rho"]),
                "avg_delay": np.array(algo_data["avg_delay"])
            }
    return distributions

def plot_ablation_boxplots(dist_data):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    
    for ax, (metric_key, ylabel, title) in zip(axes, METRICS):
        plot_data = [dist_data[algo][metric_key] for algo in ALGO_ORDER if algo in dist_data]
        labels = [algo for algo in ALGO_ORDER if algo in dist_data]
        colors = [ALGO_CONFIG[algo]["color"] for algo in ALGO_ORDER if algo in dist_data]
        
        if not plot_data:
            continue
            
        # 绘制箱线图
        bplot = ax.boxplot(
            plot_data, 
            patch_artist=True,  # 允许填充颜色
            tick_labels=labels,
            widths=0.4,
            showfliers=True,    # 显示异常值
            flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5, linestyle='none'),
            medianprops=dict(color='white', linewidth=2), # 中位数线为白色粗线
            boxprops=dict(linewidth=1.2, edgecolor='black'),
            whiskerprops=dict(linewidth=1.2, color='black'),
            capprops=dict(linewidth=1.2, color='black')
        )
        
        # 为每个箱子填充对应的颜色
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        
        # 适当调整 Y 轴，以免离群点顶着边缘
        all_vals = np.concatenate(plot_data)
        y_min, y_max = np.min(all_vals), np.max(all_vals)
        padding = (y_max - y_min) * 0.15
        if padding == 0:
            padding = y_max * 0.1
            
        if metric_key == "eval_reward":
            ax.set_ylim(max(0, y_min - padding), y_max + padding)
        else:
            ax.set_ylim(max(0, y_min - padding), y_max + padding)
            
    plt.tight_layout()
    
    out_path = os.path.join(OUT_DIR, "fig_ablation_boxplot")
    plt.savefig(f"{out_path}.png")
    plt.savefig(f"{out_path}.pdf")
    print(f"Saved {out_path}.png and .pdf")
    plt.close()

if __name__ == "__main__":
    dist_data = load_ablation_distributions()
    if dist_data:
        plot_ablation_boxplots(dist_data)
    else:
        print("No data found to plot!")
