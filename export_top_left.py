import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 导入 plot_multiseed_results 中的方法和配置
import plot_multiseed_results as pmr

def plot_single_reward():
    tag = "new_v13_ms"
    seeds = [42, 43, 44]
    window = 500
    save_dir = os.path.join(pmr.RESULT_DIR, f"{tag}_multiseed_plots")
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取算法列表
    algos = pmr.discover_algos(tag, seeds, pmr.RESULT_DIR)
    
    # 过滤掉一些随机或不需要的 baseline（保持与原图一致，或者保留所有）
    # 假设我们加载所有
    grouped = pmr.load_multiseed(tag, seeds, algos, pmr.RESULT_DIR)
    
    # 只画一张图
    fig, ax = plt.subplots(figsize=(7, 5))
    
    key = "ep_reward"
    ylabel = "Episode Reward"
    title = "Training Reward"
    
    for algo, runs in sorted(grouped.items(), key=lambda item: pmr.style_for(item[0]).get("zorder", 1)):
        style = pmr.style_for(algo)
        values = pmr.metric_array(runs, key)
        smoothed, x = pmr.smooth_stack(values, window)
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
        pmr.add_markers(ax, x, mean, style)

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best", framealpha=0.92, edgecolor="gray", fontsize=9)
    
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{tag}_multiseed_reward_only.pdf")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_single_reward()
