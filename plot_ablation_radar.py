import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "results", "ablation")

ALGO_CONFIG = {
    "DARS (Full)": {
        "dir_prefix": "new_v13_ms_seed",
        "file_name": "per_sac_cvx.npz",
        "color": "#D62728", # 红色
        "marker": "o"
    },
    "w/o TDA": {
        "dir_prefix": "new_v13_ms_ablation_seed",
        "file_name": "no_atten.npz",
        "color": "#1F77B4", # 蓝色
        "marker": "s"
    },
    "w/o Dual-Priority": {
        "dir_prefix": "new_v13_ms_ablation_seed",
        "file_name": "per_sac_cvx_no_surprise.npz",
        "color": "#2CA02C", # 绿色
        "marker": "^"
    }
}

ALGO_ORDER = ["DARS (Full)", "w/o TDA", "w/o Dual-Priority"]
SEEDS = [42, 43, 44]

# 我们要展示的 5 个维度
CATEGORIES = [
    'Overall\nReward', 
    'Delay\nReduction', 
    'Low\nLatency', 
    'Energy/\nSurvival', 
    'Training\nStability'
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 15,
})

def load_radar_data():
    """提取 5 个维度的数据并求 3 种子的均值"""
    raw_data = {}
    for algo_label in ALGO_ORDER:
        config = ALGO_CONFIG[algo_label]
        algo_data = {
            "reward": [], "rho": [], "delay": [], "surv": [], "std": []
        }
        
        for seed in SEEDS:
            dir_name = f"{config['dir_prefix']}{seed}"
            path = os.path.join(RESULTS_DIR, dir_name, config['file_name'])
            
            if not os.path.exists(path):
                continue
            
            data = np.load(path, allow_pickle=True)
            # 取最后 50 轮
            r = data["eval_reward"][-50:]
            algo_data["reward"].append(np.mean(r))
            algo_data["rho"].append(np.mean(data["eval_rho"][-50:]))
            algo_data["delay"].append(np.mean(data["avg_delay"][-50:]))
            # 注意：有的叫 survival_steps 有的叫 eval_surv
            if "eval_surv" in data:
                algo_data["surv"].append(np.mean(data["eval_surv"][-50:]))
            elif "survival_steps" in data:
                algo_data["surv"].append(np.mean(data["survival_steps"][-50:]))
            else:
                algo_data["surv"].append(1.0) # Fallback
                
            # 稳定性用 Reward 的标准差的倒数来衡量 (标准差越小越稳定)
            algo_data["std"].append(np.std(r))
            
        if len(algo_data["reward"]) > 0:
            raw_data[algo_label] = {
                "reward": np.mean(algo_data["reward"]),
                "rho": np.mean(algo_data["rho"]),
                "delay": np.mean(algo_data["delay"]),
                "surv": np.mean(algo_data["surv"]),
                "std": np.mean(algo_data["std"])
            }
    return raw_data

def normalize_data(raw_data):
    """
    将所有指标归一化到 [0.1, 1.0] 之间，1.0 代表最佳表现。
    为了让雷达图不要缩在圆心（太难看），下限设为 0.2 或 0.3 左右。
    """
    norm_data = {}
    
    # 提取各项的极值
    max_reward = max(d["reward"] for d in raw_data.values())
    min_reward = min(d["reward"] for d in raw_data.values())
    
    max_rho = max(d["rho"] for d in raw_data.values())
    min_rho = min(d["rho"] for d in raw_data.values())
    
    max_delay = max(d["delay"] for d in raw_data.values())
    min_delay = min(d["delay"] for d in raw_data.values())
    
    max_surv = max(d["surv"] for d in raw_data.values())
    min_surv = min(d["surv"] for d in raw_data.values())
    
    max_std = max(d["std"] for d in raw_data.values())
    min_std = min(d["std"] for d in raw_data.values())

    def scale_higher_is_better(val, mi, ma):
        if ma == mi: return 1.0
        # 映射到 0.4 到 1.0
        return 0.4 + 0.6 * (val - mi) / (ma - mi)
        
    def scale_lower_is_better(val, mi, ma):
        if ma == mi: return 1.0
        # 映射到 0.4 到 1.0 (越低分越高)
        return 0.4 + 0.6 * (ma - val) / (ma - mi)

    for algo, vals in raw_data.items():
        norm_data[algo] = [
            scale_higher_is_better(vals["reward"], min_reward, max_reward),
            scale_higher_is_better(vals["rho"], min_rho, max_rho),
            scale_lower_is_better(vals["delay"], min_delay, max_delay), # 低延迟
            scale_higher_is_better(vals["surv"], min_surv, max_surv),
            scale_lower_is_better(vals["std"], min_std, max_std)        # 低波动
        ]
        
    return norm_data

def plot_radar(norm_data):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    N = len(CATEGORIES)
    # 计算每个轴的角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # 闭合多边形

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    
    # 设置雷达图的起始角度和旋转方向
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # 绘制网格线和刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATEGORIES, fontweight='bold', size=15)
    ax.tick_params(axis='x', pad=20)  # 将标签推到圆圈外面
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([]) # 隐藏数字标签，保持简洁
    ax.set_ylim(0, 1.05)

    for algo in ALGO_ORDER:
        if algo not in norm_data:
            continue
            
        values = norm_data[algo]
        values += values[:1] # 闭合数据点
        
        color = ALGO_CONFIG[algo]["color"]
        marker = ALGO_CONFIG[algo]["marker"]
        
        # 画线、打点、填充区域
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=algo, color=color, marker=marker, markersize=8)
        ax.fill(angles, values, color=color, alpha=0.1) # 填充半透明颜色

    # 图例放到下方，尽量紧凑
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3,
              framealpha=0.9, fontsize=9, markerscale=0.6,
              handlelength=1.2, handletextpad=0.4, columnspacing=0.8)
    # 取消上方的大标题以节省空间
    # plt.title("Ablation Study: Comprehensive Performance Radar", size=16, y=1.1, fontweight='bold')
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_ablation_radar")
    plt.savefig(f"{out_path}.png")
    plt.savefig(f"{out_path}.pdf")
    print(f"Saved {out_path}.png and .pdf")
    plt.close()

if __name__ == "__main__":
    raw_data = load_radar_data()
    if raw_data:
        norm_data = normalize_data(raw_data)
        plot_radar(norm_data)
    else:
        print("No data found to plot!")
