"""
论文级可视化: UAV 轨迹 + 热点质心 (1×3 子图, 3 阶段)
=====================================================
输出: PDF/PNG 静态图, 适合 IEEE 论文直接插入.

特点:
  - 3 个阶段子图, 热簇高亮, 冷簇灰化
  - UAV 轨迹带箭头 (方向) + 起终点标记
  - WCT 充电站 + 簇质心
  - Times New Roman 字体, IEEE 标准

用法:
  python plot_trajectory_paper.py                  # Greedy
  python plot_trajectory_paper.py --model results/fix_v1/per_ddpg_cvx.pth
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    K, M, GROUND_LENGTH, GROUND_WIDTH, SLOT_NUM, ACTION_DIM,
    N_CLUSTERS, WCT_POSITIONS, N_WCT, E_BATTERY_INIT, L_C,
)
from UAV_env import UAVEnv
from state_normalization import StateNormalization

# ---- 全局字体 (IEEE 标准) ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

UAV_COLORS = ['#e74c3c', '#27ae60', '#2980b9', '#e67e22']
UAV_NAMES  = ['UAV 1', 'UAV 2', 'UAV 3', 'UAV 4']
CLUSTER_NAMES = ['Cluster 0 (NW)', 'Cluster 1 (NE)',
                 'Cluster 2 (SE)', 'Cluster 3 (SW)']

_third = SLOT_NUM // 3
PHASE_RANGES = [(0, _third), (_third, 2*_third), (2*_third, SLOT_NUM)]
PHASE_TITLES = [
    f'Phase I (t=0\u2013{_third-1})\nHot: C0+C1',
    f'Phase II (t={_third}\u2013{2*_third-1})\nHot: C1+C2',
    f'Phase III (t={2*_third}\u2013{SLOT_NUM-1})\nHot: C2+C3',
]
HOT_CLUSTERS = [{0, 1}, {1, 2}, {2, 3}]


def run_episode(env, agent=None):
    norm = StateNormalization()
    state = env.reset()
    for t in range(SLOT_NUM):
        if agent is not None:
            s_norm = norm.normalize(state)
            action = agent.select_action(s_norm, noise_std=0.0)
        else:
            action = np.random.uniform(-1, 1, ACTION_DIM)
        state, reward, done, info = env.step(action)
        if done:
            break
    return {
        'trajectory': np.array(env.trajectory_log),
        'ue_positions': np.array(env.ue_position_log),
        'battery': np.array(env.battery_log),
        'n_steps': len(env.trajectory_log),
    }


def compute_phase_centroid(ue_positions, user_cluster_ids, t_start, t_end):
    """计算某阶段内每个簇的平均质心。"""
    centroids = np.zeros((N_CLUSTERS, 2))
    for c in range(N_CLUSTERS):
        mask = user_cluster_ids == c
        all_pos = []
        for t in range(t_start, min(t_end, len(ue_positions))):
            all_pos.append(ue_positions[t][mask])
        if all_pos:
            all_pos = np.concatenate(all_pos, axis=0)
            centroids[c] = np.mean(all_pos, axis=0)
    return centroids


def draw_phase(ax, phase_idx, traj, ue_positions, battery,
               user_cluster_ids, n_steps):
    t_start, t_end = PHASE_RANGES[phase_idx]
    t_end = min(t_end, n_steps)
    hot_set = HOT_CLUSTERS[phase_idx]

    ax.set_xlim(-15, GROUND_LENGTH + 15)
    ax.set_ylim(-15, GROUND_WIDTH + 15)
    ax.set_aspect('equal')
    ax.set_title(PHASE_TITLES[phase_idx], fontsize=9, fontweight='bold', pad=6)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # 场地边界
    ax.plot([0, GROUND_LENGTH, GROUND_LENGTH, 0, 0],
            [0, 0, GROUND_WIDTH, GROUND_WIDTH, 0],
            'k-', linewidth=0.5, alpha=0.4)

    # 计算该阶段的簇质心
    centroids = compute_phase_centroid(ue_positions, user_cluster_ids,
                                       t_start, t_end)

    # 用户散点 (取该阶段中间时刻)
    t_mid = min((t_start + t_end) // 2, n_steps - 1)
    for c in range(N_CLUSTERS):
        mask = user_cluster_ids == c
        ue_c = ue_positions[t_mid][mask]
        is_hot = c in hot_set
        ax.scatter(ue_c[:, 0], ue_c[:, 1],
                   c=UAV_COLORS[c] if is_hot else '#888888',
                   s=6 if is_hot else 4,
                   alpha=0.5 if is_hot else 0.4,
                   zorder=2, edgecolors='none')

    # 簇质心
    for c in range(N_CLUSTERS):
        cx, cy = centroids[c]
        is_hot = c in hot_set
        if is_hot:
            ax.plot(cx, cy, '*', color=UAV_COLORS[c],
                    markersize=14, markeredgecolor='k', markeredgewidth=0.6,
                    zorder=7)
            ax.annotate(f'C{c}', (cx, cy),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=7, fontweight='bold', color=UAV_COLORS[c])
        else:
            ax.plot(cx, cy, 'o', color='#aaaaaa',
                    markersize=5, markeredgecolor='k', markeredgewidth=0.3,
                    alpha=0.4, zorder=4)
            ax.annotate(f'C{c}', (cx, cy),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6, color='#999999')

    # WCT 充电站 + 覆盖范围
    for j in range(N_WCT):
        cx, cy = WCT_POSITIONS[j, 0], WCT_POSITIONS[j, 1]
        # 充电覆盖圆 (半径 L_C)
        circle = plt.Circle((cx, cy), L_C, color='#8e44ad',
                             alpha=0.08, linestyle='--', linewidth=0.8,
                             fill=True, zorder=1)
        ax.add_patch(circle)
        # 覆盖圆边界虚线
        circle_edge = plt.Circle((cx, cy), L_C, color='#8e44ad',
                                  alpha=0.3, linestyle='--', linewidth=0.8,
                                  fill=False, zorder=2)
        ax.add_patch(circle_edge)
        # 充电站标记
        ax.plot(cx, cy, 's', color='#8e44ad', markersize=7,
                markeredgecolor='k', markeredgewidth=0.5, zorder=5)
        ax.annotate(f'WCT{j}', (cx, cy),
                    textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=5.5, color='#8e44ad',
                    fontweight='bold')

    # UAV 轨迹
    for k in range(K):
        seg = traj[t_start:t_end, k, :]
        if len(seg) < 2:
            continue

        # 轨迹线
        ax.plot(seg[:, 0], seg[:, 1], '-', color=UAV_COLORS[k],
                linewidth=1.8, alpha=0.85, zorder=6)

        # 时间进度点: 越靠后越深, 用于区分阶段内部轨迹演化
        for i in range(1, len(seg) - 1):
            alpha = 0.20 + 0.55 * i / max(len(seg) - 1, 1)
            ax.plot(seg[i, 0], seg[i, 1], 'o', color=UAV_COLORS[k],
                    markersize=2.2, alpha=alpha, zorder=6)

        # 方向箭头 (每隔 4 步画一个)
        for i in range(0, len(seg) - 1, 4):
            dx = seg[i+1, 0] - seg[i, 0]
            dy = seg[i+1, 1] - seg[i, 1]
            if abs(dx) + abs(dy) > 1.0:
                ax.annotate('', xy=(seg[i+1, 0], seg[i+1, 1]),
                            xytext=(seg[i, 0], seg[i, 1]),
                            arrowprops=dict(arrowstyle='->', color=UAV_COLORS[k],
                                            lw=1.2, mutation_scale=10),
                            zorder=6)

        # 阶段起点 (圆)
        ax.plot(seg[0, 0], seg[0, 1], 'o', color=UAV_COLORS[k],
                markersize=7, markeredgecolor='k', markeredgewidth=0.8,
                zorder=8)
        # 阶段终点 (三角)
        ax.plot(seg[-1, 0], seg[-1, 1], '^', color=UAV_COLORS[k],
                markersize=8, markeredgecolor='k', markeredgewidth=0.8,
                zorder=8)

        # 若该阶段电量上升, 标注一次充电事件
        if battery is not None and len(battery) >= t_end:
            bseg = battery[t_start:t_end, k]
            charge_idx = np.where(np.diff(bseg) > 1e-6)[0]
            if len(charge_idx) > 0:
                ci = int(charge_idx[0] + 1)
                ax.plot(seg[ci, 0], seg[ci, 1], marker='P',
                        color=UAV_COLORS[k], markersize=8,
                        markeredgecolor='white', markeredgewidth=0.7,
                        zorder=9)

    # 阶段电量摘要，帮助说明服务-充电权衡
    if battery is not None and len(battery) >= t_end and t_end > t_start:
        lines = []
        for k in range(K):
            b0 = battery[t_start, k] / E_BATTERY_INIT * 100
            b1 = battery[t_end - 1, k] / E_BATTERY_INIT * 100
            lines.append(f'UAV{k+1}: {b0:.0f}%→{b1:.0f}%')
        ax.text(0.02, 0.02, '\n'.join(lines), transform=ax.transAxes,
                ha='left', va='bottom', fontsize=6.5,
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='white', edgecolor='#cccccc',
                          alpha=0.82),
                zorder=20)

    ax.grid(True, alpha=0.12, linewidth=0.5)


def create_figure(data, env, save_path='trajectory_phases.pdf'):
    traj = data['trajectory']
    ue_pos = data['ue_positions']
    battery = data['battery']
    n_steps = data['n_steps']
    user_cluster_ids = env.user_cluster_ids

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    plt.subplots_adjust(wspace=0.28)

    for p in range(3):
        draw_phase(axes[p], p, traj, ue_pos, battery,
                   user_cluster_ids, n_steps)

    # 图例 (全局放在图外, 避免遮挡第三阶段轨迹)
    legend_elements = []
    for k in range(K):
        legend_elements.append(
            Line2D([0], [0], color=UAV_COLORS[k], lw=2,
                   marker='^', markersize=6, markeredgecolor='k',
                   label=UAV_NAMES[k]))
    legend_elements.append(
        Line2D([0], [0], marker='*', color='gray', markersize=10,
               markeredgecolor='k', linestyle='None', label='Hot centroid'))
    legend_elements.append(
        Line2D([0], [0], marker='s', color='#8e44ad', markersize=6,
               markeredgecolor='k', linestyle='None', label='WCT station'))
    legend_elements.append(
        Line2D([0], [0], color='#8e44ad', linestyle='--', linewidth=1.2,
               alpha=0.5, label=f'Charge range ($L_C$={int(L_C)}m)'))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='gray', markersize=5,
               linestyle='None', label='Phase start'))
    legend_elements.append(
        Line2D([0], [0], marker='^', color='gray', markersize=6,
               linestyle='None', label='Phase end'))
    legend_elements.append(
        Line2D([0], [0], marker='P', color='gray', markersize=7,
               linestyle='None', label='Charging event'))

    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), framealpha=0.92,
               edgecolor='#cccccc', fontsize=7, ncol=min(4, len(legend_elements)),
               handletextpad=0.45, columnspacing=1.0)

    # 保存
    for ext in ['pdf', 'png']:
        out = save_path.rsplit('.', 1)[0] + '.' + ext
        fig.savefig(out)
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--algo', type=str, default='greedy',
                        choices=['greedy', 'random'])
    parser.add_argument('--output', type=str, default='trajectory_phases.pdf')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    env = UAVEnv(use_cvx_alloc=True, use_hotspot=True)
    agent = None

    if args.model and os.path.exists(args.model):
        model_name = os.path.basename(args.model).lower()
        if 'per_sac' in model_name:
            from DARS import PERSACAgent
            agent = PERSACAgent()
        elif 'sac' in model_name:
            from sac_agent import SACAgent
            agent = SACAgent()
        elif 'per_ddpg' in model_name:
            from per_ddpg_agent import PERDDPGAgent
            agent = PERDDPGAgent()
        elif 'ddpg' in model_name:
            from ddpg_agent import DDPGAgent
            agent = DDPGAgent()
        else:
            from per_ddpg_agent import PERDDPGAgent
            agent = PERDDPGAgent()
        agent.load(args.model)
        print(f"Model: {args.model} ({type(agent).__name__})")
    elif args.algo == 'greedy':
        from greedy_agent import GreedyAgent
        agent = GreedyAgent()
        print("Policy: Greedy")

    data = run_episode(env, agent)
    print(f"Episode: {data['n_steps']} steps")
    create_figure(data, env, save_path=args.output)


if __name__ == '__main__':
    main()
