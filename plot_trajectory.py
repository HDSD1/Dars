"""
UAV 轨迹可视化 — 加载训练好的模型, 画轨迹图
================================================
用法:
  python plot_trajectory.py --tag cvx_compare --algo per_ddpg_cvx
  python plot_trajectory.py --tag bat80k --algo ddpg_vanilla --seed 99
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (K, M, GROUND_LENGTH, GROUND_WIDTH, SLOT_NUM,
                    WCT_POSITIONS, L_C, E_BATTERY_INIT, N_WCT)
from UAV_env import UAVEnv
from state_normalization import StateNormalization

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 支持任意 K 架 UAV 的颜色 (红蓝绿紫橙青 ...)
_BASE_COLORS = ['#E53935', '#1E88E5', '#43A047', '#8E24AA', '#FB8C00', '#00ACC1',
                '#D81B60', '#5E35B1', '#00897B', '#F4511E']
UAV_COLORS = (_BASE_COLORS * ((K // len(_BASE_COLORS)) + 1))[:K]
UAV_NAMES = [f'UAV {k+1}' for k in range(K)]


def load_agent(algo, model_path):
    """根据算法名加载智能体."""
    if algo.startswith('per_ddpg'):
        from per_ddpg_agent import PERDDPGAgent
        agent = PERDDPGAgent()
    elif algo.startswith('ddpg_vanilla'):
        from ddpg_vanilla import VanillaDDPGAgent
        agent = VanillaDDPGAgent()
    elif algo.startswith('sac'):
        from sac_agent import SACAgent
        agent = SACAgent()
    elif algo.startswith('ppo'):
        from ppo_agent import PPOAgent
        agent = PPOAgent()
    elif algo.startswith('maddpg'):
        from maddpg_agent import MADDPGAgent
        agent = MADDPGAgent()
    elif algo.startswith('greedy'):
        from greedy_agent import GreedyAgent
        agent = GreedyAgent()
    else:
        from ddpg_agent import DDPGAgent
        agent = DDPGAgent()
        
    if not algo.startswith('greedy'):
        agent.load(model_path)
    return agent


def run_episode(agent, env, normalizer):
    """跑一个 episode, 收集轨迹数据."""
    state = env.reset()
    trajectories = [[] for _ in range(K)]
    battery_history = [[] for _ in range(K)]
    association_history = []
    ue_positions = []
    rewards = []

    for k in range(K):
        trajectories[k].append(env.loc_uav[k].copy())
        battery_history[k].append(env.e_battery_uav[k])
    ue_positions.append(env.loc_ue.copy())  # 初始用户分布

    for step in range(SLOT_NUM):
        s_norm = normalizer.normalize(state)
        action = agent.select_action(s_norm, noise_std=0.0)
        next_state, reward, done, info = env.step(action)

        for k in range(K):
            trajectories[k].append(env.loc_uav[k].copy())
            battery_history[k].append(env.e_battery_uav[k])

        association_history.append(info['association'].copy())
        ue_positions.append(env.loc_ue.copy())
        rewards.append(reward)
        state = next_state
        if done:
            break

    return trajectories, battery_history, association_history, ue_positions, rewards


def plot_trajectory(trajectories, battery_history, association_history,
                    ue_positions, env, save_dir, tag='', ue_snapshot='initial'):
    """画论文级轨迹图."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]

    # WCT 充电覆盖范围
    for j in range(N_WCT):
        circle = plt.Circle(env.wct_positions[j], L_C,
                           fill=True, facecolor='#FFF9C4', alpha=0.4,
                           linestyle='--', linewidth=1.5, edgecolor='#F9A825')
        ax.add_patch(circle)
        ax.plot(env.wct_positions[j, 0], env.wct_positions[j, 1],
                marker='^', markersize=14, color='#F9A825',
                markeredgecolor='black', markeredgewidth=1,
                zorder=10, label=f'WCT {j+1}' if j == 0 else '')

    # 用户位置
    if ue_positions:
        ue_show = ue_positions[0] if ue_snapshot == 'initial' else ue_positions[-1]
        ue_label = f'GU init ({M})' if ue_snapshot == 'initial' else f'GU final ({M})'
        ax.scatter(ue_show[:, 0], ue_show[:, 1],
                  c='#78909C', s=30, marker='o', alpha=0.7,
                  edgecolors='white', linewidths=0.5, zorder=5,
                  label=ue_label)

    # 关联线
    if association_history:
        last_assoc = association_history[-1]
        last_ue = ue_positions[-1]
        for i in range(M):
            k = last_assoc[i]
            if k >= 0:
                ax.plot([trajectories[k][-1][0], last_ue[i, 0]],
                       [trajectories[k][-1][1], last_ue[i, 1]],
                       color=UAV_COLORS[k], alpha=0.15, linewidth=0.8, zorder=3)

    # UAV 轨迹 (每步标记 + 连线)
    _BASE_MARKERS = ['x', '+', 'x', 'd', 's', '^', 'v', 'o', 'p', 'h']
    UAV_MARKERS = (_BASE_MARKERS * ((K // len(_BASE_MARKERS)) + 1))[:K]
    for k in range(K):
        path = np.array(trajectories[k])
        n_pts = len(path)

        # 连线 (颜色渐变)
        for t in range(n_pts - 1):
            alpha = 0.3 + 0.7 * t / max(n_pts - 2, 1)
            ax.plot([path[t, 0], path[t+1, 0]],
                    [path[t, 1], path[t+1, 1]],
                    color=UAV_COLORS[k], linewidth=1.8,
                    alpha=alpha, zorder=7)

        # 每步标记 (×)
        for t in range(1, n_pts - 1):
            ax.plot(path[t, 0], path[t, 1],
                    marker=UAV_MARKERS[k], markersize=7,
                    color=UAV_COLORS[k], markeredgewidth=1.5,
                    alpha=0.4 + 0.6 * t / n_pts, zorder=9)

        # 起点 (方块)
        ax.plot(path[0, 0], path[0, 1], marker='s', markersize=10,
                color=UAV_COLORS[k], markeredgecolor='black',
                markeredgewidth=1.5, zorder=11)
        # 终点 (星) + label
        ax.plot(path[-1, 0], path[-1, 1], marker='*', markersize=16,
                color=UAV_COLORS[k], markeredgecolor='black',
                markeredgewidth=1, zorder=11, label=UAV_NAMES[k])

        # 每 5 步标注步数
        for t in range(0, n_pts, max(1, 5)):
            ax.annotate(str(t), (path[t, 0], path[t, 1]),
                       fontsize=6, fontweight='bold',
                       color=UAV_COLORS[k], alpha=0.8,
                       xytext=(4, 4), textcoords='offset points',
                       zorder=12)

    ax.set_xlim(-20, GROUND_LENGTH + 20)
    ax.set_ylim(-20, GROUND_WIDTH + 20)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('UAV Trajectory', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    rect = patches.Rectangle((0, 0), GROUND_LENGTH, GROUND_WIDTH,
                              linewidth=2, edgecolor='black',
                              facecolor='none', linestyle='-')
    ax.add_patch(rect)

    # 电池
    ax2 = axes[1]
    for k in range(K):
        battery = np.array(battery_history[k]) / 1000
        ax2.plot(range(len(battery)), battery, color=UAV_COLORS[k],
                linewidth=2.5, label=UAV_NAMES[k])
        for t in range(1, len(battery)):
            if battery[t] > battery[t-1]:
                ax2.axvspan(t-1, t, alpha=0.1, color=UAV_COLORS[k])

    ax2.axhline(y=E_BATTERY_INIT * 0.2 / 1000, color='red',
                linestyle='--', alpha=0.5, label='Low battery (20%)')
    ax2.set_xlabel('Time Slot', fontsize=12)
    ax2.set_ylabel('Battery (kJ)', fontsize=12)
    ax2.set_title('Battery Level Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'trajectory{tag}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='UAV 轨迹可视化')
    parser.add_argument('--tag', type=str, default='cvx_compare',
                        help='实验标签 (从 results/<tag>/ 加载)')
    parser.add_argument('--algo', type=str, default='per_ddpg_cvx',
                        help='算法名')
    parser.add_argument('--seed', type=int, default=123,
                        help='随机种子')
    parser.add_argument('--ue_snapshot', type=str, default='initial',
                        choices=['initial', 'final'],
                        help='轨迹图中显示用户初始分布或末时隙分布')
    parser.add_argument('--cvx', action='store_true', default=True,
                        help='使用凸优化算力分配')
    args = parser.parse_args()

    np.random.seed(args.seed)
    use_cvx = 'cvx' in args.algo or args.cvx
    env = UAVEnv(use_cvx_alloc=use_cvx)
    normalizer = StateNormalization()

    # results/<tag>/<algo>.pth
    model_path = os.path.join(
        os.path.dirname(__file__), 'results', args.tag, f'{args.algo}.pth')
    print(f"Loading model: {model_path}")
    print(f"Algorithm: {args.algo}, CVX: {use_cvx}")

    agent = load_agent(args.algo, model_path)

    print("Running episode...")
    traj, battery, assoc, ue_pos, rewards = run_episode(
        agent, env, normalizer)

    print(f"Episode reward: {sum(rewards):.2f}")
    print(f"Steps: {len(rewards)}")

    save_dir = os.path.join(os.path.dirname(__file__), 'results', args.tag)
    os.makedirs(save_dir, exist_ok=True)
    plot_trajectory(traj, battery, assoc, ue_pos, env, save_dir,
                    tag=f'_{args.algo}', ue_snapshot=args.ue_snapshot)


if __name__ == '__main__':
    main()
