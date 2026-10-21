"""
多 UAV-MEC 可视化工具 (V3, 含 WCT)
====================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import (
    K, M, GROUND_LENGTH, GROUND_WIDTH, SLOT_NUM,
    N_WCT, WCT_POSITIONS, L_C,
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
UAV_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336',
              '#00BCD4', '#795548', '#607D8B']
UE_COLORS  = ['#E91E63', '#00BCD4', '#8BC34A', '#FF5722', '#673AB7',
              '#009688', '#FF9800', '#3F51B5']


def smooth(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


# ============================================================
# 1. 训练曲线
# ============================================================
def plot_training_curves(ep_rewards, avg_delays, avg_rhos, save_dir=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, label, color, title in [
        (axes[0], ep_rewards, 'Reward', ('steelblue','navy'), 'Training Reward'),
        (axes[1], avg_delays, 'Delay (s)', ('salmon','darkred'), 'Average Delay'),
        (axes[2], avg_rhos, 'rho', ('lightgreen','darkgreen'), 'Delay Reduction Ratio'),
    ]:
        ax.plot(data, alpha=0.2, color=color[0])
        if len(data) > 50:
            ax.plot(range(49, len(data)), smooth(data, 50),
                    color=color[1], linewidth=2, label='Smooth(50)')
        ax.set_xlabel('Episode'); ax.set_ylabel(label)
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_curves.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 2. 轨迹图 (含 WCT)
# ============================================================
def plot_uav_trajectory(env, agent, normalizer, save_dir=None):
    state = env.reset()
    ue_init = env.loc_ue.copy()

    for step in range(SLOT_NUM):
        s_norm = normalizer.normalize(state)
        action = agent.select_action(s_norm, noise_std=0.0)
        state, _, _, _ = env.step(action)

    trajectories = np.array(env.trajectory_log)

    fig, ax = plt.subplots(figsize=(12, 12))

    # WCT 充电站 + 覆盖圆
    for j in range(N_WCT):
        cx, cy = WCT_POSITIONS[j]
        circle = plt.Circle((cx, cy), L_C, color='gold',
                             alpha=0.15, linestyle='--', linewidth=1.5)
        ax.add_patch(circle)
        ax.plot(cx, cy, 'P', color='gold', markersize=18,
                markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        ax.annotate(f'WCT-{j}', (cx+3, cy+3), fontsize=11,
                    fontweight='bold', color='darkgoldenrod')

    # UE (只标注前几个, 避免拥挤)
    max_ue_label = min(M, 20)
    for i in range(M):
        c = UE_COLORS[i % len(UE_COLORS)]
        ax.scatter(ue_init[i, 0], ue_init[i, 1], c=c, marker='^',
                   s=80, zorder=4, alpha=0.7, edgecolors='gray', linewidth=0.3)
        if i < max_ue_label:
            ax.annotate(f'{i}', (ue_init[i, 0]+1.5, ue_init[i, 1]+1.5),
                        fontsize=7, alpha=0.6)

    # UAV 轨迹
    for k in range(K):
        traj = trajectories[:, k, :]
        color = UAV_COLORS[k % len(UAV_COLORS)]
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color,
                linewidth=1.5, alpha=0.5)
        # 箭头
        for t in range(1, len(traj)):
            dx = traj[t, 0] - traj[t-1, 0]
            dy = traj[t, 1] - traj[t-1, 1]
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                ax.annotate('', xy=(traj[t, 0], traj[t, 1]),
                            xytext=(traj[t-1, 0], traj[t-1, 1]),
                            arrowprops=dict(arrowstyle='->', color=color,
                                            lw=1.0, alpha=0.5))
        # 步号
        for t in range(0, len(traj), 5):
            ax.plot(traj[t, 0], traj[t, 1], 'o', color=color,
                    markersize=5, zorder=7)
            ax.annotate(f'{t}', (traj[t, 0]+1.5, traj[t, 1]+1.5),
                        fontsize=7, color=color)
        # 起点/终点
        ax.plot(traj[0, 0], traj[0, 1], 's', color=color,
                markersize=14, zorder=8, label=f'UAV-{k} start')
        ax.plot(traj[-1, 0], traj[-1, 1], '*', color=color,
                markersize=18, zorder=8, markeredgecolor='black',
                markeredgewidth=0.5)

    ax.set_xlim(-10, GROUND_LENGTH + 10)
    ax.set_ylim(-10, GROUND_WIDTH + 10)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'UAV Trajectories ({SLOT_NUM} steps) | '
                 f'P=WCT, ^=UE, []->*=UAV path', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'uav_trajectory.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()
    return trajectories


# ============================================================
# 3. 关联快照
# ============================================================
def plot_association_snapshot(env, agent, normalizer, step_idx=0, save_dir=None):
    state = env.reset()
    for step in range(min(step_idx + 1, SLOT_NUM)):
        s_norm = normalizer.normalize(state)
        action = agent.select_action(s_norm, noise_std=0.0)
        state, _, _, info = env.step(action)

    association = info['association']
    uav_locs = env.loc_uav
    ue_locs = env.loc_ue

    fig, ax = plt.subplots(figsize=(10, 10))

    # WCT
    for j in range(N_WCT):
        cx, cy = WCT_POSITIONS[j]
        circle = plt.Circle((cx, cy), L_C, color='gold',
                             alpha=0.1, linestyle='--')
        ax.add_patch(circle)
        ax.plot(cx, cy, 'P', color='gold', markersize=16,
                markeredgecolor='black', zorder=10)

    for k in range(K):
        ax.plot(uav_locs[k, 0], uav_locs[k, 1], 'D',
                color=UAV_COLORS[k], markersize=14, zorder=6,
                label=f'UAV-{k}')

    for i in range(M):
        k = association[i]
        c = UAV_COLORS[k % len(UAV_COLORS)] if k >= 0 else 'gray'
        ax.plot(ue_locs[i, 0], ue_locs[i, 1], '^', color=c,
                markersize=8, zorder=5, alpha=0.8)
        if k >= 0:
            ax.plot([uav_locs[k, 0], ue_locs[i, 0]],
                    [uav_locs[k, 1], ue_locs[i, 1]],
                    '--', color=c, alpha=0.3, linewidth=0.5)

    ax.set_xlim(-10, GROUND_LENGTH + 10)
    ax.set_ylim(-10, GROUND_WIDTH + 10)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(f'UAV-UE Association (Step {step_idx})')
    ax.legend(loc='upper right'); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'association_snapshot.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 4. Episode 详情 (含充电)
# ============================================================
def plot_episode_details(env, agent, normalizer, save_dir=None):
    state = env.reset()
    for step in range(SLOT_NUM):
        s_norm = normalizer.normalize(state)
        action = agent.select_action(s_norm, noise_std=0.0)
        state, _, _, _ = env.step(action)

    delays = np.array(env.delay_log)
    rhos = np.array(env.rho_log)
    rewards = np.array(env.reward_log)
    batteries = np.array(env.battery_log)
    charges = np.array(env.charge_log)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    steps = np.arange(SLOT_NUM)

    # (a) 延迟 — 取各 UE 平均
    ax = axes[0, 0]
    ax.plot(steps, delays.mean(axis=1), '-s', markersize=3, color='darkred')
    ax.fill_between(steps, delays.min(axis=1), delays.max(axis=1),
                     alpha=0.15, color='salmon')
    ax.set_xlabel('Step'); ax.set_ylabel('Delay (s)')
    ax.set_title('Avg Delay per Step (shaded: min-max)'); ax.grid(True, alpha=0.3)

    # (b) rho — 取各 UE 平均
    ax = axes[0, 1]
    ax.plot(steps, rhos.mean(axis=1), '-s', markersize=3, color='darkgreen')
    ax.fill_between(steps, rhos.min(axis=1), rhos.max(axis=1),
                     alpha=0.15, color='lightgreen')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step'); ax.set_ylabel('rho')
    ax.set_title('Avg Delay Reduction Ratio'); ax.grid(True, alpha=0.3)

    # (c) 奖励
    ax = axes[1, 0]
    ax.plot(steps, rewards, '-s', color='navy', markersize=4)
    ax.set_xlabel('Step'); ax.set_ylabel('Reward')
    ax.set_title('Step Reward'); ax.grid(True, alpha=0.3)

    # (d) 电池
    ax = axes[1, 1]
    for k in range(K):
        ax.plot(steps, batteries[:, k] / 1000, '-',
                color=UAV_COLORS[k], linewidth=2, label=f'UAV-{k}')
    ax.set_xlabel('Step'); ax.set_ylabel('Battery (kJ)')
    ax.set_title('UAV Battery (with WCT charging)'); ax.legend()
    ax.grid(True, alpha=0.3)

    # (e) 充电功率
    ax = axes[2, 0]
    for k in range(K):
        ax.plot(steps, charges[:, k], '-o', markersize=3,
                color=UAV_COLORS[k], label=f'UAV-{k}')
    ax.set_xlabel('Step'); ax.set_ylabel('Charge Power (W)')
    ax.set_title('WCT Charging Power Received'); ax.legend()
    ax.grid(True, alpha=0.3)

    # (f) 净能耗 (能耗 vs 充电)
    ax = axes[2, 1]
    hover_p = 168.49  # P0+PI
    for k in range(K):
        energy_loss = np.diff(np.concatenate([[batteries[0, k]], batteries[:, k]]))
        ax.plot(steps, -energy_loss / 1000, '-',
                color=UAV_COLORS[k], linewidth=1.5, label=f'UAV-{k}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Net Energy Change (kJ)')
    ax.set_title('Energy Balance per Step (>0 = drain, <0 = gain)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('Single Episode Analysis (with WCT Charging)', fontsize=14)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'episode_details.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 5. 基线对比
# ============================================================
def plot_baseline_comparison(env, agent, normalizer, save_dir=None):
    def run_episode(policy_fn):
        state = env.reset()
        total_reward = 0
        for step in range(SLOT_NUM):
            action = policy_fn(state)
            state, reward, _, _ = env.step(action)
            total_reward += reward
        return total_reward

    n_eval = 20
    results = {}

    rewards = []
    for _ in range(n_eval):
        def ddpg_policy(s, a=agent, n=normalizer):
            return a.select_action(n.normalize(s), noise_std=0.0)
        rewards.append(run_episode(ddpg_policy))
    results['DDPG'] = rewards

    rewards = []
    for _ in range(n_eval):
        def local_policy(s):
            a = np.random.uniform(-1, 1, env.action_dim)
            a[2*K:] = -1.0
            return a
        rewards.append(run_episode(local_policy))
    results['All Local'] = rewards

    rewards = []
    for _ in range(n_eval):
        def full_policy(s):
            a = np.random.uniform(-1, 1, env.action_dim)
            a[2*K:] = 1.0
            return a
        rewards.append(run_episode(full_policy))
    results['All Offload'] = rewards

    rewards = []
    for _ in range(n_eval):
        def random_policy(s):
            return np.random.uniform(-1, 1, env.action_dim)
        rewards.append(run_episode(random_policy))
    results['Random'] = rewards

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = list(results.keys())
    means = [np.mean(results[l]) for l in labels]
    stds = [np.std(results[l]) for l in labels]
    colors = ['#2196F3', '#F44336', '#FF9800', '#9E9E9E']

    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Episode Reward')
    ax.set_title('Policy Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(stds)*0.1,
                f'{m:.1f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'baseline_comparison.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 6. 一键生成
# ============================================================
def plot_all(ep_rewards, avg_delays, avg_rhos,
             env, agent, normalizer, save_dir=None):
    print("\n[Viz] Training curves...")
    plot_training_curves(ep_rewards, avg_delays, avg_rhos, save_dir=save_dir)
    print("[Viz] UAV trajectory (with WCT)...")
    plot_uav_trajectory(env, agent, normalizer, save_dir=save_dir)
    print("[Viz] Association snapshot...")
    plot_association_snapshot(env, agent, normalizer, step_idx=5, save_dir=save_dir)
    print("[Viz] Episode details (with charging)...")
    plot_episode_details(env, agent, normalizer, save_dir=save_dir)
    print("[Viz] Baseline comparison...")
    plot_baseline_comparison(env, agent, normalizer, save_dir=save_dir)
    print("[Viz] Done!")
