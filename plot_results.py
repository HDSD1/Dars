"""
统一可视化 — 自动加载同 tag 下所有算法结果做对比
==================================================
用法:
  python plot_results.py --tag exp        # 绘制 tag=exp 的所有算法对比
  python plot_results.py --tag exp --ablation  # 消融实验模式

文件结构:
  results/<tag>/<algo>.npz
  例如: results/exp/ddpg.npz, results/exp/per_ddpg.npz
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# IEEE TMC 论文级画图全局配置
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.unicode_minus': False,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# ============================================================
# 算法画图样式 (IEEE TMC 高对比度配色 + 区分性 marker)
# ============================================================
ALGO_STYLES = {
    # === 本文方法 (红色系, 突出) ===
    'per_sac_cvx':     {'color': '#D62728', 'label': 'DARS (Ours)', 'ls': '-',  'marker': '*',  'ms': 10, 'lw': 2.8, 'zorder': 10},
    'per_ddpg_cvx':    {'color': '#1F77B4', 'label': 'PER-TD3+CVX',       'ls': '-',  'marker': 'D',  'ms': 7,  'lw': 2.2, 'zorder': 9},
    # === DRL Baselines ===
    'att_sac_cvx':     {'color': '#17BECF', 'label': 'No_per',       'ls': '-',  'marker': 'P',  'ms': 8,  'lw': 2.2, 'zorder': 9},
    'sac_cvx':         {'color': '#2CA02C', 'label': 'SAC',           'ls': '--', 'marker': 'o',  'ms': 7,  'lw': 2.0, 'zorder': 8},
    'td3_cvx':         {'color': '#9467BD', 'label': 'TD3',           'ls': '--', 'marker': 's',  'ms': 7,  'lw': 2.0, 'zorder': 7},
    'maddpg_cvx':      {'color': '#FF7F0E', 'label': 'MADDPG',       'ls': '-.',  'marker': '^',  'ms': 7,  'lw': 2.0, 'zorder': 6},
    'ppo_cvx':         {'color': '#8C564B', 'label': 'PPO+',           'ls': '-.',  'marker': 'v',  'ms': 7,  'lw': 2.0, 'zorder': 6},
    'tmc_ppo_cvx':     {'color': '#8D6E63', 'label': 'TMC-2025 PPO',   'ls': '-.',  'marker': 'v',  'ms': 7,  'lw': 2.0, 'zorder': 7},
    'esac_per_cvx':    {'color': '#0000FF', 'label': 'TMC-2025 ESAC-PER', 'ls': '--', 'marker': 'p',  'ms': 7,  'lw': 2.0, 'zorder': 7},
    'ddpg_cvx':        {'color': '#E377C2', 'label': 'DDPG',          'ls': '-.',  'marker': 'H',  'ms': 7,  'lw': 1.8, 'zorder': 5},
    'knn_ddpg_cvx':    {'color': '#E91E63', 'label': 'KNN-DDPG',     'ls': '-',  'marker': 'd',  'ms': 7,  'lw': 2.0, 'zorder': 7},


    'tp_drl':          {'color': '#7B1FA2', 'label': 'TP-DRL',            'ls': '--', 'marker': 'X',  'ms': 7,  'lw': 2.0, 'zorder': 7},
    # === Heuristic Baselines ===
    'greedy_cvx':      {'color': '#4682B4', 'label': 'Greedy+CVX',        'ls': ':',  'marker': 'x',  'ms': 8,  'lw': 2.0, 'zorder': 4},
    'priority_greedy_cvx': {'color': '#6B8E23', 'label': 'PriorityGreedy+CVX', 'ls': ':', 'marker': '|', 'ms': 8, 'lw': 2.0, 'zorder': 4},
    'random':          {'color': '#BCBD22', 'label': 'Random+CVX',        'ls': ':',  'marker': '+',  'ms': 8,  'lw': 1.5, 'zorder': 3},
    'all_local':       {'color': '#7F7F7F', 'label': 'All Local',         'ls': '--', 'marker': '1',  'ms': 9,  'lw': 1.5, 'zorder': 2},
    'all_offload':     {'color': '#AEC7E8', 'label': 'All Offload',       'ls': '--', 'marker': '2',  'ms': 9,  'lw': 1.5, 'zorder': 2},
    'half_offload':    {'color': '#C49C94', 'label': 'Half Offload',      'ls': '--', 'marker': '3',  'ms': 9,  'lw': 1.5, 'zorder': 2},
    # === Legacy (兼容旧实验) ===
    'ddpg':            {'color': '#1F77B4', 'label': 'DDPG',              'ls': '-',  'marker': 'D',  'ms': 6,  'lw': 2.0, 'zorder': 5},
    'per_ddpg':        {'color': '#D62728', 'label': 'PER-DDPG',          'ls': '-',  'marker': '*',  'ms': 8,  'lw': 2.0, 'zorder': 8},
    # === 消融实验 ===
    'per_sac_cvx_no_surprise': {'color': '#AB47BC', 'label': 'w/o Dual-Priority', 'ls': '-.', 'marker': 'h', 'ms': 7, 'lw': 2.0, 'zorder': 7},
    'per_sac_no_cvx':          {'color': '#795548', 'label': 'per_sac_no_cvx',           'ls': '-.', 'marker': 'v', 'ms': 7, 'lw': 2.0, 'zorder': 6},
    'per_sac_cvx_1step':       {'color': '#FFA726', 'label': 'w/o N-step (n=1)',  'ls': ':', 'marker': 'x',  'ms': 8, 'lw': 2.0, 'zorder': 5},
    'no_atten':                {'color': '#263238', 'label': 'No Attention',     'ls': '--', 'marker': 's', 'ms': 7, 'lw': 2.0, 'zorder': 8},
}

# ============================================================
# 启发式基线算法集合 (用于在画图中做特殊处理或过滤)
# ============================================================
HEURISTIC_ALGOS = {'greedy_cvx', 'priority_greedy_cvx', 'random', 'all_local', 'all_offload', 'half_offload'}


def get_style(algo_name):
    """获取算法的绘图样式, 没注册的用默认."""
    default = {'color': '#7F7F7F', 'label': algo_name, 'ls': '-', 'marker': 'p', 'ms': 6, 'lw': 2.0, 'zorder': 1}
    return ALGO_STYLES.get(algo_name, default)



def smooth(data, window=50):
    if len(data) < window:
        return data, range(len(data))
    s = np.convolve(data, np.ones(window) / window, mode='valid')
    x = range(window - 1, len(data))
    return s, x


def load_results(tag, result_dir):
    """加载 results/<tag>/ 下所有 .npz 实验结果."""
    tag_dir = os.path.join(result_dir, tag)
    if not os.path.isdir(tag_dir):
        print(f"  Directory not found: {tag_dir}")
        return []
    pattern = os.path.join(tag_dir, "*.npz")
    files = sorted(glob.glob(pattern))
    results = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        # Check if it is a training history result or another type of analysis npz
        if 'ep_reward' not in data:
            print(f"  [Ignore] {os.path.basename(f)} (missing ep_reward, likely an analysis/ablation file)")
            continue
        algo = os.path.basename(f).replace(".npz", "")
        results.append({
            'algo': algo,
            'ep_rewards': data['ep_reward'],
            'avg_delays': data['avg_delay'],
            'avg_rhos': data['avg_rho'],
            'survival_steps': data.get('survival_steps', np.full_like(data['ep_reward'], 40)),
            'avg_q': data['avg_q'] if 'avg_q' in data else None,
            'avg_td_error': data['avg_td_error'] if 'avg_td_error' in data else None,
            'critic_loss': data['critic_loss'] if 'critic_loss' in data else None,
            'total_time': float(data['total_time']),
            'n_episodes': int(data['n_episodes']),
            'file': f,
        })
        print(f"  Loaded: {os.path.basename(f)} ({algo}, {len(data['ep_reward'])} eps)")
    return results


def _add_sparse_markers(ax, x, y, style, every=None):
    """在曲线上添加稀疏 marker (每 N 个点一个), 增强黑白可辨识度."""
    if every is None:
        every = max(1, len(x) // 8)  # 约 8 个 marker
    marker = style.get('marker', None)
    if marker:
        ax.plot(list(x)[::every], list(y)[::every],
                marker=marker, markersize=style.get('ms', 7),
                color=style['color'], linestyle='none',
                zorder=style.get('zorder', 5) + 1,
                markeredgecolor='white', markeredgewidth=0.5)


def plot_training_curves(results, save_dir, tag):
    """1. 四段式训练曲线对比 (Reward / Delay / ρ / Survival)."""
    # ★ 分离: 有学习过程的 DRL 算法 vs 无学习的 Heuristic baselines
    drl_results = [r for r in results if r['algo'] not in HEURISTIC_ALGOS]
    heuristic_results = [r for r in results if r['algo'] in HEURISTIC_ALGOS]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    metrics = [
        ('ep_rewards', 'Episode Reward', 'Training Reward'),
        ('avg_delays', 'Delay (s)', 'Average Active-user Delay'),
        ('avg_rhos', r'$\rho$', r'Delay Reduction Ratio ($\rho$)'),
        ('survival_steps', 'Steps', 'Survival Steps'),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        # ★ DRL 算法: 画完整收敛曲线
        for res in sorted(drl_results, key=lambda r: get_style(r['algo']).get('zorder', 1)):
            style = get_style(res['algo'])
            data = res[key]
            ax.plot(data, alpha=0.08, color=style['color'])
            s, x = smooth(data, 50)
            ax.plot(x, s, color=style['color'],
                    linewidth=style.get('lw', 2.0),
                    linestyle=style['ls'], label=style['label'],
                    zorder=style.get('zorder', 5))
            _add_sparse_markers(ax, x, s, style)

        # ★ Heuristic baselines: 画水平参考线 (取均值)
        for res in heuristic_results:
            style = get_style(res['algo'])
            mean_val = np.mean(res[key])
            ax.axhline(y=mean_val, color=style['color'],
                       linestyle=':', linewidth=1.2, alpha=0.7,
                       label=f"{style['label']} ({mean_val:.1f})",
                       zorder=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best', framealpha=0.9, edgecolor='gray', fontsize=8)
        ax.grid(True)

    plt.tight_layout(pad=2.0)
    for ext in ['png', 'pdf']:
        path = os.path.join(save_dir, f'{tag}_training_curves.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {tag}_training_curves.png/.pdf")


def plot_convergence(results, save_dir, tag):
    """2. ρ 收敛速度对比."""
    # ★ 分离: 有学习过程的 DRL 算法 vs 无学习的 Heuristic baselines
    drl_results = [r for r in results if r['algo'] not in HEURISTIC_ALGOS]
    heuristic_results = [r for r in results if r['algo'] in HEURISTIC_ALGOS]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # ★ DRL 算法: 画完整收敛曲线
    for res in sorted(drl_results, key=lambda r: get_style(r['algo']).get('zorder', 1)):
        style = get_style(res['algo'])
        s, x = smooth(res['avg_rhos'], 50)
        ax.plot(x, s, color=style['color'],
                linewidth=style.get('lw', 2.5),
                linestyle=style['ls'], label=style['label'],
                zorder=style.get('zorder', 5))
        _add_sparse_markers(ax, x, s, style)
        
    # ★ Heuristic baselines: 画水平参考线 (取均值)
    for res in heuristic_results:
        style = get_style(res['algo'])
        mean_val = np.mean(res['avg_rhos'])
        ax.axhline(y=mean_val, color=style['color'],
                   linestyle=':', linewidth=1.2, alpha=0.7,
                   label=f"{style['label']} ({mean_val:.3f})",
                   zorder=2)
                   
    ax.set_xlabel('Episode')
    ax.set_ylabel(r'$\rho$ (smoothed)')
    ax.set_title('Convergence Speed Comparison')
    ax.legend(framealpha=0.9, edgecolor='gray', fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(save_dir, f'{tag}_convergence.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {tag}_convergence.png/.pdf")


def plot_final_bars(results, save_dir, tag):
    """3. 最终性能柱状图 (用最后 50 集的平均值)."""
    fig, axes = plt.subplots(1, 4, figsize=(max(5 * len(results) / 2, 16), 5))

    for ax, (key, ylabel, title) in zip(axes, [
        ('ep_rewards', 'Episode Reward', 'Final Reward (last 50 eps)'),
        ('avg_rhos', 'ρ', 'Delay Reduction Ratio'),
        ('avg_delays', 'Delay (s)', 'Average Active-user Delay'),
        ('survival_steps', 'Steps', 'Survival Steps'),
    ]):
        names, means, stds, colors = [], [], [], []
        for res in results:
            style = get_style(res['algo'])
            data = res[key][-50:]
            names.append(style['label'])
            means.append(np.mean(data))
            stds.append(np.std(data))
            colors.append(style['color'])

        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                      color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        for i, (bar, m) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.15,
                    f'{m:.2f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(save_dir, f'{tag}_final_performance.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {tag}_final_performance.png/.pdf")


def plot_critic_diagnostics(results, save_dir, tag):
    """4. Critic 诊断曲线 (Q 值 / TD Error / Critic Loss)."""
    # 只绘制有诊断数据的算法
    diag_results = [r for r in results if r.get('avg_q') is not None]
    if not diag_results:
        print("  [Skip] No critic diagnostics data found (need retrain with new code)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ('avg_q', 'Average Q Value', 'Q Value Convergence'),
        ('avg_td_error', 'Mean |TD Error|', 'TD Error Convergence'),
        ('critic_loss', 'Critic Loss', 'Critic Loss Convergence'),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        for res in diag_results:
            style = get_style(res['algo'])
            data = res[key]
            if data is None:
                continue
            ax.plot(data, alpha=0.1, color=style['color'])
            s, x = smooth(data, 50)
            ax.plot(x, s, color=style['color'], linewidth=2,
                    linestyle=style['ls'], label=style['label'])
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(save_dir, f'{tag}_critic_diagnostics.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {tag}_critic_diagnostics.png/.pdf")


def print_summary_table(results):
    """4. 打印文本汇总表."""
    print(f"\n{'='*80}")
    print(f"  实验结果汇总 (最后 50 集平均)")
    print(f"{'='*80}")

    header = f"{'算法':<20} | {'Reward':>12} | {'ρ':>10} | {'Delay (s)':>12} | {'Survival':>9} | {'Time':>10}"
    print(header)
    print("-" * 90)

    for res in results:
        style = get_style(res['algo'])
        r = res['ep_rewards'][-50:]
        rho = res['avg_rhos'][-50:]
        d = res['avg_delays'][-50:]
        surv = res['survival_steps'][-50:]
        print(f"{style['label']:<20} | "
              f"{np.mean(r):8.2f}±{np.std(r):.2f} | "
              f"{np.mean(rho):6.3f}±{np.std(rho):.3f} | "
              f"{np.mean(d):8.3f}±{np.std(d):.3f} | "
              f"{np.mean(surv):5.1f}±{np.std(surv):.1f} | "
              f"{res['total_time']:.0f}s")

    # 如果有标准 DDPG 和 PER-DDPG, 算提升百分比
    ddpg_res = next((r for r in results if r['algo'] == 'ddpg'), None)
    per_res = next((r for r in results if r['algo'] == 'per_ddpg'), None)
    if ddpg_res is not None and per_res is not None:
        r_ddpg = np.mean(ddpg_res['ep_rewards'][-50:])
        r_per = np.mean(per_res['ep_rewards'][-50:])
        rho_ddpg = np.mean(ddpg_res['avg_rhos'][-50:])
        rho_per = np.mean(per_res['avg_rhos'][-50:])
        d_ddpg = np.mean(ddpg_res['avg_delays'][-50:])
        d_per = np.mean(per_res['avg_delays'][-50:])
        print("-" * 80)
        print(f"  PER-DDPG vs DDPG 提升:")
        print(f"    Reward: {(r_per/r_ddpg - 1)*100:+.1f}%")
        print(f"    ρ:      {(rho_per/rho_ddpg - 1)*100:+.1f}%")
        print(f"    Delay:  {(d_per/d_ddpg - 1)*100:+.1f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='统一可视化')
    parser.add_argument('--tag', type=str, default='exp',
                        help='实验标签 (加载 results/<tag>/)')
    parser.add_argument('--dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'results'),
                        help='结果根目录')
    parser.add_argument('--ablation', action='store_true',
                        help='消融实验模式 (特殊排列)')
    parser.add_argument('--drl-only', action='store_true',
                        help='只展示 DRL 类算法，不展示启发式算法')
    args = parser.parse_args()

    tag_dir = os.path.join(args.dir, args.tag)
    print(f"Loading results from: {tag_dir}")
    results = load_results(args.tag, args.dir)

    # per_ddpg_cvx (PER-TD3+CVX) 和 per_uos_rl 是同一个算法，按用户要求删除 per_ddpg_cvx (PER-TD3)
    results = [r for r in results if r['algo'] != 'per_ddpg_cvx']

    # 按照 Option B 方案，主图中剔除 per_sac_no_cvx 训练对比
    results = [r for r in results if r['algo'] != 'per_sac_no_cvx']

    if args.drl_only:
        results = [r for r in results if r['algo'] not in HEURISTIC_ALGOS]
        print(f"  [DRL Only] Filtered out heuristic algorithms. Remaining: {[r['algo'] for r in results]}")

    if len(results) == 0:
        print(f"No results found! Run train_unified.py first.")
        print(f"Example: python train_unified.py --algo ddpg --tag {args.tag}")
        return

    print(f"\nFound {len(results)} algorithms. Plotting...")
    # 图片也保存到 tag 子目录
    plot_training_curves(results, tag_dir, args.tag)
    plot_convergence(results, tag_dir, args.tag)
    plot_final_bars(results, tag_dir, args.tag)
    plot_critic_diagnostics(results, tag_dir, args.tag)
    print_summary_table(results)
    print("\nDone!")


if __name__ == '__main__':
    main()
