import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Set up Times New Roman for all plots (IEEE TMC standard)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'legend.fontsize': 8.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Paths
DDPG_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(DDPG_DIR, "results", "sensitivity", "raw_data_K")
OUTPUT_DIR = os.path.join(DDPG_DIR, "results", "sensitivity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styles matching plot_results.py (with ESAC-PER label updated and all solid lines)
ALGO_STYLES = {
    'per_sac_cvx':     {'color': '#D62728', 'label': 'DARS (Ours)', 'ls': '-', 'marker': '*',  'ms': 8, 'lw': 2.2},
    'sac_cvx':         {'color': '#2CA02C', 'label': 'SAC+CVX',     'ls': '-', 'marker': 'o',  'ms': 6, 'lw': 1.8},
    'td3_cvx':         {'color': '#9467BD', 'label': 'TD3+CVX',     'ls': '-', 'marker': 's',  'ms': 6, 'lw': 1.8},
    'maddpg_cvx':      {'color': '#FF7F0E', 'label': 'MADDPG+CVX',  'ls': '-', 'marker': '^',  'ms': 6, 'lw': 1.8},
    'esac_per_cvx':    {'color': '#0000FF', 'label': 'ESAC-PER',    'ls': '-', 'marker': 'p',  'ms': 6, 'lw': 1.8},
}

# 1. Scheme A: Sensitivity Analysis Curves vs. K (1x3 subplots)
def plot_scheme_a():
    print("Generating Scheme A: Sensitivity curves vs K...")
    k_values = [2, 3, 4, 5, 6]
    algos = list(ALGO_STYLES.keys())
    
    # Structure: results[algo][K] = {reward, delay, rho}
    results = {algo: {k: {} for k in k_values} for algo in algos}
    
    for k in k_values:
        for algo in algos:
            filename = f"K_{k}_seed43_{algo}.npz"
            path = os.path.join(RAW_DATA_DIR, filename)
            if os.path.exists(path):
                data = np.load(path)
                # average of last 50 episodes of training
                r = np.mean(data['ep_reward'][-50:])
                delay = np.mean(data['avg_delay'][-50:])
                rho = np.mean(data['avg_rho'][-50:])
                results[algo][k] = {'reward': r, 'delay': delay, 'rho': rho}
            else:
                # K=6 is expected to be missing for td3, maddpg, and esac
                pass
                
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    metrics = [
        ('reward', 'Episode Reward', 'Episode Reward vs. $K$'),
        ('delay', 'Average Active-user Delay (s)', 'Average Delay (s) vs. $K$'),
        ('rho', r'Delay Reduction Ratio ($\rho$)', r'Delay Reduction Ratio ($\rho$) vs. $K$')
    ]
    
    for idx, (key, ylabel, title) in enumerate(metrics):
        ax = axes[idx]
        for algo in algos:
            style = ALGO_STYLES[algo]
            y_data = []
            x_data = []
            for k in k_values:
                if results[algo][k]:
                    x_data.append(k)
                    y_data.append(results[algo][k][key])
            ax.plot(x_data, y_data,
                    marker=style['marker'], color=style['color'],
                    linestyle=style['ls'], linewidth=style['lw'],
                    markersize=style['ms'], label=style['label'])
            
        ax.set_xlabel('Number of UAVs ($K$)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(k_values)
        ax.grid(True)
        ax.legend(loc='best', framealpha=0.9)
            
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR, f"fig_sens_K_1x3.{ext}"), dpi=300)
    plt.close(fig)
    print("Scheme A plot saved.")

# 2. Scheme D: Trajectory and Battery Plot for K = 3
def compute_phase_centroid(ue_positions, user_cluster_ids, t_start, t_end):
    centroids = np.zeros((4, 2))
    for c in range(4):
        mask = user_cluster_ids == c
        all_pos = []
        for t in range(t_start, min(t_end, len(ue_positions))):
            all_pos.append(ue_positions[t][mask])
        if all_pos:
            all_pos = np.concatenate(all_pos, axis=0)
            centroids[c] = np.mean(all_pos, axis=0)
    return centroids

def plot_scheme_d():
    print("Generating Scheme D: Trajectory and Battery curves for K = 3...")
    # Load environment with dynamic config K = 3, M = 30
    import config
    config.K = 3
    config.M = 30
    config.ACTION_DIM = 3 * config.K + config.M
    config.STATE_DIM = 6 * config.K + 4 * config.M + 1
    from UAV_env import UAVEnv
    
    # 2.1 UAV Trajectory Plot (1x3 subplots for 3 phases)
    dars_path = os.path.join(RAW_DATA_DIR, "K_3_seed43_per_sac_cvx.npz")
    if not os.path.exists(dars_path):
        print(f"Error: DARS results for K=3 not found at {dars_path}!")
        return
        
    dars_data = np.load(dars_path)
    traj = dars_data['last_trajectory'] # shape (50, 3, 2)
    battery = dars_data['last_battery'] # shape (50, 3)
    n_steps = len(traj)
    
    # Simulate user positions using seed 43
    np.random.seed(43)
    env = UAVEnv(use_cvx_alloc=True, use_hotspot=True)
    env.reset()
    ue_positions_history = []
    for t in range(n_steps):
        ue_positions_history.append(env.loc_ue.copy())
        env._update_ue()
        env.current_step += 1
    ue_positions = np.array(ue_positions_history)
    
    user_cluster_ids = env.user_cluster_ids
    
    # Phase settings
    _third = n_steps // 3
    phase_ranges = [(0, _third), (_third, 2*_third), (2*_third, n_steps)]
    phase_titles = [
        f'Phase I (t=0\u2013{_third-1})\nHot: C0+C1',
        f'Phase II (t={_third}\u2013{2*_third-1})\nHot: C1+C2',
        f'Phase III (t={2*_third}\u2013{n_steps-1})\nHot: C2+C3',
    ]
    hot_clusters = [{0, 1}, {1, 2}, {2, 3}]
    
    # Styling
    CLUSTER_COLORS = ['#e74c3c', '#27ae60', '#9b59b6', '#34495e']
    UAV_COLORS = ['#1E88E5', '#FB8C00', '#43A047']  # High contrast blue, orange, green
    UAV_NAMES = ['UAV 1', 'UAV 2', 'UAV 3']
    WCT_POSITIONS = env.wct_positions
    N_WCT = len(WCT_POSITIONS)
    L_C = 100.0
    GROUND_LENGTH = 500.0
    GROUND_WIDTH = 500.0
    E_BATTERY_INIT = 40000.0
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    plt.subplots_adjust(wspace=0.28)
    
    for p in range(3):
        ax = axes[p]
        t_start, t_end = phase_ranges[p]
        t_end = min(t_end, n_steps)
        hot_set = hot_clusters[p]
        
        ax.set_xlim(-15, GROUND_LENGTH + 15)
        ax.set_ylim(-15, GROUND_WIDTH + 15)
        ax.set_aspect('equal')
        ax.set_title(phase_titles[p], fontsize=10, pad=6)
        ax.set_xlabel('x (m)', fontsize=9)
        ax.set_ylabel('y (m)', fontsize=9)
        
        # Field Boundary
        ax.plot([0, GROUND_LENGTH, GROUND_LENGTH, 0, 0],
                [0, 0, GROUND_WIDTH, GROUND_WIDTH, 0],
                'k-', linewidth=0.5, alpha=0.4)
                
        # Phase Centroids
        centroids = compute_phase_centroid(ue_positions, user_cluster_ids, t_start, t_end)
        
        # Plot User Scatter (mid phase)
        t_mid = min((t_start + t_end) // 2, n_steps - 1)
        for c in range(4):
            mask = user_cluster_ids == c
            ue_c = ue_positions[t_mid][mask]
            is_hot = c in hot_set
            ax.scatter(ue_c[:, 0], ue_c[:, 1],
                       c=CLUSTER_COLORS[c] if is_hot else '#888888',
                       s=6 if is_hot else 4,
                       alpha=0.5 if is_hot else 0.4,
                       zorder=2, edgecolors='none')
                       
        # Plot Centroids
        for c in range(4):
            cx, cy = centroids[c]
            is_hot = c in hot_set
            if is_hot:
                ax.plot(cx, cy, '*', color=CLUSTER_COLORS[c],
                        markersize=12, markeredgecolor='k', markeredgewidth=0.6,
                        zorder=7)
                ax.annotate(f'C{c}', (cx, cy),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=7, fontweight='bold', color=CLUSTER_COLORS[c])
            else:
                ax.plot(cx, cy, 'o', color='#aaaaaa',
                        markersize=5, markeredgecolor='k', markeredgewidth=0.3,
                        alpha=0.4, zorder=4)
                ax.annotate(f'C{c}', (cx, cy),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=6, color='#999999')
                            
        # Plot WCT Charging Stations
        for j in range(N_WCT):
            cx, cy = WCT_POSITIONS[j, 0], WCT_POSITIONS[j, 1]
            circle = plt.Circle((cx, cy), L_C, color='#8e44ad',
                                 alpha=0.06, linestyle='--', linewidth=0.8,
                                 fill=True, zorder=1)
            ax.add_patch(circle)
            circle_edge = plt.Circle((cx, cy), L_C, color='#8e44ad',
                                      alpha=0.25, linestyle='--', linewidth=0.8,
                                      fill=False, zorder=2)
            ax.add_patch(circle_edge)
            ax.plot(cx, cy, 's', color='#8e44ad', markersize=6,
                    markeredgecolor='k', markeredgewidth=0.5, zorder=5)
            ax.annotate(f'WCT{j+1}', (cx, cy),
                        textcoords="offset points", xytext=(0, 6),
                        ha='center', fontsize=6, color='#8e44ad',
                        fontweight='bold')
                        
        # Plot UAV Flight Path
        for k in range(config.K):
            seg = traj[t_start:t_end, k, :]
            if len(seg) < 2:
                continue
                
            ax.plot(seg[:, 0], seg[:, 1], '-', color=UAV_COLORS[k],
                    linewidth=1.8, alpha=0.85, zorder=6)
                    
            # Alpha gradient steps
            for i in range(1, len(seg) - 1):
                alpha = 0.20 + 0.55 * i / max(len(seg) - 1, 1)
                ax.plot(seg[i, 0], seg[i, 1], 'o', color=UAV_COLORS[k],
                        markersize=2.2, alpha=alpha, zorder=6)
                        
            # Arrows for direction
            for i in range(0, len(seg) - 1, 4):
                dx = seg[i+1, 0] - seg[i, 0]
                dy = seg[i+1, 1] - seg[i, 1]
                if abs(dx) + abs(dy) > 1.0:
                    ax.annotate('', xy=(seg[i+1, 0], seg[i+1, 1]),
                                xytext=(seg[i, 0], seg[i, 1]),
                                arrowprops=dict(arrowstyle='->', color=UAV_COLORS[k],
                                                lw=1.2, mutation_scale=10),
                                zorder=6)
                                
            # Phase start (circle)
            ax.plot(seg[0, 0], seg[0, 1], 'o', color=UAV_COLORS[k],
                    markersize=6, markeredgecolor='k', markeredgewidth=0.8,
                    zorder=8)
            # Phase end (triangle)
            ax.plot(seg[-1, 0], seg[-1, 1], '^', color=UAV_COLORS[k],
                    markersize=7, markeredgecolor='k', markeredgewidth=0.8,
                    zorder=8)
                    
            # Draw charging events
            bseg = battery[t_start:t_end, k]
            charge_idx = np.where(np.diff(bseg) > 1e-6)[0]
            if len(charge_idx) > 0:
                ci = int(charge_idx[0] + 1)
                ax.plot(seg[ci, 0], seg[ci, 1], marker='P',
                        color=UAV_COLORS[k], markersize=8,
                        markeredgecolor='white', markeredgewidth=0.7,
                        zorder=9)
                        
        # Display phase battery percentage
        lines = []
        for k in range(config.K):
            b0 = battery[t_start, k] / E_BATTERY_INIT * 100
            b1 = battery[t_end - 1, k] / E_BATTERY_INIT * 100
            lines.append(f'UAV{k+1}: {b0:.0f}% \u2192 {b1:.0f}%')
        ax.text(0.02, 0.02, '\n'.join(lines), transform=ax.transAxes,
                ha='left', va='bottom', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='white', edgecolor='#cccccc',
                          alpha=0.85),
                zorder=20)
                
        ax.grid(True, alpha=0.12, linewidth=0.5)
        
    # Legend
    legend_elements = []
    for k in range(config.K):
        legend_elements.append(
            Line2D([0], [0], color=UAV_COLORS[k], lw=1.8,
                   marker='^', markersize=5, markeredgecolor='k',
                   label=UAV_NAMES[k]))
    legend_elements.append(
        Line2D([0], [0], marker='*', color='gray', markersize=9,
               markeredgecolor='k', linestyle='None', label='Hot centroid'))
    legend_elements.append(
        Line2D([0], [0], marker='s', color='#8e44ad', markersize=5,
               markeredgecolor='k', linestyle='None', label='WCT station'))
    legend_elements.append(
        Line2D([0], [0], color='#8e44ad', linestyle='--', linewidth=1.0,
               alpha=0.5, label=f'Charge range ($L_C$={int(L_C)}m)'))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='gray', markersize=5,
               linestyle='None', label='Phase start'))
    legend_elements.append(
        Line2D([0], [0], marker='^', color='gray', markersize=5,
               linestyle='None', label='Phase end'))
    legend_elements.append(
        Line2D([0], [0], marker='P', color='gray', markersize=6,
               linestyle='None', label='Charging event'))
               
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.04), framealpha=0.92,
               edgecolor='#cccccc', fontsize=7.5, ncol=len(legend_elements),
               handletextpad=0.45, columnspacing=1.0)
               
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f"fig_trajectory_K3.{ext}"), dpi=300)
    plt.close(fig)
    print("Trajectory subplots saved.")

    # 2.2 UAV Battery Level Comparison Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    algos = list(ALGO_STYLES.keys())
    
    for algo in algos:
        style = ALGO_STYLES[algo]
        path = os.path.join(RAW_DATA_DIR, f"K_3_seed43_{algo}.npz")
        if os.path.exists(path):
            data = np.load(path)
            # Battery data shape: (50, 3)
            bat = data['last_battery']
            # Compute average across K UAVs and convert to kJ
            avg_bat = np.mean(bat, axis=1) / 1000.0
            
            ax.plot(range(len(avg_bat)), avg_bat,
                    color=style['color'], linestyle=style['ls'],
                    linewidth=2.0, label=style['label'])
                    
            # Add shaded regions where battery charging occurs
            for t in range(1, len(avg_bat)):
                if avg_bat[t] > avg_bat[t-1]:
                    ax.axvspan(t-1, t, alpha=0.08, color=style['color'])
        else:
            print(f"Warning: File not found for battery comparison: {algo}")
            
    # Add 20% low battery threshold line
    ax.axhline(y=E_BATTERY_INIT * 0.2 / 1000.0, color='red',
               linestyle='--', alpha=0.7, linewidth=1.5, label='Low battery limit (20%)')
               
    ax.set_xlabel('Time Slot', fontsize=11)
    ax.set_ylabel('Average Battery Level (kJ)', fontsize=11)
    ax.set_title('UAV Battery Level Over Time ($K = 3$)', fontsize=12)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, E_BATTERY_INIT / 1000.0 + 2.0)
    ax.grid(True)
    ax.legend(loc='lower left', framealpha=0.9)
    
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f"fig_battery_K3.{ext}"), dpi=300)
    plt.close(fig)
    print("Battery level comparison plot saved.")

if __name__ == "__main__":
    plot_scheme_a()
    plot_scheme_d()
    print("All plots generated successfully!")
