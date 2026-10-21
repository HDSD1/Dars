"""
自动化参数敏感性分析脚本 (V4 — 支持分方法运行 + 单独 rho 曲线画图 + 6算法测试)
===========================================================================
实验参数:
  1. 用户密度 M = [15, 20, 25, 30, 35]
  2. UAV 数量 K = [2, 3, 4, 5, 6]
  3. 初始电池电量 E = [20000.0, 25000.0, 30000.0, 35000.0, 40000.0]
  4. 热冷比 HOTSPOT_HOT_PROB = [0.6, 0.7, 0.8, 0.9, 1.0]

算法列表:
  - DARS (Ours): per_sac_cvx
  - SAC: sac_cvx
  - Random: random
  - TD3: td3_cvx
  - MADDPG: maddpg_cvx
  - ESAC-PER: esac_per_cvx

核心设计:
  - 【分方法运行保护】：提取数据时，同时检查临时目录和 results/sensitivity/raw_data/ 已归档目录。
    即使你多次分批、分算法、单个算法运行，先前跑完的算法数据也绝对不会丢失，会自动累积合并到同一图表中！
  - 【单独 rho 曲线输出】：除 2x2 综合学术图外，新增单独输出的 Average Delay Reduction Ratio (rho) 专属高清图，直接用于论文排版。
  - 【多种子/单种子兼容】：通过 --seeds 参数传入单个种子（如 --seeds 42）即可实现不做多种子平均的快速测试。
"""

import os
import sys
import re
import shutil
import argparse
import subprocess
import signal
import atexit
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

# 目标文件路径
DDPG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(DDPG_DIR, "config.py")
BACKUP_PATH = os.path.join(DDPG_DIR, "config.py.bak")
SENS_DIR = os.path.join(DDPG_DIR, "results", "sensitivity")

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 敏感性实验参数配置
# ============================================================
EXPERIMENTS = {
    'M': {
        'var': 'M',
        'values': [15, 20, 25, 30, 35],
        'xlabel': 'Number of Users ($M$)',
        'coupled': {},
    },
    'K': {
        'var': 'K',
        'values': [2, 3, 4, 5, 6],
        'xlabel': 'Number of UAVs ($K$)',
        'coupled': {},
    },
    'E': {
        'var': 'E_BATTERY_INIT',
        'values': [20000.0, 25000.0, 30000.0, 35000.0, 40000.0],
        'xlabel': 'Battery Capacity $E_{max}$ (J)',
        'coupled': {},
    },
    'HOT': {
        'var': 'HOTSPOT_HOT_PROB',
        'values': [0.6, 0.7, 0.8, 0.9, 1.0],
        'xlabel': 'Hotspot Task Arrival Prob ($p_{hot}$)',
        'coupled': {
            'HOTSPOT_COLD_PROB': lambda hot: round(1.0 - hot, 2)
        },
    },
    'TASK': {
        'var': 'TASK_MAX',
        'values': [2097152, 2621440, 3145728, 3670016, 4194304],
        'xlabel': 'Maximum Task Size $D_{max}$ (bits)',
        'coupled': {},
    },
}

# 默认包含的 6 个 DRL 对比算法 (支持顺序)
TARGET_ALGOS = [
    'per_sac_cvx',      # DARS (Ours)
    'sac_cvx',          # SAC
    'random',           # Random
    'td3_cvx',          # TD3
    'maddpg_cvx',       # MADDPG
    'esac_per_cvx',     # ESAC-PER
]

# 学术画图样式
STYLES = {
    'per_sac_cvx':  {'color': '#D62728', 'marker': '*', 'ms': 12, 'label': 'DARS (Ours)'},
    'tp_drl':       {'color': '#7B1FA2', 'marker': 'X', 'ms': 8,  'label': 'TP-DRL'},
    'sac_cvx':      {'color': '#2CA02C', 'marker': 'o', 'ms': 8,  'label': 'SAC+CVX'},
    'random':       {'color': '#BCBD22', 'marker': '+', 'ms': 10, 'label': 'Random+CVX'},
    'td3_cvx':      {'color': '#9467BD', 'marker': 'D', 'ms': 8,  'label': 'TD3+CVX'},
    'maddpg_cvx':   {'color': '#FF7F0E', 'marker': '^', 'ms': 8,  'label': 'MADDPG+CVX'},
    'esac_per_cvx': {'color': '#0000FF', 'marker': 'p', 'ms': 8,  'label': 'TMC-2025 ESAC-PER'},
}


# ============================================================
# config.py 操作
# ============================================================
def backup_config():
    """备份 config.py"""
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(CONFIG_PATH, BACKUP_PATH)
        print(f"[Info] Backup config.py -> config.py.bak")


def restore_config():
    """恢复 config.py"""
    if os.path.exists(BACKUP_PATH):
        shutil.copy2(BACKUP_PATH, CONFIG_PATH)
        os.remove(BACKUP_PATH)
        print(f"[Info] Restored config.py from backup.")


def modify_config(var_name, new_value):
    """用正则替换 config.py 中的变量赋值"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = rf"^({var_name}\s*=\s*)[0-9]+\.?[0-9]*"
    replacement = rf"\g<1>{new_value}"

    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)

    if count == 0:
        print(f"[Error] Variable '{var_name}' not found in config.py!")
        return False

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"  -> Modified {var_name} = {new_value}")
    return True


# ============================================================
# 训练执行
# ============================================================
def run_training_for_all_algos(exp_name, value, episodes, algos, seeds):
    """调用 train_unified.py 执行所有种子和算法的训练 (带异常保护)"""
    failed = []

    for seed in seeds:
        tag = f"sens_{exp_name}_{value}_seed{seed}"
        for algo in algos:
            cmd = [
                sys.executable, "train_unified.py",
                "--algo", algo,
                "--episodes", str(episodes),
                "--seed", str(seed),
                "--tag", tag
            ]
            print(f"\n{'='*72}")
            print(f"[{exp_name}] {algo} | Seed={seed} | {exp_name}={value} | {episodes} eps")
            print(f"{'='*72}")

            try:
                process = subprocess.Popen(cmd, cwd=DDPG_DIR)
                process.wait()

                if process.returncode != 0:
                    print(f"[Warning] Training failed for {algo} at {exp_name}={value}, Seed={seed} "
                          f"(exit code: {process.returncode}), skipping...")
                    failed.append((algo, seed))
            except Exception as e:
                print(f"[Error] Exception running {algo} for Seed={seed}: {e}, skipping...")
                failed.append((algo, seed))

    if failed:
        print(f"\n[Summary] Failed runs for {exp_name}={value}: {failed}")
    return True


# ============================================================
# 结果提取与多种子平均 (支持分方法运行、累加合并)
# ============================================================
def extract_average_performance(exp_name, val, algos, seeds):
    """
    提取该参数点下所有成功运行的种子结果，并计算均值。
    支持分方法运行的逻辑：同时检索当前的临时结果目录以及以前跑完已归档的 raw_data 目录。
    """
    algo_metrics = {algo: {'rewards': [], 'rhos': [], 'delays': [], 'survivals': []} for algo in algos}
    
    for seed in seeds:
        for algo in algos:
            # 1. 优先尝试从本次训练产生的临时目录加载
            tag = f"sens_{exp_name}_{val}_seed{seed}"
            src_dir = os.path.join(DDPG_DIR, "results", tag)
            npz_file = os.path.join(src_dir, f"{algo}.npz")
            
            # 2. 如果临时目录不存在（比如之前分方法运行过，临时目录已删，但数据已归档），则从归档目录加载
            if not os.path.exists(npz_file):
                archive_dir = os.path.join(SENS_DIR, "raw_data")
                npz_file = os.path.join(archive_dir, f"{exp_name}_{val}_seed{seed}_{algo}.npz")
                
            if os.path.exists(npz_file):
                try:
                    data = np.load(npz_file)
                    r = np.mean(data['ep_reward'][-50:])
                    rho = np.mean(data['avg_rho'][-50:])
                    delay = np.mean(data['avg_delay'][-50:])
                    surv_key = 'survival_steps' if 'survival_steps' in data else None
                    surv = np.mean(data[surv_key][-50:]) if surv_key else 40.0
                    
                    algo_metrics[algo]['rewards'].append(float(r))
                    algo_metrics[algo]['rhos'].append(float(rho))
                    algo_metrics[algo]['delays'].append(float(delay))
                    algo_metrics[algo]['survivals'].append(float(surv))
                except Exception as e:
                    print(f"  [Warning] Failed to load {npz_file}: {e}")

    # 计算平均值 (无论是多随机种子还是单随机种子)
    perf_dict = {}
    for algo in algos:
        m = algo_metrics[algo]
        if len(m['rewards']) > 0:
            perf_dict[algo] = {
                'reward': float(np.mean(m['rewards'])),
                'rho': float(np.mean(m['rhos'])),
                'delay': float(np.mean(m['delays'])),
                'survival': float(np.mean(m['survivals'])),
                'seeds_count': len(m['rewards'])
            }
    return perf_dict


def consolidate_and_clean(exp_name, val, algos, seeds):
    """将所有临时目录的 npz 拷贝到 raw_data 归档目录，然后清理临时目录，保持工作区干净"""
    dst_dir = os.path.join(SENS_DIR, "raw_data")
    os.makedirs(dst_dir, exist_ok=True)
    
    for seed in seeds:
        tag = f"sens_{exp_name}_{val}_seed{seed}"
        src_dir = os.path.join(DDPG_DIR, "results", tag)
        if not os.path.exists(src_dir):
            continue
            
        for algo in algos:
            src_file = os.path.join(src_dir, f"{algo}.npz")
            if os.path.exists(src_file):
                dst_file = os.path.join(dst_dir, f"{exp_name}_{val}_seed{seed}_{algo}.npz")
                shutil.copy2(src_file, dst_file)
                
        try:
            shutil.rmtree(src_dir)
            print(f"  [Cleaned] Temporary folder 'results/{tag}' deleted. Raw data consolidated.")
        except Exception as e:
            print(f"  [Warning] Failed to delete 'results/{tag}': {e}")


def save_summary_data(exp_name, results_dict, seeds):
    """将所有算法和参数点的汇总提取数据，保存到单个 JSON 和单个 CSV 文件中"""
    sens_dir = SENS_DIR
    os.makedirs(sens_dir, exist_ok=True)
    
    # 1. 保存单个汇总 JSON
    json_path = os.path.join(sens_dir, f"summary_{exp_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
    print(f"[Summary] Saved consolidated JSON: {json_path}")
    
    # 2. 保存单个汇总 CSV
    csv_path = os.path.join(sens_dir, f"summary_{exp_name}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter_Value', 'Algorithm', 'Reward', 'Rho', 'Delay', 'Survival', 'Seeds_Count'])
        for val, algos in sorted(results_dict.items(), key=lambda x: float(x[0])):
            for algo, metrics in algos.items():
                display_name = STYLES.get(algo, {}).get('label', algo)
                writer.writerow([
                    val,
                    display_name,
                    round(metrics['reward'], 2),
                    round(metrics['rho'], 4),
                    round(metrics['delay'], 3),
                    round(metrics['survival'], 1),
                    metrics.get('seeds_count', len(seeds))
                ])
    print(f"[Summary] Saved consolidated CSV: {csv_path}")


# ============================================================
# 画图 (学术出版级)
# ============================================================
def plot_sensitivity_2x2(exp_name, x_values, x_label, results_dict, algos):
    """画 2x2 子图的敏感性综合对比曲线图"""
    norm_dict = {}
    for k, v in results_dict.items():
        try:
            norm_dict[float(k)] = v
        except ValueError:
            norm_dict[k] = v

    valid_x = [x for x in x_values if norm_dict.get(x)]
    if not valid_x:
        return

    found_algos = set()
    for x in valid_x:
        found_algos.update(norm_dict[x].keys())
    if not found_algos:
        return

    plot_algos = [a for a in algos if a in found_algos]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = [
        ('reward', 'Episode Reward', 'Reward'),
        ('rho', 'Delay Reduction Ratio ($\\rho$)', 'Delay Reduction Ratio'),
        ('delay', 'Average Delay (s)', 'Average Delay'),
        ('survival', 'Survival Steps', 'Survival Steps'),
    ]

    for ax, (metric_key, ylabel, title) in zip(axes, metrics):
        for algo in plot_algos:
            style = STYLES.get(algo, {'color': 'black', 'marker': 'o', 'ms': 8, 'label': algo})
            y_data = []
            plot_x = []
            for x in valid_x:
                if algo in norm_dict[x]:
                    plot_x.append(x)
                    y_data.append(norm_dict[x][algo][metric_key])

            ax.plot(plot_x, y_data,
                    marker=style['marker'], color=style['color'],
                    linewidth=2, markersize=style.get('ms', 8),
                    label=style['label'])

        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f'{title} vs {x_label}', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=9, loc='best')
        ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout(pad=2.0)
    save_dir = SENS_DIR
    os.makedirs(save_dir, exist_ok=True)

    for ext in ['png', 'pdf']:
        save_path = os.path.join(save_dir, f"fig_sens_{exp_name}.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved綜合图: fig_sens_{exp_name}.png / .pdf")


def plot_sensitivity_rho_only(exp_name, x_values, x_label, results_dict, algos):
    """画单张专属高清学术图：Average Delay Reduction Ratio (rho) vs Parameter (专用于论文排版)"""
    norm_dict = {}
    for k, v in results_dict.items():
        try:
            norm_dict[float(k)] = v
        except ValueError:
            norm_dict[k] = v

    valid_x = [x for x in x_values if norm_dict.get(x)]
    if not valid_x:
        return

    found_algos = set()
    for x in valid_x:
        found_algos.update(norm_dict[x].keys())
    if not found_algos:
        return

    plot_algos = [a for a in algos if a in found_algos]

    plt.figure(figsize=(8.5, 6))
    
    for algo in plot_algos:
        style = STYLES.get(algo, {'color': 'black', 'marker': 'o', 'ms': 8, 'label': algo})
        y_data = []
        plot_x = []
        for x in valid_x:
            if algo in norm_dict[x]:
                plot_x.append(x)
                y_data.append(norm_dict[x][algo]['rho'])

        plt.plot(plot_x, y_data,
                 marker=style['marker'], color=style['color'],
                 linewidth=2.5, markersize=style.get('ms', 8) + 1,
                 label=style['label'])

    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel('Average Delay Reduction Ratio ($\\rho$)', fontsize=14, fontweight='bold')
    plt.title(f'Delay Reduction Ratio ($\\rho$) vs {x_label}', fontsize=15, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='best')
    plt.tick_params(axis='both', which='major', labelsize=12)

    save_dir = SENS_DIR
    os.makedirs(save_dir, exist_ok=True)

    for ext in ['png', 'pdf']:
        save_path = os.path.join(save_dir, f"fig_sens_{exp_name}_rho.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved专属 rho 曲线图: fig_sens_{exp_name}_rho.png / .pdf")


# ============================================================
# 主逻辑
# ============================================================
def _emergency_restore():
    """紧急恢复 config.py"""
    if os.path.exists(BACKUP_PATH):
        try:
            shutil.copy2(BACKUP_PATH, CONFIG_PATH)
            os.remove(BACKUP_PATH)
            print(f"\n[Emergency] Config restored from backup.")
        except Exception:
            print(f"\n[Emergency] FAILED to restore config! Copy {BACKUP_PATH} -> {CONFIG_PATH}")


def main():
    global TARGET_ALGOS, SENS_DIR
    
    parser = argparse.ArgumentParser(
        description='UAV-MEC 参数敏感性分析 (6 算法、专属 rho 曲线、支持分方法运行)')
    parser.add_argument('--smoke_test', action='store_true',
                        help='冒烟测试: 每个点只跑 5 集')
    parser.add_argument('--plot_only', action='store_true',
                        help='仅画图: 基于已有的结果汇总画图')
    parser.add_argument('--episodes', type=int, default=8000,
                        help='每个参数点训练集数 (默认 8000)')
    parser.add_argument('--exp', type=str, default='HOT,E,TASK',
                        help='要运行的实验维度, 逗号分隔. 可选: M,K,E,HOT,TASK (默认 HOT,E,TASK)')
    parser.add_argument('--seeds', type=str, default='42',
                        help='随机种子列表，逗号分隔 (如传入单个种子 --seeds 42 即可不做多种子平均)')
    parser.add_argument('--algos', type=str, default='per_sac_cvx,sac_cvx,random,td3_cvx,maddpg_cvx,esac_per_cvx',
                        help='对比的对比算法列表，逗号分隔')
    parser.add_argument('--out', type=str, default='sensitivity',
                        help='results/ 下的输出文件夹名，例如 sensitivity_v1')
    args = parser.parse_args()

    SENS_DIR = os.path.join(DDPG_DIR, "results", args.out)
    episodes = 5 if args.smoke_test else args.episodes
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    algos = [a.strip() for a in args.algos.split(',') if a.strip()]
    TARGET_ALGOS = algos

    # 解析要跑的实验维度
    if args.exp:
        exp_names = [e.strip().upper() for e in args.exp.split(',')]
        for e in exp_names:
            if e not in EXPERIMENTS:
                print(f"[Error] Unknown experiment '{e}'. Choose from: {list(EXPERIMENTS.keys())}")
                return
    else:
        exp_names = list(EXPERIMENTS.keys())

    print(f"\n{'#'*75}")
    print(f"  UAV-MEC 参数敏感性分析 V4 (学术增益版)")
    print(f"  Mode: {'PLOT ONLY' if args.plot_only else ('SMOKE TEST' if args.smoke_test else 'FULL')}")
    print(f"  Experiments: {exp_names} (你可以用 --exp M,K,E 指定任意三个参数)")
    print(f"  Seeds: {seeds} (传入单个种子即为单随机种子测试)")
    print(f"  Algorithms to track: {algos} (支持分方法运行！)")
    print(f"  Episodes per point: {episodes}")
    print(f"  Output folder: {SENS_DIR}")
    print(f"{'#'*75}\n")

    if not args.plot_only:
        atexit.register(_emergency_restore)
        try:
            signal.signal(signal.SIGINT, lambda s, f: sys.exit(1))
            signal.signal(signal.SIGTERM, lambda s, f: sys.exit(1))
        except (OSError, AttributeError):
            pass

    try:
        if not args.plot_only:
            backup_config()

        for exp_name in exp_names:
            exp = EXPERIMENTS[exp_name]
            var_name = exp['var']
            values = exp['values']
            xlabel = exp['xlabel']
            coupled = exp.get('coupled', {})

            print(f"\n>>> Experiment: {exp_name} ({var_name} = {values})")

            results = {}
            json_path = os.path.join(SENS_DIR, f"summary_{exp_name}.json")
            
            # 读取以前的汇总结果（如果有），确保分方法运行时能够逐步添加算法
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    print(f"  [Load] Loaded existing results from: {json_path}")
                except Exception as e:
                    print(f"  [Load] Failed to load existing {json_path}: {e}")

            # 运行训练
            for val in values:
                if not args.plot_only:
                    success = modify_config(var_name, val)
                    if not success:
                        continue
                    for coupled_var, func in coupled.items():
                        coupled_val = func(val)
                        modify_config(coupled_var, coupled_val)

                    try:
                        run_training_for_all_algos(exp_name, val, episodes, algos, seeds)
                    except Exception as e:
                        print(f"[Error] Exception in training loop for {exp_name}={val}: {e}")

                # 提取性能（包含临时结果和 raw_data 已归档结果，合并各算法）
                perf = extract_average_performance(exp_name, val, TARGET_ALGOS, seeds)
                if perf:
                    results[str(val)] = perf
                    if not args.plot_only:
                        # 归档并清理临时目录
                        consolidate_and_clean(exp_name, val, algos, seeds)

            # 保存本次运行的累加汇总数据 (JSON + CSV)
            if not args.plot_only and results:
                save_summary_data(exp_name, results, seeds)

            # 画图（2x2综合图 + 单独学术级的 rho 曲线图）
            try:
                plot_sensitivity_2x2(exp_name, values, xlabel, results, TARGET_ALGOS)
                plot_sensitivity_rho_only(exp_name, values, xlabel, results, TARGET_ALGOS)
            except Exception as e:
                print(f"[Error] Plotting failed for {exp_name}: {e}")

    except KeyboardInterrupt:
        print("\n[Interrupt] User aborted.")
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
    finally:
        if not args.plot_only:
            restore_config()
            print("\n[Done] Config restored. All experiments finished.")


if __name__ == '__main__':
    main()
