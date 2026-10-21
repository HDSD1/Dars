"""
从三种子 HOT 敏感性数据中提取精确数值。
"""
import csv, numpy as np
from collections import defaultdict

CSV_PATH = r"d:\Codes\UAV-DDPG-main\DDPG\results\sensitivity\summary_HOT_bar.csv"

# 读取 CSV
rows = []
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# 按 (phot, algo) 分组，取 eval_rho, eval_reward, delay 的三种子均值
ALGOS = ["per_sac_cvx", "sac_cvx", "td3_cvx", "maddpg_cvx", "esac_per_cvx"]
LABELS = {"per_sac_cvx": "DARS", "sac_cvx": "SAC+CVX", "td3_cvx": "TD3+CVX",
          "maddpg_cvx": "MADDPG+CVX", "esac_per_cvx": "ESAC-PER"}
PHOTS = [0.6, 0.7, 0.8, 0.9, 1.0]

grouped = defaultdict(lambda: {"rho": [], "reward": [], "delay": []})
for row in rows:
    phot = float(row["hot"])
    algo = row["algo"]
    grouped[(phot, algo)]["rho"].append(float(row["rho_mean"]))
    grouped[(phot, algo)]["reward"].append(float(row["reward_mean"]))
    grouped[(phot, algo)]["delay"].append(float(row["delay_mean"]))

results = {}
print(f"{'phot':>5} | {'Algo':<12} | {'Eval Rho':>10} | {'Eval Reward':>12} | {'Delay':>10}")
print("-" * 65)
for phot in PHOTS:
    for algo in ALGOS:
        key = (phot, algo)
        mean_rho = np.mean(grouped[key]["rho"])
        mean_reward = np.mean(grouped[key]["reward"])
        mean_delay = np.mean(grouped[key]["delay"])
        results[key] = {"rho": mean_rho, "reward": mean_reward, "delay": mean_delay}
        print(f"{phot:>5} | {LABELS[algo]:<12} | {mean_rho:>10.4f} | {mean_reward:>12.2f} | {mean_delay:>10.3f}")
    print("-" * 65)

# DARS vs baselines 改进百分比（ρ）
print("\n=== DARS vs Baselines: Rho improvement (%) at each phot ===")
for phot in PHOTS:
    dars_rho = results[(phot, "per_sac_cvx")]["rho"]
    line = f"phot={phot}: DARS={dars_rho:.4f} | "
    for bl in ["sac_cvx", "td3_cvx", "maddpg_cvx", "esac_per_cvx"]:
        bl_rho = results[(phot, bl)]["rho"]
        pct = (dars_rho - bl_rho) / bl_rho * 100
        line += f"vs {LABELS[bl]}={pct:+.1f}% "
    print(line)

# DARS vs baselines 改进百分比（delay reduction）
print("\n=== DARS vs Baselines: Delay reduction (%) at each phot ===")
for phot in PHOTS:
    dars_d = results[(phot, "per_sac_cvx")]["delay"]
    line = f"phot={phot}: DARS_delay={dars_d:.3f} | "
    for bl in ["sac_cvx", "td3_cvx", "maddpg_cvx", "esac_per_cvx"]:
        bl_d = results[(phot, bl)]["delay"]
        pct = (bl_d - dars_d) / bl_d * 100
        line += f"vs {LABELS[bl]}={pct:+.1f}% "
    print(line)

# 平均改进
print("\n=== Average improvement across all phot ===")
for bl in ["sac_cvx", "td3_cvx", "maddpg_cvx", "esac_per_cvx"]:
    rho_imps = []
    delay_imps = []
    for phot in PHOTS:
        dars_rho = results[(phot, "per_sac_cvx")]["rho"]
        bl_rho = results[(phot, bl)]["rho"]
        rho_imps.append((dars_rho - bl_rho) / bl_rho * 100)
        dars_d = results[(phot, "per_sac_cvx")]["delay"]
        bl_d = results[(phot, bl)]["delay"]
        delay_imps.append((bl_d - dars_d) / bl_d * 100)
    print(f"  vs {LABELS[bl]}: avg rho improvement = {np.mean(rho_imps):.1f}% (range: {min(rho_imps):.1f}% ~ {max(rho_imps):.1f}%)")
    print(f"               avg delay reduction = {np.mean(delay_imps):.1f}% (range: {min(delay_imps):.1f}% ~ {max(delay_imps):.1f}%)")
