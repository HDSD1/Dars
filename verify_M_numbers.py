"""
从三种子敏感性原始数据中提取精确数值，用于论文校核。
"""
import os, numpy as np

RAW = r"d:\Codes\UAV-DDPG-main\DDPG\results\sensitivity\raw_data_M"
ALGOS = ["per_sac_cvx", "sac_cvx", "td3_cvx", "maddpg_cvx", "esac_per_cvx"]
LABELS = {"per_sac_cvx": "DARS", "sac_cvx": "SAC+CVX", "td3_cvx": "TD3+CVX",
          "maddpg_cvx": "MADDPG+CVX", "esac_per_cvx": "ESAC-PER"}
MS = [15, 20, 25, 30, 35]
SEEDS = [42, 43, 44]
LAST_N = 50

print(f"{'M':>3} | {'Algo':<12} | {'Reward':>10} | {'Delay':>10} | {'Rho':>10} | {'Surv':>8}")
print("-" * 70)

# 存储以便计算百分比改进
results = {}

for m in MS:
    for algo in ALGOS:
        vals_r, vals_d, vals_rho, vals_surv = [], [], [], []
        for seed in SEEDS:
            path = os.path.join(RAW, f"M_{m}_seed{seed}_{algo}.npz")
            if not os.path.exists(path):
                continue
            data = np.load(path, allow_pickle=True)
            # 用 eval 数据（确定性评估），如果没有则用训练数据最后 N 轮
            if "eval_reward" in data and len(data["eval_reward"]) > 0:
                vals_r.append(np.mean(data["eval_reward"]))
                vals_rho.append(np.mean(data["eval_rho"]))
            else:
                vals_r.append(np.mean(data["ep_reward"][-LAST_N:]))
                vals_rho.append(np.mean(data["avg_rho"][-LAST_N:]))
            vals_d.append(np.mean(data["avg_delay"][-LAST_N:]))
            if "eval_surv" in data:
                vals_surv.append(np.mean(data["eval_surv"]))
            elif "survival_steps" in data:
                vals_surv.append(np.mean(data["survival_steps"][-LAST_N:]))
            else:
                vals_surv.append(0)
        
        mean_r = np.mean(vals_r)
        mean_d = np.mean(vals_d)
        mean_rho = np.mean(vals_rho)
        mean_surv = np.mean(vals_surv)
        results[(m, algo)] = {"reward": mean_r, "delay": mean_d, "rho": mean_rho, "surv": mean_surv}
        print(f"{m:>3} | {LABELS[algo]:<12} | {mean_r:>10.2f} | {mean_d:>10.3f} | {mean_rho:>10.4f} | {mean_surv:>8.1f}")
    print("-" * 70)

# 计算 DARS 对每个 baseline 的 rho 改进百分比
print("\n=== DARS vs Baselines: Rho improvement (%) at each M ===")
print(f"{'M':>3} | {'vs SAC+CVX':>12} | {'vs TD3+CVX':>12} | {'vs MADDPG':>12} | {'vs ESAC-PER':>12}")
print("-" * 60)
for m in MS:
    dars_rho = results[(m, "per_sac_cvx")]["rho"]
    improvements = []
    for baseline in ["sac_cvx", "td3_cvx", "maddpg_cvx", "esac_per_cvx"]:
        base_rho = results[(m, baseline)]["rho"]
        pct = (dars_rho - base_rho) / base_rho * 100
        improvements.append(pct)
    print(f"{m:>3} | {improvements[0]:>11.1f}% | {improvements[1]:>11.1f}% | {improvements[2]:>11.1f}% | {improvements[3]:>11.1f}%")

# 计算 DARS 对每个 baseline 的 delay 改进百分比
print("\n=== DARS vs Baselines: Delay reduction (%) at each M ===")
print(f"{'M':>3} | {'vs SAC+CVX':>12} | {'vs TD3+CVX':>12} | {'vs MADDPG':>12} | {'vs ESAC-PER':>12}")
print("-" * 60)
for m in MS:
    dars_d = results[(m, "per_sac_cvx")]["delay"]
    for baseline in ["sac_cvx", "td3_cvx", "maddpg_cvx", "esac_per_cvx"]:
        base_d = results[(m, baseline)]["delay"]
        pct = (base_d - dars_d) / base_d * 100
        print(f"  M={m}, vs {LABELS[baseline]}: delay reduced by {pct:.1f}% ({base_d:.3f} -> {dars_d:.3f})")

# M=35 处 DARS 的详细数据
print("\n=== At M=35 (heaviest load), DARS performance ===")
dars = results[(35, "per_sac_cvx")]
print(f"  Reward: {dars['reward']:.2f}")
print(f"  Rho:    {dars['rho']:.4f}")
print(f"  Delay:  {dars['delay']:.3f}")

# 五种 M 下 DARS 的均值优势
print("\n=== Average improvement across all M ===")
for baseline in ["sac_cvx", "td3_cvx", "maddpg_cvx", "esac_per_cvx"]:
    rho_imps = []
    for m in MS:
        dars_rho = results[(m, "per_sac_cvx")]["rho"]
        base_rho = results[(m, baseline)]["rho"]
        rho_imps.append((dars_rho - base_rho) / base_rho * 100)
    print(f"  vs {LABELS[baseline]}: avg rho improvement = {np.mean(rho_imps):.1f}% (range: {min(rho_imps):.1f}% ~ {max(rho_imps):.1f}%)")
