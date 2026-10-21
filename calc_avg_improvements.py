import os
import numpy as np

DDPG_DIR = r"d:\Codes\UAV-DDPG-main\DDPG"
SEEDS = [42, 43, 44]
ALGOS = ["per_sac_cvx", "esac_per_cvx", "sac_cvx", "td3_cvx", "maddpg_cvx"]

print("==== M SENSITIVITY ====")
M_VALUES = [15, 20, 25, 30, 35]
RAW_M_DIR = os.path.join(DDPG_DIR, "results", "sensitivity", "raw_data_M")

m_results = {algo: {'reward': [], 'delay': [], 'rho': []} for algo in ALGOS}

for m in M_VALUES:
    for algo in ALGOS:
        reward_seeds = []
        delay_seeds = []
        rho_seeds = []
        for seed in SEEDS:
            path = os.path.join(RAW_M_DIR, f"M_{m}_seed{seed}_{algo}.npz")
            if os.path.exists(path):
                data = np.load(path)
                reward_seeds.append(np.mean(data['ep_reward'][-50:]))
                delay_seeds.append(np.mean(data['avg_delay'][-50:]))
                rho_seeds.append(np.mean(data['avg_rho'][-50:]))
        
        if reward_seeds:
            m_results[algo]['reward'].append(np.mean(reward_seeds))
            m_results[algo]['delay'].append(np.mean(delay_seeds))
            m_results[algo]['rho'].append(np.mean(rho_seeds))

# Calculate average improvements
for metric in ['reward', 'delay', 'rho']:
    dars = np.array(m_results["per_sac_cvx"][metric])
    print(f"\n--- Metric: {metric} ---")
    for algo in ALGOS[1:]:
        base = np.array(m_results[algo][metric])
        if metric == 'delay':
            # Reduction in delay
            imp = (base - dars) / base * 100
        else:
            # Improvement in reward/rho
            imp = (dars - base) / base * 100
        avg_imp = np.mean(imp)
        print(f"vs {algo}: {avg_imp:.2f}%")


print("\n==== HOT SENSITIVITY ====")
HOT_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
RAW_HOT_DIR = os.path.join(DDPG_DIR, "results", "sensitivity", "raw_data_HOT")

hot_results = {algo: {'reward': [], 'delay': [], 'rho': []} for algo in ALGOS}

for h in HOT_VALUES:
    for algo in ALGOS:
        reward_seeds = []
        delay_seeds = []
        rho_seeds = []
        for seed in SEEDS:
            path = os.path.join(RAW_HOT_DIR, f"HOT_{h}_seed{seed}_{algo}.npz")
            if os.path.exists(path):
                data = np.load(path)
                reward_seeds.append(np.mean(data['eval_reward']))
                delay_seeds.append(np.mean(data['avg_delay'][-50:]))
                rho_seeds.append(np.mean(data['eval_rho']))
                
        if reward_seeds:
            hot_results[algo]['reward'].append(np.mean(reward_seeds))
            hot_results[algo]['delay'].append(np.mean(delay_seeds))
            hot_results[algo]['rho'].append(np.mean(rho_seeds))

# Calculate average improvements
for metric in ['reward', 'delay', 'rho']:
    dars = np.array(hot_results["per_sac_cvx"][metric])
    print(f"\n--- Metric: {metric} ---")
    for algo in ALGOS[1:]:
        base = np.array(hot_results[algo][metric])
        if metric == 'delay':
            imp = (base - dars) / base * 100
        else:
            imp = (dars - base) / base * 100
        avg_imp = np.mean(imp)
        print(f"vs {algo}: {avg_imp:.2f}%")

