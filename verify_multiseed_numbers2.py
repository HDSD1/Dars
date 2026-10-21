import os
import numpy as np

DDPG_DIR = r"d:\Codes\UAV-DDPG-main\DDPG"
SEEDS = [42, 43, 44]
TAG = "new_v13_ms"
ALGOS = ["per_sac_cvx", "sac_cvx", "td3_cvx", "maddpg_cvx", "esac_per_cvx"]

print("Verifying multiseed metrics (eval_reward, avg_delay)...")

for metric in ['eval_reward', 'avg_delay']:
    print(f"\n--- Metric: {metric} ---")
    algo_stats = {}
    for algo in ALGOS:
        means = []
        for seed in SEEDS:
            path = os.path.join(DDPG_DIR, "results", f"{TAG}_seed{seed}", f"{algo}.npz")
            if os.path.exists(path):
                data = np.load(path)
                
                # Check for either 'eval_reward' or 'ep_reward'
                if metric == 'eval_reward':
                    if 'eval_reward' in data:
                        val = data['eval_reward']
                    elif 'ep_reward' in data:
                        val = data['ep_reward']
                    else:
                        print(f"Missing {metric} in {path}")
                        continue
                else:
                    if metric in data:
                        val = data[metric]
                    else:
                        print(f"Missing {metric} in {path}")
                        continue
                        
                final_val = np.mean(val[-50:])
                means.append(final_val)
            else:
                print(f"File not found: {path}")
        
        if means:
            algo_stats[algo] = {
                "mean": np.mean(means),
                "std": np.std(means),
                "all": means
            }
            print(f"[{algo}] Final {metric}: {algo_stats[algo]['mean']:.2f} \u00B1 {algo_stats[algo]['std']:.2f}")
