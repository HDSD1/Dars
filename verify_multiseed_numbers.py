import os
import numpy as np

DDPG_DIR = r"d:\Codes\UAV-DDPG-main\DDPG"
SEEDS = [42, 43, 44]
TAG = "new_v13_ms"
ALGOS = ["per_sac_cvx", "sac_cvx", "td3_cvx", "maddpg_cvx", "esac_per_cvx"]

print("Verifying multiseed metrics (eval_rho)...")

algo_stats = {}
for algo in ALGOS:
    rho_means = []
    for seed in SEEDS:
        path = os.path.join(DDPG_DIR, "results", f"{TAG}_seed{seed}", f"{algo}.npz")
        if os.path.exists(path):
            data = np.load(path)
            # Use the mean of the last 50 episodes of avg_rho or eval_rho
            if 'eval_rho' in data:
                rho = data['eval_rho']
            elif 'avg_rho' in data:
                rho = data['avg_rho']
            else:
                print(f"Missing rho metric in {path}")
                continue
            
            # The convergence plots usually show the mean over the entire training but the *final* value is the mean of the last few episodes.
            # Let's take the mean of the last 50 evaluations.
            final_rho = np.mean(rho[-50:])
            rho_means.append(final_rho)
        else:
            print(f"File not found: {path}")
    
    if rho_means:
        algo_stats[algo] = {
            "mean": np.mean(rho_means),
            "std": np.std(rho_means),
            "all": rho_means
        }
        print(f"[{algo}] Final Rho: {algo_stats[algo]['mean']:.4f} \u00B1 {algo_stats[algo]['std']:.4f}")
        
if "per_sac_cvx" in algo_stats:
    dars_mean = algo_stats["per_sac_cvx"]["mean"]
    print("\nImprovements over baselines:")
    for algo in ALGOS:
        if algo != "per_sac_cvx" and algo in algo_stats:
            base_mean = algo_stats[algo]["mean"]
            imp = (dars_mean - base_mean) / base_mean * 100
            print(f"vs {algo}: +{imp:.1f}% (DARS={dars_mean:.4f}, Base={base_mean:.4f})")
