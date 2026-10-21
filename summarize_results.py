import os
import glob
import numpy as np

result_dir = r"d:\Codes\UAV-DDPG-main\DDPG\results\new_v11"
pattern = os.path.join(result_dir, "*.npz")
files = glob.glob(pattern)

print(f"{'Algorithm':<25} | {'Reward (Last 50)':<18} | {'Rho (Last 50)':<15} | {'Delay (Last 50)':<15} | {'Survival':<10} | {'Time (min)':<10}")
print("-" * 105)

# Style maps for pretty display names
names_map = {
    'per_sac_cvx': 'DARS (Ours)',
    'att_sac_cvx': 'No_per',
    'sac_cvx': 'SAC',
    'td3_cvx': 'TD3',
    'maddpg_cvx': 'MADDPG',
    'ppo_cvx': 'PPO+',
    'tmc_ppo_cvx': 'TMC-2025 PPO',
    'esac_per_cvx': 'TMC-2025 ESAC-PER',
    'ddpg_cvx': 'DDPG',
    'knn_ddpg_cvx': 'KNN-DDPG',
    'per_uos_rl': 'PER-UOS-RL',
    'rat_cvx': 'RAT+CVX',
    'tp_drl': 'TP-DRL',
    'greedy_cvx': 'Greedy+CVX',
    'priority_greedy_cvx': 'PriorityGreedy+CVX',
    'random': 'Random+CVX',
    'all_local': 'All Local',
    'all_offload': 'All Offload',
    'half_offload': 'Half Offload',
    'per_sac_cvx_no_surprise': 'w/o Dual-Priority',
    'per_sac_no_cvx': 'per_sac_no_cvx',
    'per_sac_cvx_1step': 'w/o N-step (n=1)',
    'no_atten': 'No Attention',
}

for f in sorted(files):
    if "resource_ablation" in f:
        continue
    try:
        data = np.load(f, allow_pickle=True)
        if 'ep_reward' not in data:
            continue
        algo = os.path.basename(f).replace(".npz", "")
        display_name = names_map.get(algo, algo)
        
        rewards = data['ep_reward']
        rhos = data['avg_rho']
        delays = data['avg_delay']
        survivals = data.get('survival_steps', np.full_like(rewards, 40))
        
        last_reward = np.mean(rewards[-50:])
        last_rho = np.mean(rhos[-50:])
        last_delay = np.mean(delays[-50:])
        last_survival = np.mean(survivals[-50:])
        
        total_time = float(data.get('total_time', 0)) / 60.0
        
        print(f"{display_name:<25} | {last_reward:<18.2f} | {last_rho:<15.4f} | {last_delay:<15.3f} | {last_survival:<10.1f} | {total_time:<10.1f}")
    except Exception as e:
        print(f"Error loading {os.path.basename(f)}: {e}")
