import os
import subprocess
import sys

DDPG_DIR = os.path.dirname(os.path.abspath(__file__))
ALGOS = ["no_atten", "per_sac_cvx_no_surprise"]
SEEDS = [42, 43, 44]
EPISODES = 15000
TAG_PREFIX = "new_v13_ms_ablation"

def main():
    print(f"Starting ablation runs: {ALGOS} with seeds {SEEDS} for {EPISODES} episodes.")
    for seed in SEEDS:
        tag = f"{TAG_PREFIX}_seed{seed}"
        for algo in ALGOS:
            cmd = [
                sys.executable, "train_unified.py",
                "--algo", algo,
                "--episodes", str(EPISODES),
                "--seed", str(seed),
                "--tag", tag
            ]
            print(f"\n{'='*60}")
            print(f"Running: Algo={algo} | Seed={seed} | Episodes={EPISODES}")
            print(f"{'='*60}")
            try:
                # Run the command and stream output
                process = subprocess.Popen(cmd, cwd=DDPG_DIR)
                process.wait()
                if process.returncode != 0:
                    print(f"[Error] Algo={algo} Seed={seed} failed with code {process.returncode}")
            except Exception as e:
                print(f"[Fatal Error] Exception during {algo} Seed={seed}: {e}")
                
    print("\n[Done] All ablation runs completed.")

if __name__ == "__main__":
    main()
