"""
Same-policy resource allocation ablation.

This script loads one trained TDA-PER-SAC policy and evaluates it under two
resource-allocation modes:

  1. closed_form: KKT closed-form CPU/bandwidth allocation
  2. equal:       equal CPU/bandwidth allocation

Unlike training `per_sac_no_cvx` from scratch, this keeps the actor policy
fixed, so the comparison isolates the effect of the inner resource allocator.
"""

import argparse
import csv
import os
import random

import numpy as np
import torch

from config import SEED, SLOT_NUM
from DARS import PERSACAgent
from state_normalization import StateNormalization
from UAV_env import UAVEnv


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(agent, use_cvx_alloc, episodes, seed):
    normalizer = StateNormalization()
    rewards, rhos, delays, survs = [], [], [], []

    for ep in range(episodes):
        set_seed(seed + ep)
        env = UAVEnv(use_cvx_alloc=use_cvx_alloc)
        state = env.reset()

        ep_reward = 0.0
        ep_rhos = []
        ep_delays = []
        steps = 0

        for step in range(SLOT_NUM):
            s_norm = normalizer.normalize(state)
            action = agent.select_action(s_norm, noise_std=0.0)
            state, reward, done, info = env.step(action)

            ep_reward += reward
            if info['action_executed']:
                ep_rhos.append(info["rho_mean"])
                ep_delays.append(info["delay_active_mean"])
                steps += 1
            if done:
                break

        rewards.append(ep_reward)
        rhos.append(float(np.mean(ep_rhos)) if ep_rhos else 0.0)
        delays.append(float(np.mean(ep_delays)) if ep_delays else 0.0)
        survs.append(steps)

    return {
        "reward": np.asarray(rewards, dtype=np.float64),
        "rho": np.asarray(rhos, dtype=np.float64),
        "delay": np.asarray(delays, dtype=np.float64),
        "survival": np.asarray(survs, dtype=np.float64),
    }


def summarize(name, data):
    return {
        "name": name,
        "reward_mean": float(np.mean(data["reward"])),
        "reward_std": float(np.std(data["reward"])),
        "rho_mean": float(np.mean(data["rho"])),
        "rho_std": float(np.std(data["rho"])),
        "delay_mean": float(np.mean(data["delay"])),
        "delay_std": float(np.std(data["delay"])),
        "survival_mean": float(np.mean(data["survival"])),
        "survival_std": float(np.std(data["survival"])),
    }


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Evaluate closed-form vs equal resource allocation under the same learned policy."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(root, "results", "new_v11", "per_sac_cvx.pth"),
        help="Path to the trained TDA-PER-SAC model checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of deterministic evaluation episodes.")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Base random seed.")
    parser.add_argument("--tag", type=str, default="new_v11",
                        help="Result tag directory.")
    parser.add_argument("--name", type=str, default="resource_ablation_same_policy",
                        help="Output file prefix.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    set_seed(args.seed)
    agent = PERSACAgent()
    agent.load(args.model)

    print(f"Loaded policy: {args.model}")
    print(f"Evaluating {args.episodes} episodes with identical seeds per mode...")

    closed_form = evaluate(agent, use_cvx_alloc=True,
                           episodes=args.episodes, seed=args.seed)
    equal = evaluate(agent, use_cvx_alloc=False,
                     episodes=args.episodes, seed=args.seed)

    s_closed = summarize("Closed-form allocation", closed_form)
    s_equal = summarize("Equal allocation", equal)

    result_dir = os.path.join(root, "results", args.tag)
    os.makedirs(result_dir, exist_ok=True)
    npz_path = os.path.join(result_dir, f"{args.name}.npz")
    csv_path = os.path.join(result_dir, f"{args.name}.csv")

    np.savez(
        npz_path,
        closed_reward=closed_form["reward"],
        closed_rho=closed_form["rho"],
        closed_delay=closed_form["delay"],
        closed_survival=closed_form["survival"],
        equal_reward=equal["reward"],
        equal_rho=equal["rho"],
        equal_delay=equal["delay"],
        equal_survival=equal["survival"],
        seed=args.seed,
        episodes=args.episodes,
        model=args.model,
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(s_closed.keys()))
        writer.writeheader()
        writer.writerow(s_closed)
        writer.writerow(s_equal)

    rho_gain = (s_closed["rho_mean"] / max(s_equal["rho_mean"], 1e-12) - 1.0) * 100
    reward_gain = (s_closed["reward_mean"] / max(s_equal["reward_mean"], 1e-12) - 1.0) * 100

    print("\nSame-policy resource allocation ablation")
    print("-" * 58)
    for s in (s_closed, s_equal):
        print(
            f"{s['name']:24s} | "
            f"R={s['reward_mean']:.2f}±{s['reward_std']:.2f} | "
            f"rho={s['rho_mean']:.4f}±{s['rho_std']:.4f} | "
            f"delay={s['delay_mean']:.3f}±{s['delay_std']:.3f} | "
            f"surv={s['survival_mean']:.1f}±{s['survival_std']:.1f}"
        )
    print("-" * 58)
    print(f"Closed-form vs equal: reward {reward_gain:+.2f}%, rho {rho_gain:+.2f}%")
    print(f"Saved: {npz_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
