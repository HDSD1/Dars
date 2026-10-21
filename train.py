"""
多 UAV-MEC DDPG 训练主循环 (V2)
================================
改进:
  - Warmup 阶段: 前 N 步纯随机收集经验, 不更新网络
  - 每 episode 重置 OU 噪声
  - 更详细的日志输出
"""

import os
import time
import numpy as np
import torch

from config import (
    MAX_EPISODES, SLOT_NUM, SEED,
    EXPLORE_NOISE_INIT, EXPLORE_NOISE_MIN, EXPLORE_DECAY,
    BATCH_SIZE, WARMUP_STEPS,
)
from UAV_env import UAVEnv
from ddpg_agent import DDPGAgent
from state_normalization import StateNormalization


def train():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = UAVEnv()
    agent = DDPGAgent()
    normalizer = StateNormalization()

    noise_std = EXPLORE_NOISE_INIT
    t_start = time.time()
    total_steps = 0

    ep_reward_list = []
    avg_delay_list = []
    avg_rho_list = []

    print("=" * 60)
    print(f" Multi-UAV MEC DDPG Training (V2)")
    print(f" State: {env.state_dim}, Action: {env.action_dim}")
    print(f" Episodes: {MAX_EPISODES}, Steps/Ep: {SLOT_NUM}")
    print(f" Warmup: {WARMUP_STEPS} steps (random before learning)")
    print(f" Device: {next(agent.actor.parameters()).device}")
    print("=" * 60)

    for episode in range(MAX_EPISODES):
        state = env.reset()
        agent.ou_noise.reset()  # 每 episode 重置 OU 噪声

        ep_reward = 0.0
        ep_delays = []
        ep_rhos = []

        for step in range(SLOT_NUM):
            s_norm = normalizer.normalize(state)

            # Warmup: 纯随机动作; 之后: 策略 + 噪声
            if total_steps < WARMUP_STEPS:
                action = np.random.uniform(-1, 1, env.action_dim)
            else:
                action = agent.select_action(s_norm, noise_std=noise_std)

            next_state, reward, done, info = env.step(action)
            s_next_norm = normalizer.normalize(next_state)

            agent.store(s_norm, action, reward, s_next_norm, done)

            # 只在 warmup 结束后学习
            if total_steps >= WARMUP_STEPS and agent.buffer.ready(BATCH_SIZE):
                agent.learn()

            state = next_state
            ep_reward += reward
            ep_delays.append(info['delays'].mean())
            ep_rhos.append(info['rho'].mean())
            total_steps += 1

            if done:
                break

        # 噪声衰减 (只在 warmup 之后)
        if total_steps >= WARMUP_STEPS:
            noise_std = max(EXPLORE_NOISE_MIN, noise_std * EXPLORE_DECAY)

        ep_reward_list.append(ep_reward)
        avg_delay_list.append(np.mean(ep_delays))
        avg_rho_list.append(np.mean(ep_rhos))

        # 打印
        if (episode + 1) % 50 == 0 or episode == 0:
            recent_r = np.mean(ep_reward_list[-50:])
            recent_rho = np.mean(avg_rho_list[-50:])
            elapsed = time.time() - t_start
            warmup_str = " [WARMUP]" if total_steps < WARMUP_STEPS else ""
            print(
                f"Ep {episode+1:5d} | "
                f"R: {ep_reward:8.2f} | "
                f"Avg50: {recent_r:8.2f} | "
                f"rho: {recent_rho:.3f} | "
                f"Noise: {noise_std:.4f} | "
                f"Steps: {total_steps} | "
                f"Time: {elapsed:.0f}s{warmup_str}"
            )

        if (episode + 1) % 500 == 0:
            save_path = os.path.join(os.path.dirname(__file__),
                                     f"checkpoints/ddpg_ep{episode+1}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)
            print(f"  -> Model saved: {save_path}")

    total_time = time.time() - t_start
    print(f"\nDone! Time: {total_time:.1f}s ({total_time/60:.1f}min)")

    data_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(data_dir, exist_ok=True)
    np.savez(
        os.path.join(data_dir, "training_data.npz"),
        ep_reward=np.array(ep_reward_list),
        avg_delay=np.array(avg_delay_list),
        avg_rho=np.array(avg_rho_list),
    )
    print(f"Data saved to {data_dir}/training_data.npz")

    final_path = os.path.join(data_dir, "ddpg_final.pth")
    agent.save(final_path)
    print(f"Model saved: {final_path}")

    try:
        from visualize import plot_all
        plot_all(
            ep_reward_list, avg_delay_list, avg_rho_list,
            env, agent, normalizer, save_dir=data_dir
        )
    except Exception as e:
        print(f"Viz error: {e}")


if __name__ == '__main__':
    train()
