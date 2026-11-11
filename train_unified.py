"""
统一训练入口 — 支持所有算法和消融实验
=========================================
用法:
  python train_unified.py --algo per_ddpg --episodes 500 --tag exp1
  python train_unified.py --algo ddpg --episodes 500 --tag exp1
  python train_unified.py --algo random --episodes 100 --tag baseline
  python train_unified.py --algo all_local --episodes 100 --tag baseline
  python train_unified.py --algo all_offload --episodes 100 --tag baseline

所有结果统一保存到 results/<tag>_<algo>.npz, 供 plot_results.py 读取.
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MAX_EPISODES, SLOT_NUM, SEED, BATCH_SIZE, WARMUP_STEPS,
    EXPLORE_NOISE_INIT, EXPLORE_NOISE_MIN, EXPLORE_DECAY,
    ACTION_DIM,
)
from UAV_env import UAVEnv
from state_normalization import StateNormalization


# ==================================================================
# 算法注册表
# ==================================================================
ALGO_REGISTRY = {}


def register_algo(name):
    """装饰器: 注册一个算法工厂. 返回 (agent, needs_training, env_kwargs)."""
    def wrapper(fn):
        ALGO_REGISTRY[name] = fn
        return fn
    return wrapper


@register_algo('ddpg')
def create_ddpg():
    from ddpg_agent import DDPGAgent
    return DDPGAgent(), True, {}  # (agent, needs_training, env_kwargs)


@register_algo('per_ddpg')
def create_per_ddpg():
    from per_ddpg_agent import PERDDPGAgent
    return PERDDPGAgent(), True, {}


# ---- 凸优化版本: 与上面完全相同的 Agent, 但环境用最优算力分配 ----
@register_algo('ddpg_cvx')
def create_ddpg_cvx():
    from ddpg_agent import DDPGAgent
    return DDPGAgent(), True, {'use_cvx_alloc': True}


@register_algo('knn_ddpg_cvx')
def create_knn_ddpg_cvx():
    from knn_ddpg_agent import KNNDDPGAgent
    return KNNDDPGAgent(), True, {'use_cvx_alloc': True}


@register_algo('per_ddpg_cvx')
def create_per_ddpg_cvx():
    from per_ddpg_agent import PERDDPGAgent
    return PERDDPGAgent(), True, {'use_cvx_alloc': True}








# ---- 弱化版 Vanilla DDPG (无 LayerNorm/Xavier/梯度裁剪, 小网络) ----
@register_algo('tp_drl')
def create_tp_drl():
    # Task-priority guided DRL baseline adapted from TMC 2024:
    # "Joint Task Offloading, Resource Allocation, and Trajectory Design for
    # Multi-UAV Cooperative Edge Computing With Task Priority".
    from tp_drl_agent import TPDRLAgent
    return TPDRLAgent(), True, {'use_cvx_alloc': True}


@register_algo('ddpg_vanilla')
def create_ddpg_vanilla():
    from ddpg_vanilla import VanillaDDPGAgent
    return VanillaDDPGAgent(), True, {}


@register_algo('ddpg_vanilla_cvx')
def create_ddpg_vanilla_cvx():
    from ddpg_vanilla import VanillaDDPGAgent
    return VanillaDDPGAgent(), True, {'use_cvx_alloc': True}

class RandomAgent:
    """随机策略 — 每步随机动作."""
    def __init__(self):
        from config import STATE_DIM
        self.a_dim = ACTION_DIM
        self.ou_noise = type('', (), {'reset': lambda s: None})()
        self.buffer = type('', (), {'ready': lambda s, b: False})()

    def select_action(self, state, noise_std=0.0, **kw):
        return np.random.uniform(-1, 1, self.a_dim)

    def store(self, *args): pass
    def learn(self): pass
    def save(self, path): pass


class FixedOffloadAgent:
    """固定卸载比例策略."""
    def __init__(self, offload_ratio=1.0):
        self.a_dim = ACTION_DIM
        self.offload_ratio = offload_ratio
        self.ou_noise = type('', (), {'reset': lambda s: None})()
        self.buffer = type('', (), {'ready': lambda s, b: False})()

    def select_action(self, state, noise_std=0.0, **kw):
        from config import K
        action = np.zeros(self.a_dim)
        # 飞行: cos=0, sin=0, speed=-1 (悬停)
        for k in range(K):
            action[3 * k] = 0.0       # cos → 0
            action[3 * k + 1] = 0.0   # sin → 0
            action[3 * k + 2] = -1.0  # 速度 → 0
        # 卸载比例
        x = self.offload_ratio * 2 - 1  # [0,1] → [-1,1]
        action[3 * K:] = x
        return np.clip(action, -1, 1)

    def store(self, *args): pass
    def learn(self): pass
    def save(self, path): pass


@register_algo('random')
def create_random():
    return RandomAgent(), False, {'use_cvx_alloc': True}


@register_algo('all_local')
def create_all_local():
    return FixedOffloadAgent(offload_ratio=0.0), False, {}


@register_algo('all_offload')
def create_all_offload():
    return FixedOffloadAgent(offload_ratio=1.0), False, {}


@register_algo('half_offload')
def create_half_offload():
    return FixedOffloadAgent(offload_ratio=0.5), False, {}


# ---- 本文方法 (Attention-PER-SAC+CVX) ----
@register_algo('per_sac_cvx')
def create_per_sac_cvx():
    from DARS import PERSACAgent
    return PERSACAgent(), True, {'use_cvx_alloc': True}


@register_algo('no_atten')
def create_no_atten():
    # DARS ablation: keep PER/N-step/CVX, remove only user attention.
    from no_atten import NoAttentionPERSACAgent
    return NoAttentionPERSACAgent(), True, {'use_cvx_alloc': True}


@register_algo('att_per_sac_cvx')
def create_att_per_sac_cvx():
    from DARS import PERSACAgent
    return PERSACAgent(), True, {'use_cvx_alloc': True}


# ---- 消融实验: w/o Dual-Priority (纯 TD-error PER, λ=0) ----
@register_algo('per_sac_cvx_no_surprise')
def create_per_sac_cvx_no_surprise():
    import DARS
    agent = DARS.PERSACAgent()
    # ★ 直接设置实例变量, 确保训练时 λ=0
    agent.buffer.lambda_surprise = 0.0
    return agent, True, {'use_cvx_alloc': True}


# ---- 消融实验: w/o CVX (均分资源) ----
@register_algo('per_sac_no_cvx')
def create_per_sac_no_cvx():
    from DARS import PERSACAgent
    return PERSACAgent(), True, {'use_cvx_alloc': False}


@register_algo('att_per_sac_eq')
def create_att_per_sac_eq():
    from DARS import PERSACAgent
    return PERSACAgent(), True, {'use_cvx_alloc': False}


# ---- 消融实验: w/o N-step (n=1) ----
@register_algo('per_sac_cvx_1step')
def create_per_sac_cvx_1step():
    import DARS
    original_n = DARS.N_STEP
    DARS.N_STEP = 1
    agent = DARS.PERSACAgent()
    DARS.N_STEP = original_n
    return agent, True, {'use_cvx_alloc': True}


# ---- Baseline 对比算法 ----
@register_algo('sac_cvx')
def create_sac_cvx():
    from sac_agent import SACAgent
    return SACAgent(), True, {'use_cvx_alloc': True}


@register_algo('att_sac_cvx')
def create_att_sac_cvx():
    from att_sac_agent import AttSACAgent
    return AttSACAgent(), True, {'use_cvx_alloc': True}


@register_algo('ppo_cvx')
def create_ppo_cvx():
    from ppo_agent import PPOAgent
    return PPOAgent(), True, {'use_cvx_alloc': True}


@register_algo('tmc_ppo_cvx')
def create_tmc_ppo_cvx():
    from tmc_ppo_agent import TMCPPOAgent
    return TMCPPOAgent(), True, {'use_cvx_alloc': True}


@register_algo('esac_per_cvx')
def create_esac_per_cvx():
    from esac_per_agent import ESACPERAgent
    return ESACPERAgent(), True, {'use_cvx_alloc': True}


@register_algo('maddpg_cvx')
def create_maddpg_cvx():
    from maddpg_agent import MADDPGAgent
    return MADDPGAgent(), True, {'use_cvx_alloc': True}


@register_algo('td3_cvx')
def create_td3_cvx():
    from td3_agent import TD3Agent
    return TD3Agent(), True, {'use_cvx_alloc': True}


@register_algo('greedy_cvx')
def create_greedy_cvx():
    from greedy_agent import GreedyAgent
    return GreedyAgent(), False, {'use_cvx_alloc': True}


@register_algo('priority_greedy_cvx')
def create_priority_greedy_cvx():
    from priority_greedy_agent import PriorityGreedyAgent
    return PriorityGreedyAgent(), False, {'use_cvx_alloc': True}


# ---- 热点场景变体 ----
@register_algo('per_ddpg_cvx_hotspot')
def create_per_ddpg_cvx_hotspot():
    from per_ddpg_agent import PERDDPGAgent
    return PERDDPGAgent(), True, {'use_cvx_alloc': True, 'use_hotspot': True}


@register_algo('ddpg_cvx_hotspot')
def create_ddpg_cvx_hotspot():
    from ddpg_agent import DDPGAgent
    return DDPGAgent(), True, {'use_cvx_alloc': True, 'use_hotspot': True}


@register_algo('per_ddpg_hotspot')
def create_per_ddpg_hotspot():
    from per_ddpg_agent import PERDDPGAgent
    return PERDDPGAgent(), True, {'use_hotspot': True}


# ---- 无 WCT 消融 ----
@register_algo('per_ddpg_cvx_nowct')
def create_per_ddpg_cvx_nowct():
    from per_ddpg_agent import PERDDPGAgent
    return PERDDPGAgent(), True, {'use_cvx_alloc': True, 'disable_wct': True}


# ==================================================================
# 通用训练函数
# ==================================================================
def run_experiment(algo_name, n_episodes, seed, tag='default',
                   use_hotspot=False, disable_wct=False):
    """运行单个算法, 返回并保存结果."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent, needs_training, env_kwargs = ALGO_REGISTRY[algo_name]()
    env_kwargs = dict(env_kwargs)
    if use_hotspot:
        env_kwargs['use_hotspot'] = True
    if disable_wct:
        env_kwargs['disable_wct'] = True
    env = UAVEnv(**env_kwargs)
    normalizer = StateNormalization()

    noise_std = EXPLORE_NOISE_INIT
    total_steps = 0
    if 'ppo' in algo_name:
        actual_warmup_steps = 0
    elif algo_name == 'tp_drl':
        actual_warmup_steps = min(2000, WARMUP_STEPS)
    else:
        actual_warmup_steps = WARMUP_STEPS

    ep_rewards, avg_delays, avg_rhos, ep_survival_steps = [], [], [], []
    ep_avg_q, ep_avg_td_error, ep_critic_loss = [], [], []  # 训练诊断指标
    shield_stats = {'steps': 0, 'interventions': 0, 'search_failures': 0,
                    'deadline_proposals': 0, 'env_seconds': 0.0}
    result_dir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(result_dir, exist_ok=True)
    t_start = time.time()

    print(f"\n{'='*58}")
    print(f"  Algorithm: {algo_name.upper()}")
    print(f"  Episodes: {n_episodes}, Seed: {seed}, Tag: {tag}")
    print(f"{'='*58}")

    for ep in range(n_episodes):
        state = env.reset()
        if hasattr(agent, 'ou_noise'):
            agent.ou_noise.reset()

        ep_reward = 0.0
        ep_d, ep_r = [], []
        step_q_vals, step_td_errors, step_critic_losses = [], [], []

        for step in range(SLOT_NUM):
            s_norm = normalizer.normalize(state)

            if needs_training and total_steps < actual_warmup_steps:
                action = np.random.uniform(-1, 1, env.action_dim)
            else:
                action = agent.select_action(s_norm, noise_std=noise_std, use_ou=True)

            step_start = time.perf_counter()
            next_state, reward, done, info = env.step(action)
            shield_stats['env_seconds'] += time.perf_counter() - step_start
            shield_stats['steps'] += 1
            shield_stats['interventions'] += int(info.get('shield_intervened', False))
            shield_stats['search_failures'] += int(
                info.get('shield_status') == 'no_feasible_candidate_found')
            shield_stats['deadline_proposals'] += int(
                info.get('shield_original_failure') == 'edge_deadline')
            s_next_norm = normalizer.normalize(next_state)

            if needs_training:
                agent.store(s_norm, action, reward, s_next_norm, done)
                if total_steps >= actual_warmup_steps and agent.buffer.ready(BATCH_SIZE):
                    learn_info = agent.learn()
                    if learn_info is not None and isinstance(learn_info, dict):
                        step_q_vals.append(learn_info.get('avg_q', 0.0))
                        step_td_errors.append(learn_info.get('avg_td_error', 0.0))
                        step_critic_losses.append(learn_info.get('critic_loss', 0.0))

            state = next_state
            ep_reward += reward
            ep_d.append(info['delay_active_mean'])  # 活跃用户平均时延
            ep_r.append(info['rho_mean'])  # 只对 active 用户的 ρ 均值
            total_steps += 1

            if done:
                break

        # ★ PPO: episode 结束时 flush buffer (处理非 done 终止的 episode)
        if 'ppo' in algo_name and needs_training:
            if hasattr(agent, 'episode_buffer'):
                agent.episode_buffer.finish_episode_if_needed()
            # PPO 在 episode 边界学习 (on-policy)
            if agent.buffer.ready(0):
                learn_info = agent.learn()
                if learn_info is not None:
                    step_q_vals.append(learn_info['avg_q'])
                    step_td_errors.append(learn_info['avg_td_error'])
                    step_critic_losses.append(learn_info['critic_loss'])

        if needs_training and total_steps >= actual_warmup_steps:
            noise_std = max(EXPLORE_NOISE_MIN, noise_std * EXPLORE_DECAY)

        ep_rewards.append(ep_reward)
        avg_delays.append(np.mean(ep_d))
        avg_rhos.append(np.mean(ep_r))
        ep_survival_steps.append(step + 1)
        # 记录诊断指标 (episode 内平均)
        ep_avg_q.append(np.mean(step_q_vals) if step_q_vals else 0.0)
        ep_avg_td_error.append(np.mean(step_td_errors) if step_td_errors else 0.0)
        ep_critic_loss.append(np.mean(step_critic_losses) if step_critic_losses else 0.0)

        if (ep + 1) % min(50, max(1, n_episodes // 10)) == 0 or ep == 0:
            r50 = np.mean(ep_rewards[-50:]) if len(ep_rewards) >= 50 else np.mean(ep_rewards)
            rho50 = np.mean(avg_rhos[-50:]) if len(avg_rhos) >= 50 else np.mean(avg_rhos)
            surv50 = np.mean(ep_survival_steps[-50:]) if len(ep_survival_steps) >= 50 else np.mean(ep_survival_steps)
            elapsed = time.time() - t_start
            print(f"  [{algo_name}] Ep {ep+1:5d}/{n_episodes} | "
                  f"R_50: {r50:.2f} | ρ_50: {rho50:.3f} | Surv_50: {surv50:.1f} | "
                  f"Time: {elapsed:.0f}s")
            progress = dict(shield_stats, episode=ep + 1, episodes=n_episodes,
                            seed=seed, elapsed_seconds=elapsed,
                            recent_reward=float(r50), recent_rho=float(rho50))
            with open(os.path.join(result_dir, f'{algo_name}_progress.json'),
                      'w', encoding='utf-8') as handle:
                json.dump(progress, handle, indent=2)
            np.savez(os.path.join(result_dir, f'{algo_name}_progress.npz'),
                     ep_reward=ep_rewards, avg_delay=avg_delays,
                     avg_rho=avg_rhos, survival_steps=ep_survival_steps)
            if needs_training and (ep + 1) % 250 == 0:
                agent.save(os.path.join(result_dir, f'{algo_name}_checkpoint.pth'))

    total_time = time.time() - t_start
    print(f"  [{algo_name}] Done! {total_time:.0f}s ({total_time/60:.1f}min)")

    # ★ 确定性评估 (关闭噪声, 验证真正收敛)
    N_EVAL = 50
    eval_rewards, eval_rhos, eval_survs = [], [], []
    if needs_training:
        print(f"  [{algo_name}] Running {N_EVAL} deterministic eval episodes...")
        eval_env = UAVEnv(**env_kwargs)
        for eval_ep in range(N_EVAL):
            state = eval_env.reset()
            ep_reward = 0.0
            ep_r = []
            for step in range(SLOT_NUM):
                s_norm = normalizer.normalize(state)
                action = agent.select_action(s_norm, noise_std=0.0)  # ★ 无噪声
                next_state, reward, done, info = eval_env.step(action)
                state = next_state
                ep_reward += reward
                ep_r.append(info['rho_mean'])
                if done:
                    break
            eval_rewards.append(ep_reward)
            eval_rhos.append(np.mean(ep_r))
            eval_survs.append(step + 1)
        print(f"  [{algo_name}] Eval: R={np.mean(eval_rewards):.2f}±{np.std(eval_rewards):.2f} | "
              f"ρ={np.mean(eval_rhos):.4f}±{np.std(eval_rhos):.4f} | "
              f"Surv={np.mean(eval_survs):.1f}±{np.std(eval_survs):.1f}")

    # 保存结果 → results/<tag>/<algo>.npz
    result_dir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(result_dir, exist_ok=True)

    save_name = f"{algo_name}.npz"
    save_path = os.path.join(result_dir, save_name)
    # 提取最后一次 Episode 的各类日志（为了画图）
    last_trajectory = np.array(env.trajectory_log)  # shape: (SLOT_NUM, K, 2)
    last_battery = np.array(env.battery_log)
    np.savez(save_path,
             algo=algo_name,
             ep_reward=np.array(ep_rewards),
             avg_delay=np.array(avg_delays),
             avg_rho=np.array(avg_rhos),
             survival_steps=np.array(ep_survival_steps),
             avg_q=np.array(ep_avg_q),
             avg_td_error=np.array(ep_avg_td_error),
             critic_loss=np.array(ep_critic_loss),
             eval_reward=np.array(eval_rewards) if eval_rewards else np.array([]),
             eval_rho=np.array(eval_rhos) if eval_rhos else np.array([]),
             eval_surv=np.array(eval_survs) if eval_survs else np.array([]),
             last_trajectory=last_trajectory,
             last_battery=last_battery,
             total_time=total_time,
             n_episodes=n_episodes,
             seed=seed)
    print(f"  Saved: {save_path}")

    # 保存模型
    if needs_training:
        model_path = os.path.join(result_dir, f"{algo_name}.pth")
        agent.save(model_path)
        print(f"  Model: {model_path}")

    return {
        'algo': algo_name,
        'ep_rewards': np.array(ep_rewards),
        'avg_delays': np.array(avg_delays),
        'avg_rhos': np.array(avg_rhos),
        'total_time': total_time,
    }


# ==================================================================
# CLI 入口
# ==================================================================
def main():
    parser = argparse.ArgumentParser(description='Multi-UAV MEC 统一训练')
    parser.add_argument('--algo', type=str, default='per_ddpg',
                        help='算法名(逗号分隔多个) 或 "all". 可选: ' + str(list(ALGO_REGISTRY.keys())))
    parser.add_argument('--episodes', type=int, default=500,
                        help='训练回合数')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='随机种子')
    parser.add_argument('--tag', type=str, default='exp',
                        help='实验标签 (用于区分不同批次的实验)')
    parser.add_argument('--hotspot', action='store_true',
                        help='启用热点轮转 (3阶段热/冷簇)')
    parser.add_argument('--nowct', action='store_true',
                        help='关闭 WCT 充电 (消融实验)')
    args = parser.parse_args()

    # 解析算法列表
    if args.algo == 'all':
        algo_list = list(ALGO_REGISTRY.keys())
    elif ',' in args.algo:
        algo_list = [a.strip() for a in args.algo.split(',')]
    else:
        algo_list = [args.algo]

    # 验证算法名
    for a in algo_list:
        if a not in ALGO_REGISTRY:
            print(f"  [ERROR] Unknown algo: {a}")
            print(f"  Available: {list(ALGO_REGISTRY.keys())}")
            return

    print(f"\n{'#'*60}")
    print(f"  Batch run: {len(algo_list)} algorithms")
    print(f"  Algos: {algo_list}")
    print(f"  Episodes: {args.episodes}, Tag: {args.tag}")
    print(f"{'#'*60}")

    results = []
    for i, algo in enumerate(algo_list):
        print(f"\n>>> [{i+1}/{len(algo_list)}] {algo.upper()}")
        res = run_experiment(algo, args.episodes, args.seed, args.tag,
                             use_hotspot=args.hotspot, disable_wct=args.nowct)
        results.append(res)

    # 汇总
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  Batch Summary ({args.tag})")
        print(f"{'='*60}")
        for res in results:
            r50 = np.mean(res['ep_rewards'][-50:])
            rho50 = np.mean(res['avg_rhos'][-50:])
            print(f"  {res['algo']:20s} | R={r50:8.2f} | rho={rho50:.4f} | {res['total_time']:.0f}s")


if __name__ == '__main__':
    main()

