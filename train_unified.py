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
import random
import hashlib
from pathlib import Path
from experiment_config import EnvironmentConfig
from evaluation import (EpisodeMetrics, evaluate_policy, write_json, runtime_info,
                        jsonable, sync_device, configure_policy_environment)
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
def _new_safety_stats():
    """Counts distinguish executed slots from rejected action proposals."""
    return dict(attempted_steps=0, executed_steps=0, movement_corrections=0,
                offload_corrections=0, rejected_steps=0, energy_violation_steps=0,
                max_energy_violation=0.0, env_step_seconds=0.0)


def _record_safety_stats(stats, info, elapsed):
    stats['attempted_steps'] += 1
    stats['env_step_seconds'] += elapsed
    executed = bool(info['action_executed'])
    stats['executed_steps'] += int(executed)
    stats['rejected_steps'] += int(not executed)
    stats['movement_corrections'] += int(info.get('movement_projected', False))
    stats['offload_corrections'] += int(info.get('offload_projected', False))
    if executed:
        violation = max(float(np.max(info['flight_energy_violation'])),
                        float(np.max(info['service_energy_violation'])))
        stats['max_energy_violation'] = max(stats['max_energy_violation'], violation)
        stats['energy_violation_steps'] += int(violation > info['energy_tolerance'])


def run_experiment(algo_name, n_episodes, seed, tag='default',
                   use_hotspot=False, disable_wct=False, env_config=None,
                   warmup_steps=None, batch_size=None, eval_episodes=50):
    """运行单个算法, 返回并保存结果."""
    if not 0 < n_episodes < 100000 or not 0 <= eval_episodes < 100000 or not 0 <= seed < 2**32:
        raise ValueError('Require 1..99999 training episodes, 0..99999 validation episodes and a uint32 seed')
    if warmup_steps is not None and warmup_steps < 0:
        raise ValueError('warmup_steps must be nonnegative')
    if batch_size is not None and batch_size <= 0:
        raise ValueError('batch_size must be positive')
    env_config = env_config or EnvironmentConfig(hotspot_mode='diverse')
    if env_config.hotspot_mode.startswith('unseen'):
        raise ValueError('Held-out hotspot profiles cannot be used for training')
    result_dir = Path(__file__).parent / 'results' / tag
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = result_dir / f'{algo_name}.manifest.json'
    existing = [result_dir / f'{algo_name}{suffix}' for suffix in
                ('.manifest.json', '.npz', '.pth', '.train.jsonl', '.validation.json')]
    if any(path.exists() for path in existing):
        raise FileExistsError(f'Run already exists for {algo_name} in {result_dir}; choose a new tag')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent, needs_training, env_kwargs = ALGO_REGISTRY[algo_name]()
    env_kwargs = dict(env_kwargs)
    if use_hotspot:
        env_kwargs['use_hotspot'] = True
    if disable_wct:
        env_kwargs['disable_wct'] = True
    env = UAVEnv(**env_kwargs, experiment_config=env_config, seed=seed)
    normalizer = StateNormalization(env.battery_capacity)
    configure_policy_environment(agent, env)
    if batch_size is not None:
        from DARS import PERSACAgent
        if not isinstance(agent, PERSACAgent):
            raise ValueError('--batch-size override currently supports DARS and its inherited ablations only')
        agent.batch_size = batch_size
    actual_batch_size = getattr(agent, 'batch_size', BATCH_SIZE)

    noise_std = EXPLORE_NOISE_INIT
    total_steps = 0
    if 'ppo' in algo_name:
        actual_warmup_steps = 0
    elif algo_name == 'tp_drl':
        actual_warmup_steps = min(2000, WARMUP_STEPS)
    else:
        actual_warmup_steps = WARMUP_STEPS

    if warmup_steps is not None:
        actual_warmup_steps = warmup_steps
    import config as system_config
    system_parameters = {k: v for k, v in vars(system_config).items() if k.isupper()}
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob('*.py')}
    agent_module = sys.modules[type(agent).__module__]
    agent_parameters = {k: v for k, v in vars(agent_module).items()
                        if k.isupper() and isinstance(v, (int, float, str, tuple, list))}
    manifest = dict(schema_version=1, source_sha256=source_hashes,
                    agent_parameters=agent_parameters, algorithm=algo_name, training_seed=seed,
                    training_episode_seeds=[1000000 + seed * 100000 + ep for ep in range(n_episodes)],
                    validation_seeds=list(range(100000, 100000 + eval_episodes)),
                    environment=env_config.to_dict(), env_kwargs=env_kwargs,
                    training=dict(episodes=n_episodes, warmup_steps=actual_warmup_steps,
                                  batch_size=actual_batch_size),
                    system_parameters=system_parameters, runtime=runtime_info(),
                    status='running', timing_scope='preprocess+policy+safety+association+allocation',
                    time_feature='min(elapsed_slots/time_reference_slots, 1)',
                    validation_note='Same training hotspot distribution, independent seeds; not held-out test')
    manifest['effective_agent_settings'] = {
        'n_step': getattr(getattr(agent, 'n_step_buffer', None), 'n_step', None),
        'lambda_surprise': getattr(agent.buffer, 'lambda_surprise', None),
    }
    write_json(manifest_path, manifest)
    metrics_path = result_dir / f'{algo_name}.train.jsonl'
    metrics_path.write_text('', encoding='utf-8')

    ep_rewards, avg_delays, avg_rhos, ep_survival_steps = [], [], [], []
    ep_avg_q, ep_avg_td_error, ep_critic_loss = [], [], []  # 训练诊断指标
    safety_history, eval_safety_history = [], []
    t_start = time.time()

    print(f"\n{'='*58}")
    print(f"  Algorithm: {algo_name.upper()}")
    print(f"  Episodes: {n_episodes}, Seed: {seed}, Tag: {tag}")
    print(f"{'='*58}")

    for ep in range(n_episodes):
        episode_seed = manifest['training_episode_seeds'][ep]
        state = env.reset(seed=episode_seed)
        metrics = EpisodeMetrics(env, episode_seed)
        if hasattr(agent, 'ou_noise'):
            agent.ou_noise.reset()

        safety_stats = _new_safety_stats()
        ep_reward = 0.0
        ep_d, ep_r = [], []
        step_q_vals, step_td_errors, step_critic_losses = [], [], []

        for step in range(env.slot_num):
            sync_device()
            policy_started = time.perf_counter()
            s_norm = normalizer.normalize(state)

            if needs_training and total_steps < actual_warmup_steps:
                action = np.random.uniform(-1, 1, env.action_dim)
            else:
                action = agent.select_action(state if getattr(agent, "expects_raw_state", False) else s_norm, noise_std=noise_std, use_ou=True)

            sync_device()
            policy_seconds = time.perf_counter() - policy_started
            # The critic and entropy are defined over this original proposal.
            proposal = np.asarray(action).copy()
            step_started = time.perf_counter()
            next_state, reward, done, info = env.step(proposal)
            env_seconds = time.perf_counter() - step_started
            _record_safety_stats(safety_stats, info, env_seconds)
            metrics.record(info, reward, policy_seconds, env_seconds)
            s_next_norm = normalizer.normalize(next_state)

            if needs_training:
                agent.store(s_norm, proposal, reward, s_next_norm, done)
                if total_steps >= actual_warmup_steps and agent.buffer.ready(actual_batch_size):
                    learn_info = agent.learn()
                    if learn_info is not None and isinstance(learn_info, dict):
                        step_q_vals.append(learn_info.get('avg_q', 0.0))
                        step_td_errors.append(learn_info.get('avg_td_error', 0.0))
                        step_critic_losses.append(learn_info.get('critic_loss', 0.0))

            state = next_state
            ep_reward += reward
            if info['action_executed']:
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

        metrics_row = metrics.finish()
        metrics_row['hotspot_schedule'] = env.hotspot_schedule[:env.slot_num].tolist()
        metrics_row['hotspot_segments'] = env.hotspot_segments
        with metrics_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics_row, default=jsonable, allow_nan=False) + '\n')
        ep_rewards.append(ep_reward)
        avg_delays.append(np.mean(ep_d) if ep_d else 0.0)
        avg_rhos.append(np.mean(ep_r) if ep_r else 0.0)
        ep_survival_steps.append(safety_stats['executed_steps'])
        safety_history.append(safety_stats.copy())
        # 记录诊断指标 (episode 内平均)
        ep_avg_q.append(np.mean(step_q_vals) if step_q_vals else 0.0)
        ep_avg_td_error.append(np.mean(step_td_errors) if step_td_errors else 0.0)
        ep_critic_loss.append(np.mean(step_critic_losses) if step_critic_losses else 0.0)

        if (ep + 1) % max(1, n_episodes // 10) == 0 or ep == 0:
            r50 = np.mean(ep_rewards[-50:]) if len(ep_rewards) >= 50 else np.mean(ep_rewards)
            rho50 = np.mean(avg_rhos[-50:]) if len(avg_rhos) >= 50 else np.mean(avg_rhos)
            surv50 = np.mean(ep_survival_steps[-50:]) if len(ep_survival_steps) >= 50 else np.mean(ep_survival_steps)
            elapsed = time.time() - t_start
            print(f"  [{algo_name}] Ep {ep+1:5d}/{n_episodes} | "
                  f"R_50: {r50:.2f} | ρ_50: {rho50:.3f} | Surv_50: {surv50:.1f} | "
                  f"Time: {elapsed:.0f}s")

    total_time = time.time() - t_start
    print(f"  [{algo_name}] Done! {total_time:.0f}s ({total_time/60:.1f}min)")

    # Save before validation so a completed training run is recoverable.
    if needs_training:
        agent.save(str(result_dir / f'{algo_name}.pth'))
    validation_rows = []
    if eval_episodes:
        validation_rows = evaluate_policy(agent, env_config, env_kwargs,
                                          manifest['validation_seeds'], keep_trace=False)
        write_json(result_dir / f'{algo_name}.validation.json', validation_rows)
    eval_rewards = [row['reward'] for row in validation_rows]
    eval_rhos = [row['slrr_executed'] if row['slrr_executed'] is not None else 0.0 for row in validation_rows]
    eval_survs = [row['executed_steps'] for row in validation_rows]
    for row in validation_rows:
        eval_safety_history.append({key: row.get(key, row.get('max_energy_violation_j', 0.0)
                                                if key == 'max_energy_violation' else 0)
                                    for key in _new_safety_stats()})
    manifest.update(status='completed', training_seconds=total_time,
                    attempted_training_steps=total_steps,
                    gradient_updates=getattr(agent, 'learn_step', None))
    write_json(manifest_path, manifest)

    # 保存结果 → results/<tag>/<algo>.npz
    result_dir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(result_dir, exist_ok=True)

    save_name = f"{algo_name}.npz"
    save_path = os.path.join(result_dir, save_name)
    # 提取最后一次 Episode 的各类日志（为了画图）
    last_trajectory = np.array(env.trajectory_log)  # shape: (SLOT_NUM, K, 2)
    last_battery = np.array(env.battery_log)
    safety_arrays = {}
    for prefix, history in [('safety_', safety_history), ('eval_safety_', eval_safety_history)]:
        for key in _new_safety_stats():
            safety_arrays[prefix + key] = np.array([row[key] for row in history])
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
             seed=seed,
             **safety_arrays)
    print(f"  Saved: {save_path}")

    if needs_training:
        model_path = os.path.join(result_dir, f'{algo_name}.pth')
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
                        help='启用热点任务；具体规则由 --env-config 指定')
    parser.add_argument('--nowct', action='store_true',
                        help='关闭 WCT 充电 (消融实验)')
    parser.add_argument('--env-config', help='JSON EnvironmentConfig; default: diverse hotspot training')
    parser.add_argument('--warmup-steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None, help='DARS/ablation override; defaults unchanged')
    parser.add_argument('--eval-episodes', type=int, default=50, help='Independent validation episodes')
    parser.add_argument('--threads', type=int, default=None)
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    env_config = EnvironmentConfig.from_json(args.env_config) if args.env_config else EnvironmentConfig(hotspot_mode='diverse')

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
                             use_hotspot=args.hotspot, disable_wct=args.nowct,
                             env_config=env_config, warmup_steps=args.warmup_steps,
                             batch_size=args.batch_size, eval_episodes=args.eval_episodes)
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

