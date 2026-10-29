"""Shared per-episode metrics and frozen-policy evaluation."""
import contextlib
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from UAV_env import UAVEnv
from state_normalization import StateNormalization


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=jsonable, allow_nan=False) + '\n', encoding='utf-8')


def runtime_info():
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads(),
                device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')


def configure_policy_environment(agent, env):
    if hasattr(agent, 'configure_environment'):
        agent.configure_environment(env)


def policy_input(agent, normalizer, state):
    # Physical-unit heuristics must not divide already normalized energy again.
    return state if getattr(agent, 'expects_raw_state', False) else normalizer.normalize(state)


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def frozen_policy(agent):
    modules = [v for v in vars(agent).values() if isinstance(v, torch.nn.Module)]
    modes = [v.training for v in modules]
    for v in modules:
        v.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for v, mode in zip(modules, modes):
            v.train(mode)


def model_digest(agent):
    h = hashlib.sha256()
    for name, module in sorted(vars(agent).items()):
        if isinstance(module, torch.nn.Module):
            for key, tensor in sorted(module.state_dict().items()):
                h.update((name + key).encode())
                h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class EpisodeMetrics:
    def __init__(self, env, seed=None, keep_trace=False):
        self.horizon = env.slot_num
        self.seed = seed
        self.executed = 0
        self.attempts = 0
        self.min_battery = env.e_battery_uav.copy()
        self.initial_battery = env.e_battery_uav.copy()
        self.final_battery = env.e_battery_uav.copy()
        self.energy = {key: np.zeros_like(self.min_battery) for key in
                       ('fly', 'hover', 'comp', 'out', 'in', 'charge_accepted', 'charge_overflow')}
        self.reward = 0.0
        self.rhos, self.delays, self.decision_times = [], [], []
        self.reason = None
        self.corrections = dict(movement_corrections=0, offload_corrections=0,
                                rejected_steps=0, energy_violation_steps=0)
        self.max_violation = 0.0
        self.env_seconds = 0.0
        self.trace = [] if keep_trace else None

    def record(self, info, reward, policy_seconds, env_seconds=0.0):
        self.attempts += 1
        self.reward += float(reward)
        self.env_seconds += env_seconds
        self.decision_times.append(float(policy_seconds + info['decision_seconds']))
        self.reason = info['termination_reason']
        executed = bool(info['action_executed'])
        self.corrections['rejected_steps'] += int(not executed)
        self.corrections['movement_corrections'] += int(info['movement_projected'])
        self.corrections['offload_corrections'] += int(info['offload_projected'])
        if executed:
            self.executed += 1
            self.rhos.append(float(info['rho_mean']))
            self.delays.append(float(info['delay_active_mean']))
            self.min_battery = np.minimum(self.min_battery, np.minimum(
                info['battery_after_flight'], info['battery_after_service']))
            self.final_battery = info['battery_after_service'].copy()
            for key in self.energy:
                self.energy[key] += info['energy_' + key]
            violation = max(float(np.max(info['flight_energy_violation'])),
                            float(np.max(info['service_energy_violation'])))
            self.max_violation = max(self.max_violation, violation)
            self.corrections['energy_violation_steps'] += int(violation > info['energy_tolerance'])
        if self.trace is not None:
            keys = ('step', 'action_executed', 'hotspot_phase', 'battery_before',
                    'battery_after_flight', 'battery_after_service', 'energy_fly',
                    'energy_hover', 'energy_comp', 'energy_in', 'energy_charge_accepted',
                    'energy_charge_overflow', 'rho_mean', 'termination_reason')
            self.trace.append({key: info[key] for key in keys})

    def finish(self):
        row = dict(seed=self.seed, horizon=self.horizon, executed_steps=self.executed,
                   attempted_steps=self.attempts, survived=self.executed == self.horizon,
                   termination_reason=self.reason, reward=self.reward,
                   slrr_executed=float(np.mean(self.rhos)) if self.rhos else None,
                   slrr_fixed_horizon=float(np.sum(self.rhos) / self.horizon),
                   delay_executed=float(np.mean(self.delays)) if self.delays else None,
                   min_battery_j=float(np.min(self.min_battery)), min_battery_per_uav_j=self.min_battery,
                   initial_battery_j=self.initial_battery, final_battery_j=self.final_battery,
                   mean_energy_per_executed_slot_j=float(self.energy['out'].sum() / self.executed) if self.executed else None,
                   decision_mean_ms=float(np.mean(self.decision_times) * 1000) if self.decision_times else None,
                   decision_p95_ms=float(np.percentile(self.decision_times, 95) * 1000) if self.decision_times else None,
                   decision_total_seconds=float(np.sum(self.decision_times)), env_step_seconds=self.env_seconds,
                   max_energy_violation_j=self.max_violation, **self.corrections)
        for key, value in self.energy.items():
            row['energy_' + key + '_j'] = float(value.sum())
            row['energy_' + key + '_per_uav_j'] = value
        if self.trace is not None:
            row['trace'] = self.trace
        return row


def summarize(rows):
    if not rows:
        raise ValueError('Need at least one evaluation episode')
    n = len(rows)
    p = sum(r['survived'] for r in rows) / n
    z = 1.959963984540054
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    result = dict(episodes=n, survival_probability=p,
                  survival_wilson95=[max(0., center-half), min(1., center+half)])
    for key in ('executed_steps', 'min_battery_j', 'slrr_executed', 'slrr_fixed_horizon',
                'delay_executed', 'energy_out_j', 'mean_energy_per_executed_slot_j',
                'decision_mean_ms', 'decision_p95_ms'):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = dict(mean=float(np.mean(values)) if values else None,
                           std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                           count=len(values))
    result['worst_min_battery_j'] = min(r['min_battery_j'] for r in rows)
    result['rejected_steps'] = sum(r['rejected_steps'] for r in rows)
    result['energy_violation_steps'] = sum(r['energy_violation_steps'] for r in rows)
    result['max_energy_violation_j'] = max(r['max_energy_violation_j'] for r in rows)
    result['termination_reasons'] = {reason: sum(r['termination_reason'] == reason for r in rows)
                                   for reason in sorted(set(r['termination_reason'] for r in rows))}
    return result


def evaluate_policy(agent, env_config, env_kwargs, seeds, keep_trace=True):
    """No replay writes/optimizer calls. Includes rejected attempts in timing."""
    normalizer = StateNormalization(env_config.battery_capacity)
    rows = []
    with frozen_policy(agent):
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = UAVEnv(**env_kwargs, experiment_config=env_config, seed=seed)
            configure_policy_environment(agent, env)
            state = env.reset(seed=seed)
            # Untimed inference warm-up; do not step or change the scenario.
            # Restore RNG state so stochastic baselines get identical draws.
            np_state, py_state, th_state = np.random.get_state(), random.getstate(), torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            for _ in range(3):
                agent.select_action(policy_input(agent, normalizer, state), noise_std=0.0)
            sync_device()
            np.random.set_state(np_state)
            random.setstate(py_state)
            torch.set_rng_state(th_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            metrics = EpisodeMetrics(env, seed, keep_trace)
            for _ in range(env.slot_num):
                sync_device()
                started = time.perf_counter()
                action = agent.select_action(policy_input(agent, normalizer, state), noise_std=0.0)
                sync_device()
                policy_seconds = time.perf_counter() - started
                started = time.perf_counter()
                state, reward, done, info = env.step(action)
                metrics.record(info, reward, policy_seconds, time.perf_counter() - started)
                if done:
                    break
            row = metrics.finish()
            row['hotspot_schedule'] = env.hotspot_schedule[:env.slot_num].tolist()
            row['hotspot_segments'] = env.hotspot_segments
            if keep_trace:
                row['uav_trajectory'] = env.trajectory_log
            rows.append(row)
    return rows
