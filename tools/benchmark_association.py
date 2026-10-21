"""Compare association heuristics with exact subset-DP on small instances.

Example:
    python tools/benchmark_association.py --instances 20 --users 14 --capacity 5
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from UAV_env import UAVEnv
from config import K, M, TASK_MIN, TASK_MAX


def build_instance(seed, n_users, capacity):
    rng = np.random.default_rng(seed)
    env = UAVEnv(use_cvx_alloc=True, use_hotspot=False,
                 association_method="marginal",
                 association_local_search_passes=2,
                 association_capacity=capacity)
    env.reset()
    # Two partially overlapping service regions, with mild UAV-position jitter.
    env.loc_uav = np.array([
        [160.0, 250.0], [340.0, 250.0]
    ]) + rng.normal(0.0, 12.0, size=(K, 2))
    env.loc_ue = rng.uniform([40.0, 40.0], [460.0, 460.0], size=(M, 2))
    cluster = rng.integers(0, K, size=n_users)
    env.loc_ue[:n_users, 0] = env.loc_uav[cluster, 0] + rng.normal(0.0, 82.0, n_users)
    env.loc_ue[:n_users, 1] = env.loc_uav[cluster, 1] + rng.normal(0.0, 92.0, n_users)
    env.loc_ue[:n_users, 0] = np.clip(env.loc_ue[:n_users, 0], 0.0, 500.0)
    env.loc_ue[:n_users, 1] = np.clip(env.loc_ue[:n_users, 1], 0.0, 500.0)
    env.active_mask[:] = False
    env.active_mask[:n_users] = True
    env.task_list[:] = 0.0
    env.task_list[:n_users] = rng.integers(TASK_MIN, TASK_MAX + 1, n_users)
    x = np.zeros(M, dtype=np.float64)
    x[:n_users] = rng.uniform(0.2, 1.0, n_users)
    return env, x, n_users


def assignment_value(env, association, context):
    masks = [0] * K
    for i, k in enumerate(association):
        if k >= 0:
            masks[int(k)] |= 1 << i
    return sum(env._association_subset_slrr(k, masks[k], context)
               for k in range(K))


def exact_two_uav_dp(env, context, n_users, capacity):
    """Exact optimum via two 2^M subset tables and max-over-submasks DP."""
    if K != 2:
        raise ValueError("This benchmark's exact DP currently assumes K=2")
    n_masks = 1 << n_users
    full = n_masks - 1
    values = []
    precompute_started = time.perf_counter()
    for k in range(K):
        eligible = 0
        for i in range(n_users):
            if context['coverage'][i] & (1 << k):
                eligible |= 1 << i
        table = np.full(n_masks, -np.inf, dtype=np.float64)
        for mask in range(n_masks):
            if mask & ~eligible or env._mask_size(mask) > capacity:
                continue
            table[mask] = env._association_subset_slrr(k, mask, context)
        values.append(table)
    precompute_ms = (time.perf_counter() - precompute_started) * 1000.0

    combine_started = time.perf_counter()
    best2 = values[1].copy()
    for bit in range(n_users):
        flag = 1 << bit
        blocks = best2.reshape(-1, 2 * flag)
        blocks[:, flag:] = np.maximum(blocks[:, flag:], blocks[:, :flag])
    best_value = float(np.max(values[0] + best2[full ^ np.arange(n_masks)]))
    combine_ms = (time.perf_counter() - combine_started) * 1000.0
    return best_value, precompute_ms, combine_ms


def online_timing(seed, repeats):
    """Measure the default M-user association path with dense overlap."""
    rng = np.random.default_rng(seed)
    env = UAVEnv(use_cvx_alloc=True, use_hotspot=False,
                 association_method="marginal",
                 association_local_search_passes=2)
    env.reset()
    env.loc_uav = np.array([[240.0, 250.0], [260.0, 250.0]])
    env.loc_ue = rng.uniform([125.0, 125.0], [375.0, 375.0], size=(M, 2))
    env.active_mask[:] = True
    env.task_list[:] = rng.integers(TASK_MIN, TASK_MAX + 1, M)
    x = rng.uniform(0.2, 1.0, M)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        env._associate(env.loc_uav, x)
        samples.append((time.perf_counter() - started) * 1000.0)
    return float(np.mean(samples)), float(np.max(samples))


def summarize(name, ratios, times_ms):
    print(f"{name:22s} mean={np.mean(ratios)*100:7.3f}% "
          f"worst={np.min(ratios)*100:7.3f}% "
          f"time={np.mean(times_ms):8.3f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--users", type=int, default=14)
    parser.add_argument("--capacity", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--online-repeats", type=int, default=100)
    parser.add_argument("--local-search-passes", type=int, default=2)
    args = parser.parse_args()
    if not (1 <= args.users <= min(M, 18)):
        parser.error(f"--users must be between 1 and min(configured M={M}, 18)")
    if (args.instances < 1 or args.capacity < 0 or args.online_repeats < 1 or
            args.local_search_passes < 0):
        parser.error("instances and online-repeats must be positive; capacity and passes must be nonnegative")

    names = ("distance", "marginal", "marginal+move/swap")
    ratios = {name: [] for name in names}
    times_ms = {name: [] for name in names}
    exact_precompute_ms = []
    exact_combine_ms = []

    for instance in range(args.instances):
        env, x, n_users = build_instance(args.seed + instance,
                                         args.users, args.capacity)
        context = env._association_value_context(env.loc_uav, x)

        optimum, precompute_ms, combine_ms = exact_two_uav_dp(
            env, context, n_users, args.capacity)
        exact_precompute_ms.append(precompute_ms)
        exact_combine_ms.append(combine_ms)
        if abs(optimum) < 1e-12:
            raise RuntimeError("The exact objective was zero; relative ratios are undefined")

        env.association_method = "distance"
        started = time.perf_counter()
        distance_assignment = env._distance_association(env.loc_uav)
        distance_ms = (time.perf_counter() - started) * 1000.0
        distance_value = assignment_value(env, distance_assignment, context)
        ratios["distance"].append(distance_value / optimum if optimum > 0 else 1.0)
        times_ms["distance"].append(distance_ms)

        env.association_method = "marginal"
        env.association_local_search_passes = 0
        started = time.perf_counter()
        greedy_assignment, _ = env._marginal_gain_association(env.loc_uav, x)
        greedy_ms = (time.perf_counter() - started) * 1000.0
        greedy_value = assignment_value(env, greedy_assignment, context)
        ratios["marginal"].append(greedy_value / optimum if optimum > 0 else 1.0)
        times_ms["marginal"].append(greedy_ms)

        env.association_local_search_passes = args.local_search_passes
        started = time.perf_counter()
        improved_assignment, _ = env._marginal_gain_association(env.loc_uav, x)
        improved_ms = (time.perf_counter() - started) * 1000.0
        improved_value = assignment_value(env, improved_assignment, context)
        ratios["marginal+move/swap"].append(
            improved_value / optimum if optimum > 0 else 1.0)
        times_ms["marginal+move/swap"].append(improved_ms)

        if improved_value + 1e-10 < greedy_value:
            raise AssertionError("Move/swap search reduced the greedy objective")

    print(f"K={K}, M={args.users}, N_max={args.capacity}, "
          f"instances={args.instances}, local-search passes={args.local_search_passes}")
    print(f"Assignment choices: 3^{args.users} = {3 ** args.users:,}")
    print(f"Exact subset-value precompute: {np.mean(exact_precompute_ms):.3f} ms; "
          f"subset-DP combination: {np.mean(exact_combine_ms):.3f} ms")
    for name in names:
        summarize(name, ratios[name], times_ms[name])
    online_mean, online_max = online_timing(args.seed + args.instances,
                                            args.online_repeats)
    print(f"Dense-overlap M={M} online marginal+move/swap: "
          f"mean={online_mean:.3f} ms, max={online_max:.3f} ms "
          f"over {args.online_repeats} repetitions")


if __name__ == "__main__":
    main()
