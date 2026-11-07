#!/usr/bin/env python
"""Frozen DARS association-module comparison; only Greedy advances the world."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import random
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[1]
METHODS = ('greedy', 'matching', 'swap')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, value):
    tmp = Path(path).with_suffix('.tmp')
    tmp.write_text(json.dumps(value, default=serial, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, default=serial, allow_nan=False)+'\n')


def stats(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                p95=float(np.percentile(a, 95)) if len(a) else None)


def summarize(rows, episodes, bootstrap_reps=2000):
    import numpy as np
    output = dict(paired_states=len(rows), episodes=episodes, methods={})
    for method in METHODS:
        results = [r['methods'][method] for r in rows]
        gains = [r['methods'][method]['slrr']-r['methods']['greedy']['slrr'] for r in rows]
        groups = {}
        for row, gain in zip(rows, gains):
            groups.setdefault(row['seed'], []).append(gain)
        ci = None
        if len(groups) >= 2:
            sums = np.array([sum(g) for g in groups.values()])
            counts = np.array([len(g) for g in groups.values()])
            rng = np.random.default_rng(91827)
            indices = rng.integers(0, len(sums), (bootstrap_reps, len(sums)))
            means = sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
            ci = np.percentile(means, [2.5, 97.5]).tolist()
        improvements = [100*(r['methods']['greedy']['J']-r['methods'][method]['J'])/
                        r['methods']['greedy']['J'] for r in rows if r['methods']['greedy']['J'] > 0]
        objective_diffs = np.array([r['methods']['greedy']['J']-r['methods'][method]['J'] for r in rows])
        output['methods'][method] = dict(
            slrr=stats([r['slrr'] for r in results]), slrr_gain=stats(gains),
            slrr_gain_cluster_bootstrap_95ci=ci,
            objective_improvement_percent=stats(improvements),
            total_ms=stats([r['total_ms'] for r in results]),
            selection_ms=stats([r['selection_ms'] for r in results]),
            better_fraction=float(np.mean(objective_diffs > 1e-8)) if rows else None,
            tie_fraction=float(np.mean(np.abs(objective_diffs) <= 1e-8)) if rows else None,
            worse_fraction=float(np.mean(objective_diffs < -1e-8)) if rows else None,
            evaluations=stats([r['evaluations'] for r in results]),
            accepted_changes=stats([r['accepted_changes'] for r in results]),
            budget_exhausted_fraction=float(np.mean([r['budget_exhausted'] for r in results])) if rows else None,
            offload_projection_fraction=float(np.mean([r['projected'] for r in results])) if rows else None,
            associated_users=stats([r['associated_users'] for r in results]),
            offloaded_users=stats([r['offloaded_users'] for r in results]),
            computation_energy_j=stats([np.sum(r['computation_energy_j']) for r in results]),
            feasible_fraction=float(np.mean([r['feasible'] for r in results])) if rows else None)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-manifest', type=Path, default=ROOT/'results/reviewer_direct_objective_base_v3_seed200000_200099_20261030/manifest.json',
                   help='Existing direct-objective evaluation manifest supplying exact model/source/config paths')
    p.add_argument('--model', type=Path)
    p.add_argument('--model-manifest', type=Path)
    p.add_argument('--source-dir', type=Path)
    p.add_argument('--env-config', type=Path)
    p.add_argument('--scenario', choices=['base', 'high_load', 'unseen_order', 'low_battery'], default='base')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=310000)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--stride', type=int, default=1, help='Measure every nth attempted slot, without outcome-based sampling')
    p.add_argument('--timing-repeats', type=int, default=3)
    p.add_argument('--swap-evaluations', type=int, default=1000)
    p.add_argument('--swap-rounds', type=int, default=20)
    p.add_argument('--max-steps', type=int, help='Smoke tests only: truncate rollout; explicitly logged')
    args = p.parse_args()
    if min(args.episodes, args.stride, args.timing_repeats, args.swap_evaluations, args.swap_rounds) <= 0:
        p.error('Counts and budgets must be positive')
    if args.max_steps is not None and args.max_steps <= 0:
        p.error('--max-steps must be positive')
    ref = json.loads(args.reference_manifest.read_text(encoding='utf-8-sig'))
    source = (args.source_dir or Path(ref['training_source_snapshot'])).resolve()
    model = (args.model or Path(ref['model'])).resolve()
    model_manifest = (args.model_manifest or Path(ref['training_manifest'])).resolve()
    config_path = (args.env_config or Path(ref['env_config'])).resolve()
    for path in (source, model, model_manifest, config_path):
        if not path.exists():
            raise FileNotFoundError(path)
    training = json.loads(model_manifest.read_text(encoding='utf-8-sig'))
    if training.get('status') != 'completed' or training.get('algorithm') != 'per_sac_cvx':
        raise ValueError('Require a completed per_sac_cvx checkpoint')
    seeds = list(range(args.seed_start, args.seed_start+args.episodes))
    used = set(training.get('training_episode_seeds', [])) | set(training.get('validation_seeds', []))
    if used.intersection(seeds):
        raise ValueError('Test seeds overlap training/validation seeds')
    # Import the exact checkpoint source, not mutable working-tree training code.
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from DARS import PERSACAgent
    from UAV_env import UAVEnv
    from evaluation import configure_policy_environment, frozen_policy, model_digest, policy_input
    from experiment_config import EnvironmentConfig
    from state_normalization import StateNormalization
    from association_benchmarks import AssociationSnapshot
    from config import K, M, N_MAX, R_COV, F_UAV, BANDWIDTH, HOTSPOT_HOT_PROB, HOTSPOT_COLD_PROB
    torch.set_num_threads(1)
    cfg = EnvironmentConfig.from_json(config_path)
    if args.scenario == 'unseen_order':
        cfg = replace(cfg, hotspot_mode='unseen_order')
    if args.scenario == 'low_battery':
        cfg = replace(cfg, initial_energy=20000.0)
    hot, cold = ((0.98, 0.45) if args.scenario == 'high_load' else
                 (HOTSPOT_HOT_PROB, HOTSPOT_COLD_PROB))
    # reset() reads these module defaults before generating the first tasks.
    # Override them in this evaluation process, leaving training sources intact.
    import UAV_env as evaluation_environment
    evaluation_environment.HOTSPOT_HOT_PROB = hot
    evaluation_environment.HOTSPOT_COLD_PROB = cold
    hashes = {name: sha(source/name) for name in
              ('DARS.py', 'UAV_env.py', 'config.py', 'experiment_config.py', 'evaluation.py', 'state_normalization.py')}
    mismatches = [name for name, digest in hashes.items()
                  if name in training.get('source_sha256', {}) and training['source_sha256'][name] != digest]
    if mismatches:
        raise RuntimeError('Checkpoint source hash mismatch: '+', '.join(mismatches))
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError('Choose a new output directory: '+str(out))
    out.mkdir(parents=True)
    manifest = dict(status='running', started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    arguments=vars(args), model=str(model), model_sha256=sha(model),
                    model_manifest=str(model_manifest), source_dir=str(source), source_sha256=hashes,
                    benchmark_sha256={n: sha(ROOT/'experiments'/n) for n in
                                      ('association_benchmarks.py', 'compare_association.py')},
                    environment=cfg.to_dict(), hot_prob=hot, cold_prob=cold, test_seeds=seeds,
                    constants=dict(K=K, M=M, N_MAX=N_MAX, R_COV=R_COV, F_UAV=F_UAV, BANDWIDTH=BANDWIDTH),
                    runtime=dict(python=sys.executable, version=sys.version, numpy=np.__version__,
                                 torch=torch.__version__, processor=platform.processor(), platform=platform.platform(),
                                 device='CPU', threads=1),
                    design='Frozen Greedy-policy occupancy; adapted matching and swap only evaluated as sidecars',
                    timing='Per-state median of full repeated runs; includes preference construction, search and final resource/safety/delay evaluation; excludes snapshot, IO, policy and movement',
                    references=['10.1109/INFOCOM52122.2024.10621095', '10.1109/JIOT.2025.3542025'])
    write(out/'manifest.json', manifest)
    agent = PERSACAgent()
    agent.load(str(model))
    normalizer = StateNormalization(cfg.battery_capacity)
    before_digest = model_digest(agent)
    rows, episodes = [], []
    warmed = False
    try:
        with frozen_policy(agent):
            for seed in seeds:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                env = UAVEnv(use_cvx_alloc=True, experiment_config=cfg, seed=seed)
                configure_policy_environment(agent, env)
                state = env.reset(seed=seed)
                assert (env._hot_prob, env._cold_prob) == (hot, cold)
                executed = measured = rejected = 0
                reason = None
                steps = min(env.slot_num, args.max_steps or env.slot_num)
                for step in range(steps):
                    action = agent.select_action(policy_input(agent, normalizer, state), noise_std=0.0)
                    previous = copy.copy(env)
                    for name in ('active_mask', 'task_list', 'loc_ue', 'loc_uav', 'e_battery_uav'):
                        setattr(previous, name, np.array(getattr(env, name), copy=True))
                    state, reward, done, info = env.step(action)
                    reason = info.get('termination_reason')
                    if not info['action_executed']:
                        rejected += 1
                        append(out/'rejected.jsonl', dict(seed=seed, step=step, reason=reason))
                    else:
                        executed += 1
                        if np.any(previous.active_mask) and step % args.stride == 0:
                            snap = AssociationSnapshot(previous, env.loc_uav, action, info)
                            world_before = pickle.dumps(env, protocol=5)
                            if not warmed:
                                for method in METHODS:
                                    snap.run(method, args.swap_evaluations, args.swap_rounds)
                                warmed = True
                            repetitions = {m: [] for m in METHODS}
                            for repeat in range(args.timing_repeats):
                                offset = (seed+step+repeat) % len(METHODS)
                                order = METHODS[offset:]+METHODS[:offset]
                                for method in order:
                                    repetitions[method].append(snap.run(method, args.swap_evaluations, args.swap_rounds))
                            results = {}
                            for method, reps in repetitions.items():
                                result = reps[0]
                                for extra in reps[1:]:
                                    if not np.array_equal(result['association'], extra['association']) or abs(result['J']-extra['J']) > 1e-10:
                                        raise AssertionError('Repeated method changed output')
                                result['timing_repetitions_ms'] = [r['total_ms'] for r in reps]
                                result['total_ms'] = float(np.median(result['timing_repetitions_ms']))
                                result['selection_ms'] = float(np.median([r['selection_ms'] for r in reps]))
                                results[method] = result
                            # Verify adapter exactly reproduces actual Greedy execution.
                            g = results['greedy']
                            if not np.array_equal(g['association'], info['association']):
                                raise AssertionError('Greedy association reconstruction mismatch')
                            for ours, actual in [('offload','offload_ratios'), ('cpu','cpu_allocation'),
                                                 ('bandwidth','bandwidth_allocation'), ('delay','delays'),
                                                 ('computation_energy_j','energy_comp')]:
                                np.testing.assert_allclose(g[ours], info[actual], rtol=1e-10, atol=1e-9)
                            if pickle.dumps(env, protocol=5) != world_before:
                                raise AssertionError('Sidecar mutated rollout environment')
                            row = dict(seed=seed, step=step, active_users=int(np.sum(snap.active)),
                                       active_mask=snap.active, tasks=previous.task_list,
                                       ue_positions=previous.loc_ue, candidate_uav_positions=snap.positions,
                                       battery_before=previous.e_battery_uav, raw_action=action,
                                       raw_offload_proposal=snap.proposal, computation_budget_j=snap.budget,
                                       energy_fly=snap.fly, energy_hover=snap.hover, energy_in=snap.harvest,
                                       methods=results)
                            rows.append(row)
                            append(out/'state_results.jsonl', row)
                            measured += 1
                    if done:
                        break
                episode = dict(seed=seed, executed_steps=executed, measured_states=measured,
                               rejected_attempts=rejected, termination_reason=reason,
                               survived_greedy_rollout=executed == env.slot_num,
                               smoke_truncated=steps < env.slot_num)
                episodes.append(episode)
                append(out/'episode_results.jsonl', episode)
                manifest['episodes_completed'] = len(episodes)
                write(out/'manifest.json', manifest)
                print(f"[{len(episodes)}/{len(seeds)}] seed={seed} states={measured} executed={executed} rejected={rejected}", flush=True)
        after_digest = model_digest(agent)
        if after_digest != before_digest or sha(model) != manifest['model_sha256']:
            raise AssertionError('Policy/checkpoint changed')
        summary = summarize(rows, episodes)
        write(out/'summary.json', summary)
        lines = ['# Frozen-policy association comparison', '',
                 '| Method | Mean SLRR | SLRR gain | Mean / P95 total ms | Search budget hit |',
                 '|---|---:|---:|---:|---:|']
        for method, result in summary['methods'].items():
            lines.append(f"| {method} | {result['slrr']['mean']} | {result['slrr_gain']['mean']} | {result['total_ms']['mean']} / {result['total_ms']['p95']} | {result['budget_exhausted_fraction']} |")
        lines += ['', 'Conditional sidecar comparison only; not a global optimum or a closed-loop replacement-policy comparison.']
        (out/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        manifest.update(status='completed', model_digest_before=before_digest, model_digest_after=after_digest,
                        completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        write(out/'manifest.json', manifest)
        print('Completed: '+str(out), flush=True)
    except Exception:
        manifest.update(status='failed', error=traceback.format_exc())
        write(out/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
