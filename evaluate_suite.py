"""Evaluate saved models on common seeds without retraining.

Smoke presets are development checks, never publication evidence. Full presets
are candidate stress ranges; freeze the final protocol before formal testing.
"""
import argparse
import csv
from dataclasses import replace
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time
import numpy as np
import torch
import config
from experiment_config import EnvironmentConfig
from evaluation import evaluate_policy, summarize, write_json, model_digest, runtime_info
from train_unified import ALGO_REGISTRY


def scenarios(base, suite):
    cases = [('in_distribution', base)]
    if suite == 'smoke':
        cases += [('initial_20000', replace(base, initial_energy=min(20000., base.battery_capacity))),
                  ('horizon_100', replace(base, horizon=100)),
                  ('wct_0_2', replace(base, wct_indices=(0, 2))),
                  ('power_100', replace(base, wct_power=100.)),
                  ('joint', replace(base, initial_energy=min(20000., base.battery_capacity),
                                    horizon=100, wct_indices=(0, 2), wct_power=100.))]
    elif suite in ('energy', 'all'):
        cases += [(f'initial_{int(e)}', replace(base, initial_energy=e)) for e in (30000., 20000., 10000.)
                  if e <= base.battery_capacity and e != base.initial_energy]
        cases += [(f'horizon_{t}', replace(base, horizon=t)) for t in (100, 200, 500) if t != base.horizon]
        # All subsets, reported separately: do not hide placement effects.
        for n in (3, 2, 1):
            for subset in combinations(base.wct_indices, n):
                cases.append(('wct_' + '_'.join(map(str, subset)), replace(base, wct_indices=subset)))
        cases += [(f'power_{int(p)}', replace(base, wct_power=p)) for p in (180., 160., 100.) if p != base.wct_power]
        cases += [('joint', replace(base, initial_energy=min(20000., base.battery_capacity),
                                   horizon=200, wct_indices=(0, 2), wct_power=160.))]
    if suite in ('smoke', 'hotspot', 'all'):
        cases += [('unseen_order', replace(base, hotspot_mode='unseen_order')),
                  ('unseen_timing', replace(base, hotspot_mode='unseen_timing'))]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', nargs='+', required=True, help='One manifest per algorithm/training seed')
    parser.add_argument('--suite', choices=['smoke', 'standard', 'energy', 'hotspot', 'all', 'custom'], default='smoke')
    parser.add_argument('--scenario-file', help='Custom JSON mapping case names to environment overrides')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=200000)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--no-traces', action='store_true')
    args = parser.parse_args()
    if args.episodes <= 0 or args.seed < 0:
        parser.error('Require positive episodes and nonnegative seed')
    custom_cases = None
    if args.suite == 'custom':
        if not args.scenario_file:
            parser.error('--suite custom requires --scenario-file')
        custom_cases = json.loads(Path(args.scenario_file).read_text(encoding='utf-8'))
        allowed = {'initial_energy', 'horizon', 'wct_indices', 'wct_power', 'hotspot_mode'}
        if not isinstance(custom_cases, dict) or not custom_cases:
            parser.error('Custom cases must be a nonempty object')
        for name, overrides in custom_cases.items():
            if not name or not all(c.isalnum() or c in '_-' for c in name):
                parser.error('Case names must contain only letters, digits, underscores or hyphens')
            if not isinstance(overrides, dict) or set(overrides) - allowed:
                parser.error('Custom overrides must preserve capacity, time encoding and training/test partitions')
    torch.set_num_threads(args.threads)
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Choose a new/empty output directory')
    destination.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed, args.seed + args.episodes))
    records, case_summaries = [], []
    start = time.perf_counter()
    for index, manifest_path in enumerate(args.manifests):
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['status'] != 'completed':
            raise ValueError('Training manifest is not completed')
        if set(seeds) & set(manifest['training_episode_seeds'] + manifest['validation_seeds']):
            raise ValueError('Evaluation seeds overlap training/validation seeds')
        # Preserve physical model and dimensions. Environment overrides live in
        # EnvironmentConfig, not mutations of config.py.
        for key, value in manifest['system_parameters'].items():
            actual = getattr(config, key)
            if not np.array_equal(np.asarray(actual), np.asarray(value)):
                raise ValueError(f'config.py differs from training at {key}; restore training configuration')
        base = EnvironmentConfig(**manifest['environment'])
        agent, needs_training, defaults = ALGO_REGISTRY[manifest['algorithm']]()
        env_kwargs = manifest['env_kwargs']
        if args.suite in ('smoke', 'hotspot', 'all') and not env_kwargs.get('use_hotspot', True):
            raise ValueError('Hotspot evaluation requires hotspot-enabled training')
        checkpoint = manifest_path.with_name(manifest['algorithm'] + '.pth')
        checkpoint_hash = None
        if needs_training:
            agent.load(str(checkpoint))
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        before = model_digest(agent)
        case_records = []
        cases = ([(name, replace(base, **overrides)) for name, overrides in custom_cases.items()]
                 if custom_cases is not None else scenarios(base, args.suite))
        for name, cfg in cases:
            rows = evaluate_policy(agent, cfg, env_kwargs, seeds, keep_trace=not args.no_traces)
            summary = summarize(rows)
            summary.update(case=name, algorithm=manifest['algorithm'], training_seed=manifest['training_seed'],
                           model_index=index, environment=cfg.to_dict())
            path = f'model{index}_{name}.json'
            write_json(destination / path, dict(summary=summary, episodes=rows))
            case_summaries.append(summary)
            case_records.append(dict(case=name, file=path, environment=cfg.to_dict()))
            print(f"{manifest['algorithm']} seed={manifest['training_seed']} {name}: "
                  f"survival={summary['survival_probability']:.3f}, "
                  f"SLRR(T)={summary['slrr_fixed_horizon']['mean']:.4f}, "
                  f"Bmin={summary['min_battery_j']['mean']:.2f} J", flush=True)
        after = model_digest(agent)
        if before != after:
            raise AssertionError('Evaluation changed network parameters/buffers')
        records.append(dict(manifest=manifest, checkpoint_sha256=checkpoint_hash,
                            model_digest_before=before, model_digest_after=after, cases=case_records))
    # Keep within-model Wilson intervals separate; episodes across training seeds
    # are not silently pooled as independent Bernoulli observations.
    groups = {}
    for row in case_summaries:
        key = (row['algorithm'], row['case'], json.dumps(row['environment'], sort_keys=True))
        groups.setdefault(key, []).append(row)
    aggregate = []
    for (algo, case, _), group in groups.items():
        if len({row['training_seed'] for row in group}) != len(group):
            raise ValueError('Duplicate training seed in an aggregation group')
        item = dict(algorithm=algo, case=case, training_seeds=[r['training_seed'] for r in group],
                    environment=group[0]['environment'])
        for metric in ('survival_probability', 'min_battery_j', 'slrr_fixed_horizon', 'mean_energy_per_executed_slot_j', 'decision_mean_ms'):
            values = [r[metric] if metric == 'survival_probability' else r[metric]['mean'] for r in group]
            values = [x for x in values if x is not None]
            item[metric] = dict(mean=float(np.mean(values)) if values else None,
                                between_training_seed_std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
        aggregate.append(item)
    with (destination / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['algorithm', 'training_seed', 'case', 'episodes',
            'survival_probability', 'min_battery_mean_j', 'slrr_fixed_horizon', 'energy_per_slot_j', 'decision_mean_ms'])
        writer.writeheader()
        for row in case_summaries:
            writer.writerow(dict(algorithm=row['algorithm'], training_seed=row['training_seed'], case=row['case'],
                episodes=row['episodes'], survival_probability=row['survival_probability'],
                min_battery_mean_j=row['min_battery_j']['mean'], slrr_fixed_horizon=row['slrr_fixed_horizon']['mean'],
                energy_per_slot_j=row['mean_energy_per_executed_slot_j']['mean'], decision_mean_ms=row['decision_mean_ms']['mean']))
    write_json(destination / 'report.json', dict(suite=args.suite,
        purpose='development smoke check; not converged policy evidence' if args.suite == 'smoke' else 'fixed-policy evaluation',
        seeds=seeds, runtime=runtime_info(), elapsed_seconds=time.perf_counter()-start,
        models=records, per_model_summary=case_summaries, across_training_seeds=aggregate,
        timing_note='Mean/P95 per episode over all attempted decisions, including rejections; summaries average episode statistics. No learning updates.',
        survival_ci_note='Wilson intervals within each trained model only. Across models report mean and between-seed SD.'))


if __name__ == '__main__':
    main()
