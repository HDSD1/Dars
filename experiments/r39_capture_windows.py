"""Capture full-load, real frozen-DARS windows for the R3.9 joint reference.

This script only reads completed checkpoints/snapshots. No tasks, user masks,
batteries or model parameters are altered during rollout. Each invocation uses
one source snapshot so Python's imported configuration cannot mix user counts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]


def serial(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, default=serial, ensure_ascii=False,
                               indent=2, allow_nan=False)+'\n', encoding='utf-8')


def catalog():
    phase = json.loads((ROOT/'results/reviewer_phase2_three_seeds_20261101/formal/protocol.json').read_text(encoding='utf-8'))
    entries = {}
    for seed, entry in phase['models'].items():
        entries[30, int(seed)] = {key: Path(entry[key]) for key in ('source', 'checkpoint', 'manifest')}
    sweep = ROOT/'results/reviewer_users_m20_m25_m30_seed42_5000_20261102'
    for users in (20, 25):
        source = sweep/f'm{users}/source_snapshot'
        folder = source/f'results/formal_users_m{users}_seed42'
        entries[users, 42] = dict(source=source, checkpoint=folder/'per_sac_cvx.pth',
                                  manifest=folder/'per_sac_cvx.manifest.json')
    return entries


def capture(args):
    import numpy as np
    import torch
    entry = catalog()[args.users, args.training_seed]
    training = json.loads(entry['manifest'].read_text(encoding='utf-8'))
    assert training['status'] == 'completed'
    assert training['training_seed'] == args.training_seed
    assert training['agent_parameters']['M'] == args.users
    seeds = list(range(args.seed_start, args.seed_start+args.episodes))
    used = set(training['training_episode_seeds']) | set(training['validation_seeds'])
    assert not used.intersection(seeds), 'Test seed overlaps training/validation'
    hashes = {name: sha(entry['source']/name) for name in
              ('config.py', 'UAV_env.py', 'DARS.py', 'experiment_config.py',
               'evaluation.py', 'state_normalization.py')}
    for name, digest in hashes.items():
        assert digest == training['source_sha256'][name], (name, 'snapshot hash mismatch')
    sys.path.insert(0, str(entry['source']))
    import config as c
    from DARS import PERSACAgent
    from UAV_env import UAVEnv
    from experiment_config import EnvironmentConfig
    from evaluation import frozen_policy, policy_input, model_digest
    from state_normalization import StateNormalization
    assert c.M == args.users and c.K == 2
    torch.set_num_threads(1)
    cfg = EnvironmentConfig(**training['environment'])
    env_kwargs = training.get('env_kwargs', dict(use_cvx_alloc=True))
    env_kwargs = {k: v for k, v in env_kwargs.items()
                  if k in ('use_cvx_alloc', 'use_hotspot', 'disable_wct')}
    assert env_kwargs.get('use_cvx_alloc', True)
    env_kwargs.setdefault('use_cvx_alloc', True)
    agent = PERSACAgent()
    agent.load(str(entry['checkpoint']))
    model_before = model_digest(agent)
    checkpoint_hash = sha(entry['checkpoint'])
    normalizer = StateNormalization(cfg.battery_capacity)
    constants = {name: getattr(c, name) for name in dir(c) if name.isupper()
                 and isinstance(getattr(c, name), (int, float))}
    meta = dict(status='running', created_epoch=time.time(), pid=os.getpid(),
                arguments=vars(args), model=entry, model_sha256=checkpoint_hash,
                model_manifest_sha256=sha(entry['manifest']), source_sha256=hashes,
                training_manifest_source_sha256=training['source_sha256'],
                environment=cfg.to_dict(), env_kwargs=env_kwargs,
                network_users=c.M, training_seed=args.training_seed,
                hot_prob=c.HOTSPOT_HOT_PROB, cold_prob=c.HOTSPOT_COLD_PROB,
                test_seeds=seeds, start_slots=args.start_slots, horizon=args.horizon,
                low_battery_rule='min_k B_k(start)/battery_capacity <= 0.2',
                selection='All predefined start slots. Complete windows retained regardless of reward; incomplete windows recorded separately.',
                script_sha256=sha(Path(__file__)), episodes_completed=0,
                windows_completed=0, low_battery_windows=0, incomplete_windows=0)
    write(args.output/'manifest.json', meta)
    with (args.output/'cases.jsonl').open('x', encoding='utf-8') as cases, \
         (args.output/'incomplete_windows.jsonl').open('x', encoding='utf-8') as incomplete, \
         (args.output/'episodes.jsonl').open('x', encoding='utf-8') as episodes, \
         frozen_policy(agent):
        for seed in seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            env = UAVEnv(experiment_config=cfg, seed=seed, **env_kwargs)
            state = env.reset(seed=seed)
            traces = []
            attempts = []
            for slot in range(env.slot_num):
                users = env.loc_ue.copy()
                tasks = env.task_list.copy()
                active = env.active_mask.copy()
                ids = np.flatnonzero(active).tolist()
                before = env.loc_uav.copy()
                battery = env.e_battery_uav.copy()
                action = agent.select_action(policy_input(agent, normalizer, state), noise_std=0.0)
                state, reward, done, info = env.step(action)
                attempts.append(dict(slot=slot, action_executed=bool(info['action_executed']),
                                     termination_reason=info['termination_reason']))
                if info['action_executed']:
                    shape = (len(ids), c.K)
                    matrices = {name: np.zeros(shape) for name in
                                ('association', 'x_ik', 'fraction_cpu', 'bwshare')}
                    for row, uid in enumerate(ids):
                        k = int(info['association'][uid])
                        if k >= 0:
                            matrices['association'][row,k] = 1
                            matrices['x_ik'][row,k] = info['offload_ratios'][uid]
                            matrices['fraction_cpu'][row,k] = info['cpu_allocation'][uid]/c.F_UAV
                            matrices['bwshare'][row,k] = info['bandwidth_allocation'][uid]/c.BANDWIDTH
                    z = np.asarray(info['delays'])[ids]/(tasks[ids]*c.S_CYCLES/c.F_UE)
                    trace = dict(slot=slot, user_positions=users, tasks=tasks, active_mask=active,
                                 active_indices=ids, hotspot_phase=int(info['hotspot_phase']),
                                 positions_before=before, positions_after=env.loc_uav.copy(),
                                 battery=battery, battery_before=battery,
                                 battery_after=np.asarray(info['battery_after_service']).copy(),
                                 battery_after_flight=np.asarray(info['battery_after_flight']).copy(),
                                 time_normalized=z, rho=np.asarray(info['rho'])[ids],
                                 total_slrr=float(np.sum(info['rho'])), training_reward=float(reward),
                                 association_users=np.asarray(info['association']).copy(),
                                 raw_action=np.asarray(action).copy(),
                                 executed_action=np.asarray(info['executed_action']).copy(),
                                 speeds=np.array([(info['executed_action'][3*k+2]+1)*c.V_MAX/2 for k in range(c.K)]),
                                 charge_power=np.asarray(info['charge_power']).copy(),
                                 movement_projected=bool(info['movement_projected']),
                                 offload_projected=bool(info['offload_projected']), **matrices)
                    for key in ('fly', 'hover', 'comp', 'in', 'out', 'charge_accepted', 'charge_overflow'):
                        value = np.asarray(info['energy_'+key]).copy()
                        trace['energy_'+key] = value
                        trace['energy_'+key+'_j'] = value
                    traces.append(trace)
                if done:
                    break
            by_slot = {r['slot']: r for r in traces}
            for start in args.start_slots:
                wanted = list(range(start, start+args.horizon))
                missing = [slot for slot in wanted if slot not in by_slot]
                identity = dict(case_id=f'm{c.M}_train{args.training_seed}_test{seed}_t{start}_h{args.horizon}',
                                network_users=c.M, training_seed=args.training_seed,
                                test_seed=seed, start_slot=start, horizon=args.horizon)
                if missing:
                    row = dict(**identity, missing_slots=missing, executed_slots=len(traces),
                               termination_reason=attempts[-1]['termination_reason'],
                               reason='DARS did not execute every slot in the requested window')
                    incomplete.write(json.dumps(row, default=serial, allow_nan=False)+'\n')
                    meta['incomplete_windows'] += 1
                    continue
                window = [by_slot[slot] for slot in wanted]
                low = bool(np.min(window[0]['battery_before'])/cfg.battery_capacity <= .2)
                stage_keys = ('slot', 'user_positions', 'tasks', 'active_mask', 'active_indices', 'hotspot_phase')
                row = dict(**identity, constants=constants, environment=cfg.to_dict(), env_kwargs=env_kwargs,
                           wct_positions=env.wct_positions.copy(),
                           initial_positions=window[0]['positions_before'],
                           initial_battery=window[0]['battery_before'],
                           initial_hotspot_phase=window[0]['hotspot_phase'],
                           terminal_battery_dars=window[-1]['battery_after'],
                           low_battery=low,
                           stages=[{k: step[k] for k in stage_keys} for step in window],
                           dars=window, dars_total_slrr=sum(s['total_slrr'] for s in window),
                           hotspot_schedule=env.hotspot_schedule[start:start+args.horizon].copy(),
                           hotspot_segments=env.hotspot_segments,
                           source=str(entry['source']), checkpoint=str(entry['checkpoint']),
                           checkpoint_sha256=checkpoint_hash, source_sha256=hashes)
                cases.write(json.dumps(row, default=serial, allow_nan=False)+'\n')
                meta['windows_completed'] += 1
                meta['low_battery_windows'] += int(low)
            episodes.write(json.dumps(dict(test_seed=seed, executed_slots=len(traces),
                                           attempts=attempts, survived=len(traces)==cfg.horizon),
                                      default=serial, allow_nan=False)+'\n')
            for stream in (cases, incomplete, episodes): stream.flush()
            meta['episodes_completed'] += 1
            meta['updated_epoch'] = time.time()
            write(args.output/'manifest.json', meta)
            print(f"M{c.M}/train{args.training_seed}: {meta['episodes_completed']}/{args.episodes} episodes, "
                  f"{meta['windows_completed']} complete windows, {meta['low_battery_windows']} low battery", flush=True)
    assert model_before == model_digest(agent), 'Policy changed during capture'
    assert checkpoint_hash == sha(entry['checkpoint']), 'Checkpoint changed during capture'
    for name, digest in hashes.items(): assert digest == sha(entry['source']/name)
    meta.update(status='completed', ended_epoch=time.time(), model_unchanged=True,
                cases_sha256=sha(args.output/'cases.jsonl'))
    write(args.output/'manifest.json', meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--users', type=int, choices=(20,25,30), default=30)
    p.add_argument('--training-seed', type=int, choices=(42,43,44), default=42)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed-start', type=int, default=970000)
    p.add_argument('--horizon', type=int, choices=(3,5), default=5)
    p.add_argument('--start-slots', type=lambda s: [int(v) for v in s.split(',')], default=[0,10,20,30,40])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if (args.users,args.training_seed) not in catalog():
        p.error('No registered completed model for that user count/training seed')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    capture(args)


if __name__ == '__main__':
    main()
