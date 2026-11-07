"""Closed-loop frozen-policy association replacement; never edits training sources."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from types import MethodType

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'results/reviewer_phase2_three_seeds_20261101/formal'
METHODS = ['greedy', 'matching', 'swap']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def serial(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, default=serial, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)

def append(stream, value):
    stream.write(json.dumps(value, default=serial, allow_nan=False)+'\n')

def worker(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    protocol = json.loads((args.output/'protocol.json').read_text(encoding='utf-8'))
    meta = protocol['models'][str(args.worker)]
    source = Path(meta['source_dir'])
    for f, digest in meta['source_sha256'].items():
        assert sha(source/f) == digest
    assert sha(meta['model']) == meta['model_sha256']
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(args.output/'source_snapshot'))
    import config
    from UAV_env import UAVEnv
    from experiment_config import EnvironmentConfig
    from DARS import PERSACAgent
    from evaluation import frozen_policy, model_digest, policy_input, configure_policy_environment
    from state_normalization import StateNormalization
    from association_benchmarks import AssociationSnapshot

    class ReplacementEnv(UAVEnv):
        def __init__(self, method, **kwargs):
            self.replacement_method = method
            self.last_selection = None
            self.max_adapter_error = 0.
            super().__init__(**kwargs)

        def _project_movement_for_energy(self, *values):
            projected = super()._project_movement_for_energy(*values)
            self.movement_result = projected
            return projected

        def _distance_association(self, positions):
            result = self.movement_result
            info = dict(energy_fly=result[3], energy_in=result[5],
                        energy_hover=np.full(config.K, self._hover_power()*config.T_COM))
            snapshot = AssociationSnapshot(self, positions, self.current_raw_action, info)
            # Snapshot methods must call the original distance initializer, not this hook.
            snapshot.env._distance_association = MethodType(UAVEnv._distance_association, snapshot.env)
            self.last_selection = snapshot.run(self.replacement_method, 1000, 20)
            return self.last_selection['association'].copy()

        def step(self, action):
            self.current_raw_action = np.asarray(action).copy()
            self.last_selection = None
            state, reward, done, info = super().step(action)
            if info['action_executed']:
                selected = self.last_selection
                assert selected is not None and selected['feasible']
                np.testing.assert_array_equal(selected['association'], info['association'])
                for expected, actual in [('offload','offload_ratios'), ('cpu','cpu_allocation'),
                                         ('bandwidth','bandwidth_allocation'), ('delay','delays'),
                                         ('computation_energy_j','energy_comp')]:
                    np.testing.assert_allclose(selected[expected], info[actual], rtol=1e-10, atol=1e-9)
                    self.max_adapter_error = max(self.max_adapter_error, float(np.max(np.abs(selected[expected]-info[actual]))))
            return state, reward, done, info

    cfg = EnvironmentConfig(**meta['environment'])
    kwargs = dict(use_cvx_alloc=True, experiment_config=cfg)
    agent = PERSACAgent(); agent.load(meta['model'])
    normalizer = StateNormalization(cfg.battery_capacity)
    digest_before = model_digest(agent)
    out = args.output/f'seed{args.worker}'
    out.mkdir(exist_ok=False)
    progress = dict(status='running',training_seed=args.worker,episodes_completed=0,
                    total_episodes=3*len(protocol['test_seeds']),started_epoch=time.time())
    write(out/'progress.json',progress)

    def action_for(state):
        return agent.select_action(policy_input(agent,normalizer,state),noise_std=0.)

    def external_hash(env):
        h = hashlib.sha256()
        for v in (env.loc_ue,env.task_list,env.active_mask):
            h.update(v.tobytes())
        return h.hexdigest()

    max_native_error = 0.
    try:
        with frozen_policy(agent):
            # One full trajectory checks that the adapter leaves Greedy behavior unchanged.
            seed = protocol['verification_seed']
            native = UAVEnv(seed=seed,**kwargs)
            wrapped = ReplacementEnv('greedy',seed=seed,**kwargs)
            configure_policy_environment(agent,native); configure_policy_environment(agent,wrapped)
            sn = native.reset(seed=seed); sw = wrapped.reset(seed=seed)
            for slot in range(cfg.horizon):
                np.testing.assert_array_equal(sn,sw)
                action = action_for(sn)
                sn,rn,dn,inn = native.step(action)
                sw,rw,dw,inw = wrapped.step(action)
                assert dn == dw and rn == rw
                for key in ['association','offload_ratios','cpu_allocation','bandwidth_allocation','delays','rho','battery_after_service']:
                    np.testing.assert_array_equal(inn[key],inw[key])
                max_native_error = max(max_native_error,float(np.max(np.abs(sn-sw))))
                if dn:
                    break
            write(out/'verification.json',dict(native_greedy_exact_match=True,max_observation_error=max_native_error,
                                              verification_seed=seed,attempted_slots=slot+1))
            with (out/'episodes.jsonl').open('x',encoding='utf-8') as ef, (out/'steps.jsonl').open('x',encoding='utf-8') as sf:
                for episode_idx,seed in enumerate(protocol['test_seeds']):
                    # Exogenous trajectories are generated independently of policy survival.
                    shadow = UAVEnv(seed=seed,**kwargs); shadow.reset(seed=seed)
                    schedule = []
                    for slot in range(cfg.horizon):
                        schedule.append(external_hash(shadow))
                        shadow.current_step += 1; shadow._update_ue()
                    offset = episode_idx % 3
                    for method in METHODS[offset:]+METHODS[:offset]:
                        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                        env = ReplacementEnv(method,seed=seed,**kwargs)
                        configure_policy_environment(agent,env); state = env.reset(seed=seed)
                        total = 0.; executed = 0; delays = []; association_ms = []; decision_ms = []
                        minimum = float(env.e_battery_uav.min()); comp = fly = hover = harvested = 0.
                        max_violation = 0.; reason = None; budget_hits = 0
                        for slot in range(cfg.horizon):
                            assert external_hash(env) == schedule[slot], 'Unpaired exogenous state'
                            active = env.active_mask.copy(); tasks = env.task_list.copy()
                            start = time.perf_counter(); action = action_for(state)
                            actor_ms = (time.perf_counter()-start)*1000
                            state,reward,done,info = env.step(action)
                            assert np.isfinite(state).all()
                            selected = env.last_selection
                            is_executed = bool(info['action_executed'])
                            slot_slrr = float(np.sum(info['rho'])) if is_executed else 0.
                            pipeline_ms = actor_ms + 1000*info.get('decision_seconds',0.)
                            decision_ms.append(pipeline_ms)
                            minimum = min(minimum, float(np.min(info['battery_before'])),float(np.min(info['battery_after_flight'])),float(np.min(info['battery_after_service'])))
                            if is_executed:
                                executed += 1; total += slot_slrr
                                np.testing.assert_allclose(np.sum(1-info['delays'][active]/(tasks[active]*config.S_CYCLES/config.F_UE)),slot_slrr,atol=1e-9)
                                delays.extend(info['delays'][active].tolist())
                                comp += float(np.sum(info['energy_comp'])); fly += float(np.sum(info['energy_fly']))
                                hover += float(np.sum(info['energy_hover'])); harvested += float(np.sum(info['energy_in']))
                                max_violation = max(max_violation,float(np.max(info['flight_energy_violation'])),float(np.max(info['service_energy_violation'])))
                                association_ms.append(selected['total_ms'])
                                budget_hits += int(selected['budget_exhausted'])
                            reason = info['termination_reason']
                            append(sf,dict(training_seed=args.worker,test_seed=seed,method=method,slot=slot,
                                executed=is_executed,total_slrr=slot_slrr,active_users=int(active.sum()),
                                association=info['association'],offload=info['offload_ratios'],uav_positions=env.loc_uav,
                                battery_after_service=info['battery_after_service'],energy_comp=info['energy_comp'],
                                actor_ms=actor_ms,association_allocation_ms=selected['total_ms'] if selected else None,
                                prototype_decision_ms=pipeline_ms,termination_reason=reason))
                            if done:
                                break
                        row=dict(training_seed=args.worker,test_seed=seed,method=method,executed_slots=executed,
                            survived=executed==cfg.horizon,termination_reason=reason,cumulative_total_slrr=total,
                            mean_total_slrr_fixed_horizon=total/cfg.horizon,min_battery_j=minimum,
                            final_battery_j=env.e_battery_uav,comp_energy_j=comp,flight_energy_j=fly,hover_energy_j=hover,
                            harvested_energy_j=harvested,mean_delay_executed_active_s=float(np.mean(delays)) if delays else None,
                            association_ms_mean=float(np.mean(association_ms)) if association_ms else None,
                            prototype_decision_ms_mean=float(np.mean(decision_ms)),max_energy_violation_j=max_violation,
                            max_adapter_error=env.max_adapter_error,swap_budget_hits=budget_hits)
                        append(ef,row);ef.flush();sf.flush()
                        progress.update(episodes_completed=progress['episodes_completed']+1,current_test_seed=seed,current_method=method,updated_epoch=time.time())
                    if (episode_idx+1)%10==0 or episode_idx==0 or episode_idx+1==len(protocol['test_seeds']):
                        write(out/'progress.json',progress)
                        print(f'training seed {args.worker}: {progress["episodes_completed"]}/{progress["total_episodes"]} episodes',flush=True)
        assert model_digest(agent)==digest_before and sha(meta['model'])==meta['model_sha256']
        progress.update(status='completed',ended_epoch=time.time(),model_unchanged=True)
        write(out/'progress.json',progress)
    except Exception:
        progress.update(status='failed',error=traceback.format_exc());write(out/'progress.json',progress);raise

def batch(args):
    args.output.mkdir(parents=True,exist_ok=False)
    archive=args.output/'source_snapshot';archive.mkdir()
    shutil.copy2(__file__,archive/Path(__file__).name)
    shutil.copy2(REFERENCE/'source_snapshot/association_benchmarks.py',archive/'association_benchmarks.py')
    seeds=list(range(args.seed_start,args.seed_start+args.episodes));models={}
    for seed in [42,43,44]:
        m=json.loads((REFERENCE/'association'/f'seed{seed}_base'/'manifest.json').read_text(encoding='utf-8'))
        training=json.loads(Path(m['model_manifest']).read_text(encoding='utf-8-sig'))
        assert training['status']=='completed'
        assert not set(seeds+[args.seed_start-1]) & set(training.get('training_episode_seeds',[])+training.get('validation_seeds',[]))
        models[seed]={k:m[k] for k in ['source_dir','source_sha256','model','model_sha256','model_manifest','environment']}
    protocol=dict(status='prespecified',models=models,test_seeds=seeds,verification_seed=args.seed_start-1,
        methods=METHODS,total_episodes=3*3*args.episodes,scenario='nominal only',swap_evaluation_budget=1000,swap_round_budget=20,
        intervention='Only association rule replaced. Selected association enters original environment step and affects resources, battery, subsequent observations and Actor actions. Original movement filtering, resource allocation, safety, and reward unchanged.',
        policy='Three frozen policies trained with distance Greedy; evaluates plug-in replacement, not retrained Matching/Swap policies.',
        metric='Sum over users and slots of SLRR, divided by fixed 50-slot horizon. Rejected and post-termination slots contribute zero. Also report survival, minimum battery, execution energy, and conditional delay.',
        pairing='Same independent exogenous user/task seeds; verify full exogenous hashes at every executed or attempted slot. Rotate method order between episodes.',
        timing='CPU 1 thread. association_allocation_ms is the benchmark search including resource/safety/delay evaluations. prototype_decision_ms additionally includes Actor/movement/common environment allocation pass; excludes post-step audit and physical state update. No onboard or uncontended timing claim.',
        validation='Native vs adapted Greedy full trajectory exact match for each model; every executed candidate checked against original environment final allocation and delay; frozen model digests verified.',
        started_epoch=time.time(),script_sha256=sha(__file__),benchmark_sha256=sha(archive/'association_benchmarks.py'))
    write(args.output/'protocol.json',protocol)
    status=dict(status='running',completed_models=0,total_models=3,pid=os.getpid(),started_epoch=time.time())
    write(args.output/'batch_status.json',status)
    try:
        for seed in [42,43,44]:
            status.update(current_training_seed=seed);write(args.output/'batch_status.json',status)
            command=[sys.executable,'-B','-u',str(archive/Path(__file__).name),'--output',str(args.output),'--worker',str(seed)]
            # Worker resolves all training paths from the saved protocol.
            with (args.output/f'seed{seed}.console.log').open('w',encoding='utf-8') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0)
                status.update(worker_pid=process.pid);write(args.output/'batch_status.json',status)
                code=process.wait()
            if code:
                raise RuntimeError(f'Worker {seed} exited {code}; inspect seed{seed}.console.log')
            status['completed_models']+=1;write(args.output/'batch_status.json',status)
            print(f'Completed training seed {seed}',flush=True)
        status.update(status='completed',current_training_seed=None,ended_epoch=time.time());write(args.output/'batch_status.json',status)
    except Exception:
        status.update(status='failed',error=traceback.format_exc());write(args.output/'batch_status.json',status);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',type=int)
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--seed-start',type=int,default=931000)
    args=parser.parse_args();args.output=args.output.resolve()
    worker(args) if args.worker else batch(args)
