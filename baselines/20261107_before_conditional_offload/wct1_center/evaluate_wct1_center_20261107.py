"""Frozen complete-rollout evaluation with physical metrics and slot timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from run_wct1_center_20261107 import OUT, ROOT, ALGOS, src, result, read, sha, now
TRAIN_SEED = 42

def evaluate(m,algo,smoke=False):
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='2'
    sys.path.insert(0,str(src(m)))
    import numpy as np
    import torch
    torch.set_num_threads(2)
    import config
    from UAV_env import UAVEnv
    from experiment_config import EnvironmentConfig
    from train_unified import ALGO_REGISTRY
    from state_normalization import StateNormalization
    from evaluation import EpisodeMetrics, frozen_policy, policy_input, configure_policy_environment, sync_device, write_json, runtime_info, model_digest
    folder=result(m,algo,smoke);meta=read(folder/f'{algo}.manifest.json');checkpoint=folder/f'{algo}.pth'
    assert meta['status']=='completed' and config.M==m and config.K==2
    cfg=EnvironmentConfig(**meta['environment']);agent,_,kwargs=ALGO_REGISTRY[algo]();agent.load(str(checkpoint))
    normalizer=StateNormalization(cfg.battery_capacity);before=model_digest(agent)
    dest=OUT/('smoke_evaluation' if smoke else 'evaluation')/f'seed{TRAIN_SEED}'/f'm{m}'/algo;dest.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',started_at=now(),M=m,algorithm=algo,training_seed=TRAIN_SEED,training_episodes=meta['training']['episodes'],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source_manifest_sha256=sha(folder/f'{algo}.manifest.json'),runtime=runtime_info(),environment=cfg.to_dict(),seed_start=931000,test_episodes=2 if smoke else 100,
        timing_scope='preprocessing+policy CUDA synchronized+safety+association+allocation; includes rejected attempts; excludes training, environment reward/battery updates and file logging',
        hardware_scope='Local GPU; own training ended before evaluation. Other processes not excluded. Diagnostic workstation timing, not uncontended or airborne hardware measurements.',
        energy_scope='Simulated UAV flight/hover/task-computation energy, not neural-inference or control-radio energy.',
        statistics='Only one freshly trained seed42 model. Test-episode bootstrap CI is conditional on this model, not training-seed robustness.')
    write_json(dest/'manifest.json',manifest);rows=[];times=[]
    def exohash(env):
        h=hashlib.sha256()
        for a in (env.loc_ue,env.task_list,env.active_mask):h.update(a.tobytes())
        return h.hexdigest()
    with frozen_policy(agent),(dest/'episodes.jsonl').open('x',encoding='utf-8') as f:
        for seed in range(931000,931000+manifest['test_episodes']):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            shadow=UAVEnv(experiment_config=cfg,seed=seed,**kwargs);shadow.reset(seed=seed);external=[]
            for _ in range(cfg.horizon):
                external.append(exohash(shadow));shadow.current_step+=1;shadow._update_ue()
            env=UAVEnv(experiment_config=cfg,seed=seed,**kwargs);state=env.reset(seed=seed);configure_policy_environment(agent,env)
            rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            for _ in range(5):agent.select_action(policy_input(agent,normalizer,state),noise_std=0.0)
            sync_device();random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
            metrics=EpisodeMetrics(env,seed,keep_trace=False);slots=[]
            for slot in range(cfg.horizon):
                assert exohash(env)==external[slot]
                active=env.active_mask.copy();tasks=env.task_list.copy()
                sync_device();start=time.perf_counter()
                action=agent.select_action(policy_input(agent,normalizer,state),noise_std=0.0)
                sync_device();policy_seconds=time.perf_counter()-start
                start=time.perf_counter();state,reward,done,info=env.step(action);env_seconds=time.perf_counter()-start
                metrics.record(info,reward,policy_seconds,env_seconds)
                executed=bool(info['action_executed']);total=float(np.sum(info['rho'])) if executed else 0.
                if executed:
                    baseline=tasks[active]*config.S_CYCLES/config.F_UE
                    physical=float(np.sum(1-info['delays'][active]/baseline))
                    assert abs(physical-total)<1e-8
                    assert abs(total-info['rho_mean']*int(active.sum()))<1e-8
                slots.append(dict(slot=slot,executed=executed,total_slrr=total,decision_ms=1000*(policy_seconds+info['decision_seconds']),policy_ms=1000*policy_seconds,
                    safety_association_allocation_ms=1000*info['decision_seconds'],battery_after_service_j=info['battery_after_service'],energy_fly_j=info['energy_fly'],energy_hover_j=info['energy_hover'],energy_comp_j=info['energy_comp'],termination_reason=info['termination_reason']))
                if done:break
            row=metrics.finish();assert row['energy_violation_steps']==0
            assert abs(sum(x['total_slrr'] for x in slots)-row['cumulative_total_slrr'])<1e-8
            row.update(algorithm=algo,M=m,training_seed=TRAIN_SEED,external_state_hashes=external,slots=slots)
            from evaluation import jsonable
            f.write(json.dumps(row,default=jsonable,allow_nan=False)+'\n');f.flush();rows.append(row)
            times.extend(x['decision_ms'] for x in slots)
            write_json(dest/'progress.json',dict(status='running',episodes=len(rows),target=manifest['test_episodes'],updated_at=now()))
    # Every rollout was compared to an independently generated external trajectory.
    assert before==model_digest(agent)
    fields=['reward','mean_total_slrr_fixed_horizon','cumulative_total_slrr','delay_executed','survived','executed_steps','min_battery_j','energy_fly_j','energy_hover_j','energy_comp_j','energy_out_j','energy_in_j']
    rng=np.random.default_rng(20261102);summary=dict(M=m,algorithm=algo,training_seed=TRAIN_SEED,training_episodes=meta['training']['episodes'],test_episodes=len(rows),decision_mean_ms=float(np.mean(times)),decision_p95_ms=float(np.percentile(times,95)),timed_attempts=len(times),energy_violation_steps=0)
    for field in fields:
        vals=np.array([r[field] for r in rows if r[field] is not None],dtype=float)
        bs=rng.choice(vals,size=(2000,len(vals)),replace=True).mean(axis=1) if len(vals) else np.array([np.nan])
        summary[field]=dict(mean=float(vals.mean()) if len(vals) else None,ci95=np.percentile(bs,[2.5,97.5]).tolist() if len(vals) else None,n=len(vals))
    write_json(dest/'summary.json',summary)
    manifest.update(status='completed',ended_at=now(),model_unchanged=True,physical_slrr_verified=True,external_states_verified=True);write_json(dest/'manifest.json',manifest)
    write_json(dest/'progress.json',dict(status='completed',episodes=len(rows),updated_at=now()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--m',type=int,required=True);p.add_argument('--algo',choices=ALGOS,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();evaluate(a.m,a.algo,a.smoke)
