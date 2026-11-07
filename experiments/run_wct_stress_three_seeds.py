"""Frozen-policy WCT-availability stress test, with paired exogenous seeds."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def serial(x):
    if hasattr(x,'tolist'):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)

def write(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,default=serial,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def priority():
    if os.name=='nt':
        import ctypes
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000)

def sources():
    old=ROOT/'results/original_main_seed42_gpu_retry_20261029_234000'
    other=ROOT/'results/formal_all_five_seeds43_44_restart_20261031/source_snapshot'
    return {s:dict(source=old/'source_snapshot' if s==42 else other,
        folder=old if s==42 else other/f'results/formal_all_five_seeds43_44_session2_20261031_seed{s}') for s in (42,43,44)}

def worker(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    source=sources()[args.worker]['source']; folder=sources()[args.worker]['folder']
    sys.path.insert(0,str(source))
    import config
    from UAV_env import UAVEnv
    from experiment_config import EnvironmentConfig
    from DARS import PERSACAgent
    from state_normalization import StateNormalization
    from evaluation import EpisodeMetrics, frozen_policy, model_digest, summarize, policy_input, configure_policy_environment
    protocol=json.loads((args.output/'protocol.json').read_text(encoding='utf-8'))
    meta=protocol['models'][str(args.worker)]
    manifest=json.loads(Path(meta['manifest']).read_text(encoding='utf-8'))
    assert sha(meta['checkpoint'])==meta['checkpoint_sha256']
    for name,digest in meta['source_hashes'].items():assert sha(source/name)==digest
    for key,value in manifest['system_parameters'].items():
        assert np.array_equal(np.asarray(getattr(config,key)),np.asarray(value)),key
    base=EnvironmentConfig(**manifest['environment'])
    assert (base.initial_energy,base.horizon,base.wct_power)==(40000.,50,200.)
    agent=PERSACAgent();agent.load(meta['checkpoint']);before=model_digest(agent)
    normalizer=StateNormalization(base.battery_capacity)
    seeds=protocol['test_seeds']; reference_exogenous={}; summaries=[]
    destination=args.output/f'seed{args.worker}';destination.mkdir(exist_ok=False)
    progress=dict(status='running',pid=os.getpid(),training_seed=args.worker,cases_completed=0,
                  total_cases=len(protocol['cases']),episodes_completed=0,started_epoch=time.time())
    write(destination/'progress.json',progress)
    try:
        with frozen_policy(agent):
            for case in protocol['cases']:
                indices=tuple(case['wct_indices']); outage=case['active_wct_count']==0
                cfg=replace(base,wct_indices=indices)
                kwargs=dict(manifest['env_kwargs']);kwargs['disable_wct']=outage
                case_dir=destination/case['name'];case_dir.mkdir()
                cm=dict(status='running',case=case,environment=cfg.to_dict(),env_kwargs=kwargs,
                        training_seed=args.worker,model=meta,started_epoch=time.time())
                write(case_dir/'manifest.json',cm)
                rows=[]
                with (case_dir/'episodes.jsonl').open('x',encoding='utf-8') as stream:
                    for seed in seeds:
                        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                        env=UAVEnv(experiment_config=cfg,seed=seed,**kwargs)
                        configure_policy_environment(agent,env);state=env.reset(seed=seed)
                        assert env.n_wct==len(indices) and env.disable_wct==outage
                        assert np.isfinite(state).all()
                        metrics=EpisodeMetrics(env,seed,keep_trace=False)
                        slot_slrr=[];trace=[];exogenous=[]
                        for t in range(cfg.horizon):
                            # User mobility/tasks do not depend on policy or available WCTs.
                            h=hashlib.sha256()
                            for v in (env.loc_ue,env.task_list,env.active_mask):h.update(v.tobytes())
                            exogenous.append(h.hexdigest())
                            if case['active_wct_count']!=4:
                                assert exogenous[-1]==reference_exogenous[seed][t], 'Unpaired external state'
                            active=env.active_mask.copy(); tasks=env.task_list.copy()
                            start=time.perf_counter()
                            action=agent.select_action(policy_input(agent,normalizer,state),noise_std=0.0)
                            policy_seconds=time.perf_counter()-start
                            start=time.perf_counter();state,reward,done,info=env.step(action)
                            metrics.record(info,reward,policy_seconds,time.perf_counter()-start)
                            assert np.isfinite(state).all()
                            executed=bool(info['action_executed'])
                            total=float(np.sum(info['rho'])) if executed else 0.
                            if executed:
                                tb=tasks[active]*config.S_CYCLES/config.F_UE
                                physical=float(np.sum(1-info['delays'][active]/tb))
                                assert abs(physical-total)<1e-9
                                assert abs(total-info['rho_mean']*int(active.sum()))<1e-9
                                if outage: assert np.max(np.abs(info['energy_in']))==0.
                            slot_slrr.append(total)
                            trace.append(dict(slot=t,executed=executed,total_slrr=total,active_users=int(active.sum()),
                                battery_before=info['battery_before'],battery_after_flight=info['battery_after_flight'],
                                battery_after_service=info['battery_after_service'],
                                energy_fly=info['energy_fly'],energy_hover=info['energy_hover'],
                                energy_comp=info['energy_comp'],energy_in=info['energy_in'],
                                termination_reason=info['termination_reason']))
                            if done:break
                        if case['active_wct_count']==4:
                            # Continue an independent shadow environment to generate the full exogenous schedule
                            # even if the baseline policy terminates early. These are not scored states.
                            shadow=UAVEnv(experiment_config=base,seed=seed,**manifest['env_kwargs'])
                            shadow.reset(seed=seed);full=[]
                            for t in range(base.horizon):
                                h=hashlib.sha256()
                                for v in (shadow.loc_ue,shadow.task_list,shadow.active_mask):h.update(v.tobytes())
                                full.append(h.hexdigest())
                                shadow.current_step+=1;shadow._update_ue()
                            assert full[:len(exogenous)]==exogenous
                            reference_exogenous[seed]=full
                        row=metrics.finish()
                        row.update(training_seed=args.worker,case=case['name'],active_wct_count=case['active_wct_count'],
                            active_wct_indices=[] if outage else list(indices),
                            cumulative_total_slrr=float(sum(slot_slrr)),
                            mean_total_slrr_fixed_horizon=float(sum(slot_slrr)/cfg.horizon),
                            mean_total_slrr_executed=float(sum(slot_slrr)/metrics.executed) if metrics.executed else None,
                            external_state_sha256=exogenous,trace=trace)
                        stream.write(json.dumps(row,default=serial,allow_nan=False)+'\n');stream.flush();rows.append(row)
                        progress.update(current_case=case['name'],case_episodes=len(rows),
                            episodes_completed=progress['episodes_completed']+1,updated_epoch=time.time())
                        if len(rows)%10==0 or len(rows)==len(seeds):write(destination/'progress.json',progress)
                summary=summarize(rows)
                for key in ('cumulative_total_slrr','mean_total_slrr_fixed_horizon'):
                    vv=np.array([r[key] for r in rows]);summary[key]=dict(mean=float(vv.mean()),sd=float(vv.std(ddof=1)) if len(vv)>1 else None)
                summary.update(case=case,training_seed=args.worker)
                write(case_dir/'summary.json',summary);summaries.append(summary)
                cm.update(status='completed',episodes=len(rows),ended_epoch=time.time());write(case_dir/'manifest.json',cm)
                progress['cases_completed']+=1;write(destination/'progress.json',progress)
                print(f"seed{args.worker} {case['name']} done: survival={summary['survival_probability']:.3f}, SLRR={summary['mean_total_slrr_fixed_horizon']['mean']:.5f}",flush=True)
        assert before==model_digest(agent) and sha(meta['checkpoint'])==meta['checkpoint_sha256']
        write(destination/'summary.json',dict(status='completed',model_digest_before=before,model_digest_after=model_digest(agent),cases=summaries))
        progress.update(status='completed',ended_epoch=time.time());write(destination/'progress.json',progress)
    except Exception:
        progress.update(status='failed',error=traceback.format_exc());write(destination/'progress.json',progress);raise

def batch(args):
    args.output.mkdir(parents=True,exist_ok=False)
    models={};seeds=list(range(args.seed_start,args.seed_start+args.episodes))
    archive=args.output/'source_snapshot';archive.mkdir();shutil.copy2(__file__,archive/Path(__file__).name)
    for seed,location in sources().items():
        folder=location['folder'];source=location['source'];mf=folder/'per_sac_cvx.manifest.json'
        m=json.loads(mf.read_text(encoding='utf-8'));assert m['status']=='completed' and m['training_seed']==seed
        assert not set(seeds)&set(m['training_episode_seeds']+m['validation_seeds'])
        files=['UAV_env.py','config.py','experiment_config.py','evaluation.py','state_normalization.py','per_sac_agent.py','DARS.py']
        hashes={f:sha(source/f) for f in files}
        assert all(m['source_sha256'].get(f,d)==d for f,d in hashes.items())
        snap=archive/f'training_seed{seed}';snap.mkdir()
        for f in files:shutil.copy2(source/f,snap/f)
        models[seed]=dict(source=source,manifest=mf,checkpoint=folder/'per_sac_cvx.pth',
            checkpoint_sha256=sha(folder/'per_sac_cvx.pth'),source_hashes=hashes)
    cases=[dict(name='wct4_0123',active_wct_count=4,wct_indices=[0,1,2,3])]
    for count in (2,1):
        for subset in combinations(range(4),count):
            cases.append(dict(name=f'wct{count}_'+''.join(map(str,subset)),active_wct_count=count,wct_indices=list(subset)))
    cases.append(dict(name='wct0_outage',active_wct_count=0,wct_indices=[0,1,2,3]))
    protocol=dict(created_epoch=time.time(),status='frozen_before_evaluation',purpose='Frozen existing DARS WCT-availability stress evaluation; no retraining',
        models=models,cases=cases,test_seeds=seeds,total_episodes=len(cases)*3*len(seeds),
        training_version='Existing four-cluster diverse policies; original training reward unchanged.',
        zero_wct='Zero ACTIVE WCTs implemented with original disable_wct=True. Fixed charger landmarks remain in observations and safety candidate generation. This is charging outage, not physically absent infrastructure with a retrained representation.',
        estimands='Full-horizon survival, minimum battery over initial/flight/service states, sum_i rho_i accumulated over horizon and divided by fixed 50 slots; zero service after policy termination. Preserve existing active-user-mean SLRR as a separate diagnostic.',
        pairing='Same independent exogenous seeds across all trained models and station subsets; verify identical user/task state hashes at matching attempted slots.',
        aggregation='Average episodes and all subsets equally within each WCT count and training seed; then average three training seeds and report sample SD. Report each subset separately. Do not pool repeats as independent training replicates.',
        cpu_threads=1,script_sha256=sha(__file__))
    write(args.output/'protocol.json',protocol)
    status=dict(status='running',pid=os.getpid(),started_epoch=time.time(),jobs=[],completed_models=0,total_models=3)
    write(args.output/'batch_status.json',status)
    try:
        for seed in (42,43,44):
            cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--output',str(args.output),'--worker',str(seed)]
            job=dict(training_seed=seed,status='running',command=cmd,started_epoch=time.time());status['jobs'].append(job)
            status['current_training_seed']=seed
            with (args.output/f'seed{seed}.console.log').open('w',encoding='utf-8') as log:
                p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08004000 if os.name=='nt' else 0)
                job['pid']=p.pid;write(args.output/'batch_status.json',status)
                code=p.wait()
            job.update(status='completed' if code==0 else 'failed',exit_code=code,ended_epoch=time.time())
            if code:raise RuntimeError(f'seed{seed} evaluation failed; see console log')
            status['completed_models']+=1;write(args.output/'batch_status.json',status)
        status.update(status='completed',current_training_seed=None,ended_epoch=time.time());write(args.output/'batch_status.json',status)
    except Exception:
        status.update(status='failed',error=traceback.format_exc());write(args.output/'batch_status.json',status);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',type=int);parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--seed-start',type=int,default=911000)
    args=parser.parse_args();args.output=args.output.resolve();priority()
    worker(args) if args.worker else batch(args)
