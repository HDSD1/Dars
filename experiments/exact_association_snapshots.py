"""Exact fixed-state association search under the existing resource/safety mapping.

For two UAVs, enumerate each UAV's feasible user subsets and combine disjoint
shared-user masks. Local execution has zero SLRR. Resource and service-energy
budgets are per UAV; positions/charging/raw offload proposals are fixed.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'results/reviewer_phase2_three_seeds_20261101/formal'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def serial(x):
    if hasattr(x,'tolist'):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,default=serial,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def worker(args):
    import numpy as np
    protocol=json.loads((args.output/'protocol.json').read_text(encoding='utf-8'))
    meta=protocol['inputs'][str(args.worker)]
    source=Path(meta['source_dir']); snapshot=Path(meta['states'])
    assert sha(snapshot)==meta['states_sha256']
    for name,digest in meta['source_sha256'].items():assert sha(source/name)==digest
    sys.path.insert(0,str(source));sys.path.insert(0,str(args.output/'source_snapshot'))
    from config import K,M,N_MAX
    from UAV_env import UAVEnv
    from experiment_config import EnvironmentConfig
    from association_benchmarks import AssociationSnapshot
    assert K==2
    out=args.output/f'seed{args.worker}';out.mkdir(exist_ok=False)
    env=UAVEnv(use_cvx_alloc=True,experiment_config=EnvironmentConfig(**meta['environment']),seed=999)
    progress=dict(status='running',states_completed=0,subset_evaluations=0,bruteforce_checks=0,started_epoch=time.time())
    write(out/'progress.json',progress)
    try:
        with snapshot.open(encoding='utf-8') as inp,(out/'states.jsonl').open('x',encoding='utf-8') as stream:
            for line in inp:
                row=json.loads(line)
                env.active_mask=np.array(row['active_mask'],dtype=bool);env.task_list=np.array(row['tasks'])
                env.loc_ue=np.array(row['ue_positions']);env.loc_uav=np.array(row['candidate_uav_positions'])
                env.e_battery_uav=np.array(row['battery_before'])
                info={key:np.array(row[key]) for key in ['energy_fly','energy_hover','energy_in']}
                snap=AssociationSnapshot(env,env.loc_uav,np.array(row['raw_action']),info)
                shared=np.flatnonzero(np.all(snap.eligible,axis=1)).tolist()
                bits={u:1<<j for j,u in enumerate(shared)}
                mask_count=1<<len(shared)
                scores=[];groups=[];evaluations=0
                started=time.perf_counter()
                for k in range(K):
                    users=np.flatnonzero(snap.eligible[:,k]).tolist()
                    best=np.full(mask_count,-np.inf);best_groups=[None]*mask_count
                    for count in range(min(len(users),N_MAX)+1):
                        for chosen in itertools.combinations(users,count):
                            association=np.full(M,-1,dtype=int);association[list(chosen)]=k
                            result=snap.evaluate(association);evaluations+=1
                            if not result['feasible']:continue
                            value=float(np.sum(result['rho']))
                            mask=sum(bits.get(u,0) for u in chosen)
                            if value>best[mask]+1e-12:
                                best[mask]=value;best_groups[mask]=chosen
                    scores.append(best);groups.append(best_groups)
                # Subset-max DP: best UAV-1 subset contained in the complement
                # of the shared users already assigned to UAV 0.
                best1=scores[1].copy();arg1=np.arange(mask_count)
                for bit in range(len(shared)):
                    for mask in range(mask_count):
                        if mask & (1<<bit):
                            smaller=mask^(1<<bit)
                            if best1[smaller]>best1[mask]+1e-12:
                                best1[mask]=best1[smaller];arg1[mask]=arg1[smaller]
                combined=scores[0]+best1[(mask_count-1)^np.arange(mask_count)]
                best0=int(np.argmax(combined));best_other=int(arg1[(mask_count-1)^best0])
                association=np.full(M,-1,dtype=int)
                association[list(groups[0][best0])]=0
                association[list(groups[1][best_other])]=1
                optimal=snap.evaluate(association)
                optimum=float(np.sum(optimal['rho']))
                assert optimal['feasible'] and abs(optimum-float(combined[best0]))<1e-9
                search_ms=(time.perf_counter()-started)*1000
                original=snap.evaluate(np.array(row['methods']['greedy']['association'],dtype=int))
                np.testing.assert_allclose(original['rho'],row['methods']['greedy']['rho'],rtol=0,atol=1e-10)
                for method in ['greedy','matching','swap']:
                    assert optimum+1e-9>=float(np.sum(row['methods'][method]['rho']))
                # Independently count all coverage/capacity-feasible joint mappings.
                dp={(0,0):1};choices=[]
                for i in np.flatnonzero(snap.active):
                    options=[-1]+np.flatnonzero(snap.eligible[i]).tolist();choices.append((int(i),options))
                    nxt={}
                    for (n0,n1),number in dp.items():
                        for k in options:
                            key=(n0+(k==0),n1+(k==1))
                            if max(key)<=N_MAX:nxt[key]=nxt.get(key,0)+number
                    dp=nxt
                n_joint=sum(dp.values())
                brute_checked=False
                # Predeclared check: first five states per model with <=512 maps.
                if progress['bruteforce_checks']<5 and n_joint<=512:
                    brute=-np.inf
                    for selected in itertools.product(*(options for _,options in choices)):
                        candidate=np.full(M,-1,dtype=int)
                        for (i,_),k in zip(choices,selected):candidate[i]=k
                        if not snap.valid_association(candidate):continue
                        result=snap.evaluate(candidate)
                        if result['feasible']:brute=max(brute,float(np.sum(result['rho'])))
                    assert abs(brute-optimum)<1e-9
                    progress['bruteforce_checks']+=1;brute_checked=True
                greedy=np.array(row['methods']['greedy']['association'])
                counts=dict(coverage_capacity_feasible_joint_maps=n_joint,per_uav_subsets_evaluated=evaluations,
                    shared_users=len(shared),covered_users_per_uav=snap.eligible.sum(axis=0))
                output=dict(training_seed=args.worker,test_seed=row['seed'],slot=row['step'],active_users=int(snap.active.sum()),
                    greedy_total_slrr=float(np.sum(row['methods']['greedy']['rho'])),
                    matching_total_slrr=float(np.sum(row['methods']['matching']['rho'])),
                    swap_total_slrr=float(np.sum(row['methods']['swap']['rho'])),
                    exact_total_slrr=optimum,exact_gain=optimum-float(np.sum(row['methods']['greedy']['rho'])),
                    changed_associations=int(np.sum(association!=greedy)),
                    newly_associated=int(np.sum((association>=0)&(greedy<0))),
                    moved_between_uavs=int(np.sum((association>=0)&(greedy>=0)&(association!=greedy))),
                    returned_to_local=int(np.sum((association<0)&(greedy>=0))),
                    exact_association=association,greedy_association=greedy,
                    exact_offload=optimal['offload'],exact_cpu=optimal['cpu'],exact_bandwidth=optimal['bandwidth'],
                    exact_delays=optimal['delay'],exact_rho=optimal['rho'],energy_excess_j=optimal['energy_excess_j'],
                    exact_computation_energy_j=optimal['computation_energy_j'],search_ms=search_ms,
                    counts=counts,bruteforce_checked=brute_checked)
                stream.write(json.dumps(output,default=serial,allow_nan=False)+'\n')
                progress['states_completed']+=1;progress['subset_evaluations']+=evaluations
                if progress['states_completed']%50==0:
                    stream.flush();progress['updated_epoch']=time.time();write(out/'progress.json',progress)
                    print(f"seed{args.worker}: {progress['states_completed']}/500 states",flush=True)
        assert sha(snapshot)==meta['states_sha256']
        assert progress['bruteforce_checks']==5
        progress.update(status='completed',ended_epoch=time.time());write(out/'progress.json',progress)
    except Exception:
        progress.update(status='failed',error=traceback.format_exc());write(out/'progress.json',progress);raise

def batch(args):
    args.output.mkdir(parents=True,exist_ok=False)
    archive=args.output/'source_snapshot';archive.mkdir()
    shutil.copy2(__file__,archive/Path(__file__).name)
    shutil.copy2(INPUT/'source_snapshot/association_benchmarks.py',archive/'association_benchmarks.py')
    inputs={}
    for seed in [42,43,44]:
        folder=INPUT/'association'/f'seed{seed}_base'
        m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        inputs[seed]={k:m[k] for k in ['source_dir','source_sha256','environment','model_sha256']}
        inputs[seed].update(states=str(folder/'state_results.jsonl'),states_sha256=sha(folder/'state_results.jsonl'))
    protocol=dict(inputs=inputs,scope='Exact association optimum over all coverage/capacity-feasible user assignments, INCLUDING local execution, under the existing closed-form resource and energy-safety mapping. Not joint P1 optimum and not a reference optimum over independently variable CPU/bandwidth.',
        fixed='UAV positions, user tasks/channels, raw Actor offloading proposals, flight/hover energy, charging, battery and per-UAV budgets. Resource allocation and safety scaling recomputed for every selected subset.',
        method='Enumerate every feasible user subset for each UAV. Per-UAV SLRR sums and energy constraints are separable at fixed positions/charging; optimize disjoint overlapping users by subset-max dynamic programming.',
        sample='All 1500 nominal snapshots from three frozen trained policies; no outcome selection.',
        checks='Reconstruct Greedy at every state; validate exact winning full assignment against sum of subset scores; compare all three existing heuristics; independently brute-force first five states per trained seed with <=512 joint assignments.',
        benchmark_sha256=sha(archive/'association_benchmarks.py'),script_sha256=sha(__file__),started_epoch=time.time())
    write(args.output/'protocol.json',protocol)
    status=dict(status='running',completed_models=0,started_epoch=time.time());write(args.output/'batch_status.json',status)
    try:
        for seed in [42,43,44]:
            status.update(current_training_seed=seed);write(args.output/'batch_status.json',status)
            with (args.output/f'seed{seed}.console.log').open('w',encoding='utf-8') as log:
                p=subprocess.Popen([sys.executable,'-B','-u',str(archive/Path(__file__).name),'--output',str(args.output),'--worker',str(seed)],
                    stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0)
                status.update(worker_pid=p.pid);write(args.output/'batch_status.json',status)
                code=p.wait()
            if code:raise RuntimeError(f'seed{seed} failed; inspect log')
            status['completed_models']+=1;write(args.output/'batch_status.json',status)
            print(f'Completed seed {seed}',flush=True)
        status.update(status='completed',current_training_seed=None,ended_epoch=time.time());write(args.output/'batch_status.json',status)
    except Exception:
        status.update(status='failed',error=traceback.format_exc());write(args.output/'batch_status.json',status);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--worker',type=int)
    args=p.parse_args();args.output=args.output.resolve()
    worker(args) if args.worker else batch(args)
