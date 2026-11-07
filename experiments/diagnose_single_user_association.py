"""Single-user reassociation audit of existing frozen-DARS snapshots."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'results/reviewer_phase2_three_seeds_20261101/formal'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if hasattr(x,'tolist'):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def write(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,default=serial,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def append(stream,obj):stream.write(json.dumps(obj,default=serial,allow_nan=False)+'\n')

def stats(values):
    import numpy as np
    v=np.asarray(values,dtype=float)
    return dict(n=len(v),mean=float(v.mean()),p95=float(np.percentile(v,95)),minimum=float(v.min()),maximum=float(v.max())) if len(v) else dict(n=0,mean=None,p95=None,minimum=None,maximum=None)

def summarize(users,candidates,states):
    eligible=[u for u in users if u['feasible_alternatives']>0]
    faster=[c for c in candidates if c['feasible'] and c['user_delay_improved']]
    feasible=[c for c in candidates if c['feasible']]
    return dict(states=len(states),active_user_states=sum(s['active_users'] for s in states),
        greedy_associated_user_states=len(users),unassociated_active_user_states=sum(s['unassociated_active'] for s in states),
        associated_but_zero_offload_user_states=sum(u['greedy_offload']==0 for u in users),
        with_geometric_alternative=sum(u['coverage_alternatives']>0 for u in users),
        with_capacity_eligible_alternative=sum(u['capacity_eligible_alternatives']>0 for u in users),
        with_feasible_alternative=len(eligible),
        no_alternative_reason_counts=dict(Counter(u['reason'] for u in users if not u['feasible_alternatives'])),
        worse_latency_user_states=sum(u['any_user_delay_improved'] for u in eligible),
        system_improving_user_states=sum(u['any_system_slrr_improved'] for u in eligible),
        jointly_improving_user_states=sum(u['any_joint_improvement'] for u in eligible),
        eligible_fraction=len(eligible)/len(users) if users else None,
        worse_latency_fraction_among_eligible=sum(u['any_user_delay_improved'] for u in eligible)/len(eligible) if eligible else None,
        system_improving_fraction_among_eligible=sum(u['any_system_slrr_improved'] for u in eligible)/len(eligible) if eligible else None,
        jointly_improving_fraction_among_eligible=sum(u['any_joint_improvement'] for u in eligible)/len(eligible) if eligible else None,
        candidate_count=len(candidates),feasible_candidate_count=len(feasible),
        faster_user_but_system_worse=sum(c['system_slrr_worsened'] for c in faster),
        user_delay_reduction_s_all=stats([c['user_delay_reduction_s'] for c in feasible]),
        user_delay_reduction_s_improved=stats([c['user_delay_reduction_s'] for c in faster]),
        user_delay_reduction_percent_improved=stats([c['user_delay_reduction_percent'] for c in faster]),
        system_total_slrr_gain_all=stats([c['system_total_slrr_gain'] for c in feasible]),
        system_total_slrr_gain_faster_users=stats([c['system_total_slrr_gain'] for c in faster]),
        faster_users_with_more_cpu=sum(c['alternative_cpu_hz']>c['greedy_cpu_hz']+1e-8*max(1,c['greedy_cpu_hz']) for c in faster),
        faster_users_with_more_bandwidth=sum(c['alternative_bandwidth_hz']>c['greedy_bandwidth_hz']+1e-8*max(1,c['greedy_bandwidth_hz']) for c in faster),
        faster_users_with_more_battery=sum(c['target_battery_before_j']>c['source_battery_before_j']+1e-6 for c in faster),
        faster_users_with_more_computation_budget=sum(c['target_computation_budget_j']>c['source_computation_budget_j']+1e-6 for c in faster),
        faster_users_with_changed_executed_offload=sum(abs(c['alternative_offload']-c['greedy_offload'])>1e-8 for c in faster),
        faster_users_with_farther_uav=sum(c['target_distance_m']>c['source_distance_m']+1e-8 for c in faster),
        energy_excess_j_max=max((c['energy_excess_j'] for c in feasible),default=0),
        greedy_reconstruction_max_error=max((s['reconstruction_error'] for s in states),default=0),
        states_with_any_user_improvement=sum(s['any_user_delay_improved'] for s in states),
        states_with_any_system_improvement=sum(s['any_system_slrr_improved'] for s in states))

def worker(args):
    import numpy as np
    inp=INPUT/'association'/args.job
    m=json.loads((inp/'manifest.json').read_text(encoding='utf-8'));assert m['status']=='completed'
    source=Path(m['source_dir']);bench=INPUT/'source_snapshot'
    for f,d in m['source_sha256'].items():assert sha(source/f)==d,(f,'training source changed')
    assert sha(bench/'association_benchmarks.py')==m['benchmark_sha256']['association_benchmarks.py']
    sys.path.insert(0,str(source));sys.path.insert(0,str(bench))
    from UAV_env import UAVEnv
    from experiment_config import EnvironmentConfig
    from association_benchmarks import AssociationSnapshot
    import config
    assert config.K==2, 'This audit has exactly one alternative UAV per associated user'
    out=args.output/args.job;out.mkdir(exist_ok=False)
    input_file=inp/'state_results.jsonl';input_hash=sha(input_file)
    meta=dict(status='running',input=str(input_file),input_sha256=input_hash,training_source=str(source),
        checkpoint_sha256=m['model_sha256'],association_benchmark_sha256=m['benchmark_sha256']['association_benchmarks.py'],
        source_sha256=m['source_sha256'],started_epoch=time.time(),pid=os.getpid(),
        delay_improvement_tolerance='1e-8*max(1,T_greedy) seconds',system_slrr_tolerance=1e-8)
    write(out/'manifest.json',meta)
    env=UAVEnv(use_cvx_alloc=True,experiment_config=EnvironmentConfig(**m['environment']),seed=999)
    users=[];candidates=[];states=[]
    try:
        with input_file.open(encoding='utf-8') as inpstream, (out/'users.jsonl').open('x',encoding='utf-8') as uf, (out/'candidates.jsonl').open('x',encoding='utf-8') as cf, (out/'states.jsonl').open('x',encoding='utf-8') as sf:
            for line in inpstream:
                row=json.loads(line);original=row['methods']['greedy']
                env.active_mask=np.array(row['active_mask'],dtype=bool);env.task_list=np.array(row['tasks'])
                env.loc_ue=np.array(row['ue_positions']);env.loc_uav=np.array(row['candidate_uav_positions'])
                env.e_battery_uav=np.array(row['battery_before'])
                info={k:np.array(row[k]) for k in ('energy_fly','energy_hover','energy_in')}
                snap=AssociationSnapshot(env,np.array(row['candidate_uav_positions']),np.array(row['raw_action']),info)
                a=np.array(original['association'],dtype=int)
                np.testing.assert_array_equal(env._distance_association(snap.positions),a)
                baseline=snap.evaluate(a);assert baseline['feasible']
                err=0.
                for key in ('offload','cpu','bandwidth','delay','rho','computation_energy_j'):
                    np.testing.assert_allclose(baseline[key],original[key],rtol=1e-10,atol=1e-8)
                    err=max(err,float(np.max(np.abs(baseline[key]-np.array(original[key])))))
                assert abs(baseline['J']-original['J'])<1e-10
                world=pickle.dumps(snap,protocol=5)
                counts=np.bincount(a[a>=0],minlength=config.K)
                dist=np.linalg.norm(env.loc_ue[:,None,:]-snap.positions[None,:,:],axis=2)
                st=dict(test_seed=row['seed'],slot=row['step'],active_users=int(snap.active.sum()),
                    associated_active=int(np.sum(snap.active&(a>=0))),unassociated_active=int(np.sum(snap.active&(a<0))),
                    reconstruction_error=err,any_user_delay_improved=False,any_system_slrr_improved=False)
                for i in np.flatnonzero(snap.active&(a>=0)):
                    k=int(a[i]); target=1-k
                    user=dict(test_seed=row['seed'],slot=row['step'],user=int(i),greedy_uav=k,
                        greedy_offload=float(baseline['offload'][i]),coverage_alternatives=0,capacity_eligible_alternatives=0,
                        feasible_alternatives=0,any_user_delay_improved=False,any_system_slrr_improved=False,
                        any_joint_improvement=False,reason='outside_alternative_coverage')
                    if snap.eligible[i,target]:
                        user['coverage_alternatives']=1;user['reason']='alternative_capacity_full'
                        if counts[target]<config.N_MAX:
                            user['capacity_eligible_alternatives']=1;user['reason']='alternative_energy_infeasible'
                            other=a.copy();other[i]=target
                            assert np.sum(other!=a)==1
                            alt=snap.evaluate(other)
                            gain=float(baseline['delay'][i]-alt['delay'][i]);sysgain=float(baseline['J']-alt['J'])
                            own=float(alt['rho'][i]-baseline['rho'][i])
                            faster=gain>1e-8*max(1,float(baseline['delay'][i]));sysbetter=sysgain>1e-8
                            candidate=dict(test_seed=row['seed'],slot=row['step'],user=int(i),source_uav=k,target_uav=target,
                                feasible=alt['feasible'],user_delay_improved=bool(faster),system_slrr_improved=bool(sysbetter),
                                system_slrr_worsened=bool(sysgain < -1e-8),
                                greedy_delay_s=float(baseline['delay'][i]),alternative_delay_s=float(alt['delay'][i]),
                                user_delay_reduction_s=gain,user_delay_reduction_percent=100*gain/float(baseline['delay'][i]),
                                user_slrr_gain=own,system_total_slrr_gain=sysgain,other_users_total_slrr_gain=sysgain-own,
                                greedy_cpu_hz=float(baseline['cpu'][i]),alternative_cpu_hz=float(alt['cpu'][i]),
                                greedy_bandwidth_hz=float(baseline['bandwidth'][i]),alternative_bandwidth_hz=float(alt['bandwidth'][i]),
                                greedy_offload=float(baseline['offload'][i]),alternative_offload=float(alt['offload'][i]),
                                raw_offload_proposal=float(snap.proposal[i]),source_battery_before_j=float(env.e_battery_uav[k]),
                                target_battery_before_j=float(env.e_battery_uav[target]),source_computation_budget_j=float(snap.budget[k]),
                                target_computation_budget_j=float(snap.budget[target]),
                                source_associated_users_before=int(counts[k]),target_associated_users_before=int(counts[target]),
                                source_distance_m=float(dist[i,k]),target_distance_m=float(dist[i,target]),
                                greedy_uav_comp_energy_j=baseline['computation_energy_j'],alternative_uav_comp_energy_j=alt['computation_energy_j'],
                                greedy_safety_scales=baseline['scales'],alternative_safety_scales=alt['scales'],
                                all_user_slrr_delta=alt['rho']-baseline['rho'],energy_excess_j=alt['energy_excess_j'])
                            append(cf,candidate);candidates.append(candidate)
                            if alt['feasible']:
                                user.update(feasible_alternatives=1,reason='feasible_alternative',any_user_delay_improved=bool(faster),
                                    any_system_slrr_improved=bool(sysbetter),any_joint_improvement=bool(faster and sysbetter))
                                st['any_user_delay_improved']|=bool(faster);st['any_system_slrr_improved']|=bool(sysbetter)
                    append(uf,user);users.append(user)
                assert pickle.dumps(snap,protocol=5)==world,'Reassociation mutated snapshot'
                append(sf,st);states.append(st)
                if len(states)%50==0:
                    uf.flush();cf.flush();sf.flush();write(out/'progress.json',dict(states=len(states),users=len(users),candidates=len(candidates)))
        assert sha(input_file)==input_hash
        summary=summarize(users,candidates,states);write(out/'summary.json',summary)
        meta.update(status='completed',states=len(states),ended_epoch=time.time());write(out/'manifest.json',meta)
        print(args.job,json.dumps(summary),flush=True)
    except Exception:
        meta.update(status='failed',error=traceback.format_exc());write(out/'manifest.json',meta);raise

def batch(args):
    args.output.mkdir(parents=True,exist_ok=False)
    archive=args.output/'source_snapshot';archive.mkdir()
    shutil.copy2(__file__,archive/Path(__file__).name)
    shutil.copy2(INPUT/'source_snapshot/association_benchmarks.py',archive/'association_benchmarks.py')
    protocol=dict(created_epoch=time.time(),input=str(INPUT),training_seeds=[42,43,44],scenarios=['base','high_load','low_battery'],
        population='All active users assigned by Greedy in all existing 2576 measured states. Unassociated users counted separately; no filtering by performance.',
        intervention='Change only one associated user to the other covered UAV if association quota permits; all other associations fixed. Recompute allocations and the original safety corrections for all users.',
        fixed='Positions, tasks, channels, battery, raw Actor offload proposals, flight/hover/charging energy and resource budgets.',
        outcomes='User completion latency; CPU and bandwidth allocation; battery and computation budget; safety scaling; own and other-user SLRR changes.',
        scope='Conditional one-user relocation audit, not global association optimization or full-horizon policy comparison. No user swaps or drops; excluded capacity-full alternatives do not prove optimality. Resource/battery differences are co-occurring descriptors, not isolated causal attribution.',
        counting='Eligible denominator is active Greedy-associated user-state pairs with a feasible single relocation. All exclusions and raw numerators retained. Summaries also report each training seed.',
        delay_tolerance='1e-8*max(1,T_greedy) seconds',system_slrr_tolerance=1e-8,script_sha256=sha(__file__))
    write(args.output/'protocol.json',protocol)
    batch=dict(status='running',pid=os.getpid(),jobs=[],completed_jobs=0,total_jobs=9,started_epoch=time.time());write(args.output/'batch_status.json',batch)
    try:
        for scene in protocol['scenarios']:
            for seed in protocol['training_seeds']:
                name=f'seed{seed}_{scene}';job=dict(name=name,status='running');batch['jobs'].append(job);batch['current_job']=name
                cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--output',str(args.output),'--job',name]
                with (args.output/f'{name}.log').open('w',encoding='utf-8') as log:
                    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08004000 if os.name=='nt' else 0)
                    job['pid']=p.pid;write(args.output/'batch_status.json',batch);code=p.wait()
                job.update(status='completed' if code==0 else 'failed',exit_code=code)
                if code:raise RuntimeError(f'{name} failed; see log')
                batch['completed_jobs']+=1;write(args.output/'batch_status.json',batch)
        batch.update(status='completed',current_job=None,ended_epoch=time.time());write(args.output/'batch_status.json',batch)
    except Exception:
        batch.update(status='failed',error=traceback.format_exc());write(args.output/'batch_status.json',batch);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--job')
    args=p.parse_args();args.output=args.output.resolve()
    worker(args) if args.job else batch(args)
