"""Read paired snapshots, solve independent conic reference and cross-check SLSQP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from conic_resource_reference import solve_group, self_test
import clarabel
import compare_direct_objective as old


def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def stats(values):
    a=np.asarray([v for v in values if v is not None],dtype=float)
    return dict(n=len(a),mean=float(a.mean()),p95=float(np.percentile(a,95)),
                maximum=float(a.max()),minimum=float(a.min())) if len(a) else dict(n=0,mean=None,p95=None)


def run(args):
    inp=Path(args.input).resolve(); out=Path(args.output).resolve()
    m=json.loads((inp/'manifest.json').read_text(encoding='utf-8'))
    assert m['status']=='completed'
    sys.path.insert(0,m['source_dir'])
    import config as c
    from UAV_env import UAVEnv, OFFLOAD_EPS
    from experiment_config import EnvironmentConfig
    const={name:float(getattr(c,name)) for name in
           ('F_UAV','BANDWIDTH','F_UE','S_CYCLES','KAPPA','P_TX','SIGMA2')}
    const.update(K=int(c.K),OFFLOAD_EPS=float(OFFLOAD_EPS))
    sourcefile=inp/'state_results.jsonl'
    input_hash=hashlib.sha256(sourcefile.read_bytes()).hexdigest()
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',input=str(inp),input_sha256=input_hash,
                  checkpoint_sha256=m['model_sha256'],training_seed=json.loads(Path(m['model_manifest']).read_text())['training_seed'],
                  clarabel_version=clarabel.__version__,constants=const,
                  analytic_self_tests=self_test(),started_epoch=time.time(),pid=os.getpid(),
                  scope='Fixed position/association/safety-corrected offload; original sum max(Tloc,Tedge)/Tbase with identical computation-energy budgets. No environment writeback.',
                  numerical_acceptance='Solved; physical resource/energy feasible after tiny cap repair; residual-corrected numerical primal-dual gap <= 1e-6*(1+J). No whole-P1 optimality claim.',
                  timing='Single CPU thread, concurrent-training diagnostic; CF resource allocators vs conic formulation+solve+physical repair. Formal timing remains pending.')
    write(out/'manifest.json',manifest)
    env=UAVEnv(use_cvx_alloc=True,experiment_config=EnvironmentConfig(**m['environment']),seed=999)
    rows=[]
    try:
        with sourcefile.open(encoding='utf-8') as f, (out/'state_results.jsonl').open('x',encoding='utf-8') as target:
            for line in f:
                r=json.loads(line); g=r['methods']['greedy']
                active=np.array(r['active_mask'],dtype=bool); tasks=np.array(r['tasks'])
                x=np.array(g['offload']); assoc=np.array(g['association'],dtype=int)
                cpu=np.array(g['cpu']); bw=np.array(g['bandwidth'])
                pos=np.array(r['candidate_uav_positions']); ue=np.array(r['ue_positions'])
                budget=np.array(r['computation_budget_j']); ids=np.flatnonzero(active)
                base=tasks[ids]*c.S_CYCLES/c.F_UE
                reconstructed=old.cf_delays_from_snapshot(active,tasks,x,assoc,cpu,bw,pos,ue,env._channel_gain,const)
                error=float(np.max(np.abs(reconstructed[ids]-np.array(g['delay'])[ids])))
                assert error<1e-6, 'CF physical delay reconstruction mismatch'
                jcf=float(np.sum(reconstructed[ids]/base))
                off=active & (assoc>=0) & (x>OFFLOAD_EPS)
                fixed=int(np.sum(active & ~off))
                groups=[]; began=time.perf_counter()
                for k in range(c.K):
                    ix=np.flatnonzero(off & (assoc==k)); task=tasks[ix]; ratio=x[ix]
                    cycles=task*c.S_CYCLES; tbase=cycles/c.F_UE
                    logs=np.array([np.log2(1+c.P_TX*env._channel_gain(pos[k],ue[i])/c.SIGMA2) for i in ix])
                    a=ratio*cycles/c.F_UAV/tbase
                    b=ratio*task/(c.BANDWIDTH*logs)/tbase
                    e=c.KAPPA*ratio*cycles*c.F_UAV**2
                    result=solve_group(a,b,1-ratio,e,float(budget[k]),cpu[ix]/c.F_UAV,bw[ix]/c.BANDWIDTH)
                    result['indices']=ix.tolist(); groups.append(result)
                conic_ms=1000*(time.perf_counter()-began)
                valid=all(z['valid'] for z in groups)
                jopt=fixed+sum(z['objective'] for z in groups) if valid else None
                lower=fixed+sum(z['lower_bound'] for z in groups) if valid else None
                if valid:
                    valid=bool(jopt>0 and jopt<=jcf+1e-6*(1+abs(jcf)))
                # Exact same snapshot through an independent optimization algorithm.
                sq=old.solve_reference(active,tasks,x,assoc,cpu,bw,pos,ue,budget,env._channel_gain,const,250)
                sq_j=float(np.sum(sq['delays'][ids]/base)) if sq['success'] else None
                env.active_mask=active.copy(); env.task_list=tasks.copy(); env.loc_ue=ue.copy(); env.loc_uav=pos.copy()
                counts=np.array([np.sum(off & (assoc==k)) for k in range(c.K)])
                cf_ms=[]
                for rep in range(4):
                    start=time.perf_counter()
                    fnew=env._compute_allocation(assoc,counts,x)
                    bnew=env._compute_bandwidth_allocation(assoc,counts,x,pos)
                    elapsed=1000*(time.perf_counter()-start)
                    if rep: cf_ms.append(elapsed)
                np.testing.assert_allclose(fnew,cpu,rtol=1e-10,atol=1e-6)
                np.testing.assert_allclose(bnew,bw,rtol=1e-10,atol=1e-6)
                row=dict(test_seed=r['seed'],slot=r['step'],active_users=len(ids),offloaded_users=int(off.sum()),
                         reference_valid=valid,cf_j=jcf,opt_j=jopt,dual_lower=lower,
                         raw_gap_percent=100*(jcf-jopt)/jopt if valid else None,
                         total_slrr_gain=jcf-jopt if valid else None,
                         mean_active_slrr_gain=(jcf-jopt)/len(ids) if valid else None,
                         slsqp_success=bool(sq['success']),slsqp_j=sq_j,
                         slsqp_minus_conic_j=sq_j-jopt if valid and sq_j is not None else None,
                         cf_delay_max_error_s=error,cf_allocation_ms=float(np.median(cf_ms)),
                         conic_allocation_ms=conic_ms,slsqp_allocation_ms=1000*sq['seconds'],
                         conic_groups=groups)
                target.write(json.dumps(row,allow_nan=False)+'\n');target.flush(); rows.append(row)
                if len(rows)%25==0:
                    write(out/'progress.json',dict(states=len(rows),valid=sum(v['reference_valid'] for v in rows)))
        valid=[r for r in rows if r['reference_valid']]
        numeric=[r for r in rows if r['offloaded_users']>0]
        paired=[r for r in valid if r['slsqp_success']]
        group_results=[g for r in rows for g in r['conic_groups'] if not g.get('trivial',False)]
        summary=dict(status='completed',states=len(rows),valid_states=len(valid),invalid_states=len(rows)-len(valid),
                     numerical_states=len(numeric),numerical_successes=sum(r['reference_valid'] for r in numeric),
                     slsqp_successes=sum(r['slsqp_success'] for r in numeric),
                     paired_solver_states=len(paired),
                     paired_solver_disagreement_states=sum(abs(r['slsqp_minus_conic_j'])>1e-5*(1+abs(r['opt_j'])) for r in paired),
                     relative_gap_percent=stats([r['raw_gap_percent'] for r in valid]),
                     total_slrr_gain=stats([r['total_slrr_gain'] for r in valid]),
                     mean_active_slrr_gain=stats([r['mean_active_slrr_gain'] for r in valid]),
                     numerical_certificate_gap=stats([g.get('relative_certificate_gap') for g in group_results]),
                     energy_excess_j=stats([g.get('energy_excess_j') for g in group_results]),
                     cf_allocation_ms_diagnostic=stats([r['cf_allocation_ms'] for r in valid]),
                     conic_allocation_ms_diagnostic=stats([r['conic_allocation_ms'] for r in valid]),
                     slsqp_minus_conic_j=stats([r['slsqp_minus_conic_j'] for r in paired]),
                     cf_delay_reconstruction_max_error=max((r['cf_delay_max_error_s'] for r in rows),default=0),
                     elapsed_seconds=time.time()-manifest['started_epoch'])
        assert hashlib.sha256(sourcefile.read_bytes()).hexdigest()==input_hash
        write(out/'summary.json',summary)
        manifest.update(status='completed',states=len(rows),ended_epoch=time.time())
        write(out/'manifest.json',manifest)
        print(json.dumps(summary),flush=True)
    except Exception:
        manifest.update(status='failed',error=traceback.format_exc())
        write(out/'manifest.json',manifest)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
