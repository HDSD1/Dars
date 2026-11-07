"""Short-horizon continuous joint MINLP benchmark, independent of training.

All motion endpoints, associations, offloading fractions, CPU and bandwidth
are optimized. Energy constraints are removed ONLY after a worst-case bound
proves them redundant for the entire selected horizon, even without charging.
The result is a short-horizon P1 special case, not a 50-slot P1 optimum.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/r39_solver_deps'))


def serial(x):
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def write(path, data):
    path.write_text(json.dumps(data, default=serial, indent=2,
                               ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    import numpy as np
    import torch
    base = ROOT/'results/original_main_seed42_gpu_retry_20261029_234000'
    queue = ROOT/'results/formal_all_five_seeds43_44_restart_20261031/source_snapshot'
    source = base/'source_snapshot' if args.training_seed == 42 else queue
    modeldir = base if args.training_seed == 42 else queue/f'results/formal_all_five_seeds43_44_session2_20261031_seed{args.training_seed}'
    checkpoint = modeldir/'per_sac_cvx.pth'
    manifest = json.loads((modeldir/'per_sac_cvx.manifest.json').read_text(encoding='utf-8'))
    assert args.seed not in set(manifest['training_episode_seeds']) | set(manifest['validation_seeds'])
    sys.path.insert(0, str(source))
    import config as c
    from DARS import PERSACAgent
    from UAV_env import UAVEnv
    from experiment_config import EnvironmentConfig
    from evaluation import frozen_policy, policy_input
    from state_normalization import StateNormalization
    torch.set_num_threads(1)
    cfg = EnvironmentConfig.from_json(ROOT/'experiments/train_diverse.json')
    env = UAVEnv(use_cvx_alloc=True, experiment_config=cfg, seed=args.seed)
    env.reset(seed=args.seed)
    agent = PERSACAgent()
    agent.load(str(checkpoint))
    normalizer = StateNormalization(cfg.battery_capacity)
    rng = np.random.default_rng(args.seed+19871)
    # Choose all test users/loads before observing any solver result.
    selected = [sorted(rng.choice(c.M, args.active, replace=False).tolist()) for _ in range(args.horizon)]
    loads = [rng.integers(c.TASK_MIN, c.TASK_MAX, args.active) for _ in range(args.horizon)]
    initial_positions = env.loc_uav.copy()
    stages, dars = [], []
    started = time.perf_counter()
    with frozen_policy(agent):
        for t in range(args.horizon):
            ids = selected[t]
            env.active_mask[:] = False
            env.active_mask[ids] = True
            env.task_list[:] = 0
            env.task_list[ids] = loads[t]
            before = env.loc_uav.copy()
            battery = env.e_battery_uav.copy()
            ue = env.loc_ue.copy()
            tasks = env.task_list.copy()
            state = env._get_obs()
            action = agent.select_action(policy_input(agent, normalizer, state), noise_std=0.0)
            _, reward, _, info = env.step(action)
            assert info['action_executed']
            matrices = {name: np.zeros((args.active, c.K)) for name in ('association','x_ik','fraction_cpu','bwshare')}
            for j, i in enumerate(ids):
                k = int(info['association'][i])
                if k >= 0:
                    matrices['association'][j,k] = 1
                    matrices['x_ik'][j,k] = info['offload_ratios'][i]
                    matrices['fraction_cpu'][j,k] = info['cpu_allocation'][i]/c.F_UAV
                    matrices['bwshare'][j,k] = info['bandwidth_allocation'][i]/c.BANDWIDTH
            z = 1-np.asarray(env.rho_log[-1])[ids]
            stage = dict(user_positions=ue, tasks=tasks, active_indices=ids)
            stages.append(stage)
            dars.append(dict(**stage, **matrices, positions_before=before,
                             positions_after=env.loc_uav.copy(), battery=battery,
                             time_normalized=z, training_reward=reward))
    constants = {n:getattr(c,n) for n in dir(c) if n.isupper()
                 and isinstance(getattr(c,n),(int,float))}
    return dict(test_seed=args.seed, training_seed=args.training_seed,
                horizon=args.horizon, network_users=c.M, active_per_slot=args.active,
                initial_positions=initial_positions, initial_battery=np.full(c.K,cfg.initial_energy),
                constants=constants, stages=stages, dars=dars,
                dars_total_slrr=sum(float(np.sum(1-r['time_normalized'])) for r in dars),
                source=str(source), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                source_sha256={n:sha(source/n) for n in ['config.py','UAV_env.py','DARS.py']},
                dars_evaluation_seconds=time.perf_counter()-started)


def build_and_solve(case, args):
    import numpy as np
    import pyscipopt
    from pyscipopt import Model, quicksum, sin, exp, log
    from r39_validate import validate_solution, energy_upper_bound
    c = case['constants']
    proof = energy_upper_bound(c)
    T, A, K = case['horizon'], case['active_per_slot'], int(c['K'])
    # No probability or charging approximation is used: all modeled energy
    # constraints are redundant on this chosen high-battery short horizon.
    assert min(case['initial_battery']) > T*proof['total_per_slot_j']
    H, distance = c['H'], c['V_MAX']*c['T_FLY']
    q0 = np.array(case['initial_positions'])/H
    # Keeps the simulator's wall-reflection rule irrelevant for every feasible path.
    assert np.all(q0*H-T*distance > 0)
    assert np.all(q0*H+T*distance < min(c['GROUND_LENGTH'],c['GROUND_WIDTH']))
    # Keep all artifact IO in Python: native SCIP file IO does not support this
    # Windows workspace reliably. Solver output is captured by the launching tool.
    m = Model('continuous_short_horizon_joint_P1')
    m.setRealParam('limits/time', args.time_limit)
    m.setRealParam('limits/gap', args.gap)
    m.setRealParam('numerics/feastol', 1e-8)
    m.setIntParam('parallel/maxnthreads', 1)
    m.setIntParam('display/verblevel', 3)
    m.setRealParam('limits/memory', 2048.)
    q, data, zs = {}, {}, []
    for t in range(T):
        for k in range(K):
            for d in range(2):
                span = (t+1)*distance/H
                q[t,k,d] = m.addVar(f'q_{t}_{k}_{d}', lb=q0[k,d]-span, ub=q0[k,d]+span)
            prior = [q[t-1,k,d] if t else q0[k,d] for d in range(2)]
            m.addCons(quicksum((q[t,k,d]-prior[d])**2 for d in range(2)) <= (distance/H)**2)
        row = case['stages'][t]
        ue = np.array(row['user_positions'])[row['active_indices']]/H
        for i in range(A):
            z = m.addVar(f'z_{t}_{i}', lb=0, ub=1)
            zs.append(z)
            edges = []
            for k in range(K):
                span=(t+1)*distance/H
                lo=q0[k]-span; hi=q0[k]+span
                nearest=np.maximum(np.maximum(lo-ue[i],ue[i]-hi),0)
                farthest=np.maximum(abs(lo-ue[i]),abs(hi-ue[i]))
                dlo=float(np.sqrt(1+np.sum(nearest**2)))
                dhi=float(np.sqrt(1+np.sum(farthest**2)))
                eligible=bool(np.sqrt(np.sum(nearest**2)) <= c['R_COV']/H)
                a=m.addVar(f'a_{t}_{i}_{k}',vtype='B',ub=int(eligible))
                x=m.addVar(f'x_{t}_{i}_{k}',lb=0,ub=int(eligible))
                f=m.addVar(f'f_{t}_{i}_{k}',lb=0,ub=int(eligible))
                b=m.addVar(f'b_{t}_{i}_{k}',lb=0,ub=int(eligible))
                ut=m.addVar(f'ut_{t}_{i}_{k}',lb=0,ub=int(eligible))
                uc=m.addVar(f'uc_{t}_{i}_{k}',lb=0,ub=int(eligible))
                ed=dict(a=a,x=x,f=f,b=b,ut=ut,uc=uc,z=z)
                for v in (x,f,b,ut,uc): m.addCons(v<=a)
                if eligible:
                    d=m.addVar(f'd_{t}_{i}_{k}',lb=dlo,ub=dhi)
                    theta=m.addVar(f'theta_{t}_{i}_{k}',lb=math.asin(1/dhi),ub=math.asin(min(1.,1/dlo)))
                    def se_of_distance(v):
                        p=1/(1+c['A_ENV']*math.exp(-c['B_ENV']*(math.degrees(math.asin(1/v))-c['A_ENV'])))
                        loss=p*c['MU_LOS']+(1-p)*c['MU_NLOS']
                        return math.log2(1+c['P_TX']*c['ALPHA0']/(c['SIGMA2']*(H*v)**c['ALPHA_PL']*loss))
                    se=m.addVar(f'se_{t}_{i}_{k}',lb=se_of_distance(dhi),ub=se_of_distance(dlo))
                    m.addCons(d*d==1+quicksum((q[t,k,h]-ue[i,h])**2 for h in range(2)))
                    m.addCons(d*sin(theta)==1)
                    p=1/(1+c['A_ENV']*exp(-c['B_ENV']*(theta*180/math.pi-c['A_ENV'])))
                    loss=m.addVar(f'loss_{t}_{i}_{k}',lb=c['MU_LOS'],ub=c['MU_NLOS'])
                    m.addCons(loss==c['MU_NLOS']+(c['MU_LOS']-c['MU_NLOS'])*p)
                    snr_factor=c['P_TX']*c['ALPHA0']/(c['SIGMA2']*H**c['ALPHA_PL'])
                    m.addCons(se==log(1+snr_factor/(d**c['ALPHA_PL']*loss))/math.log(2))
                    m.addCons(d*d<=1+(c['R_COV']/H)**2+(dhi*dhi-1)*(1-a))
                    m.addCons(ut*b*se >= c['F_UE']/(c['S_CYCLES']*c['BANDWIDTH'])*x)
                    m.addCons(uc*f >= c['F_UE']/c['F_UAV']*x)
                    ed.update(d=d,theta=theta,se=se,loss=loss)
                edges.append(ed)
                data[t,i,k]=ed
            m.addCons(quicksum(e['a'] for e in edges)<=1)
            m.addCons(z>=1-quicksum(e['x'] for e in edges))
            m.addCons(z>=quicksum(e['ut']+e['uc'] for e in edges))
        for k in range(K):
            m.addCons(quicksum(data[t,i,k]['f'] for i in range(A))<=1)
            m.addCons(quicksum(data[t,i,k]['b'] for i in range(A))<=1)
            m.addCons(quicksum(data[t,i,k]['a'] for i in range(A))<=c['N_MAX'])
    m.setObjective(quicksum(zs),'minimize')
    # Seed an analytic feasible all-local point, then the frozen policy's point.
    warm = []
    for mode in ('local','dars'):
        sol=m.createSol()
        for t in range(T):
            rows=case['dars'][t]
            pos=np.array(rows['positions_after'])/H if mode=='dars' else q0
            ue=np.array(rows['user_positions'])[rows['active_indices']]/H
            for k in range(K):
                for h in range(2): m.setSolVal(sol,q[t,k,h],pos[k,h])
            for i in range(A):
                m.setSolVal(sol,data[t,i,0]['z'],float(rows['time_normalized'][i]) if mode=='dars' else 1.)
                for k in range(K):
                    e=data[t,i,k]
                    for v,field in [('a','association'),('x','x_ik'),('f','fraction_cpu'),('b','bwshare')]:
                        m.setSolVal(sol,e[v],float(rows[field][i][k]) if mode=='dars' else 0.)
                    d=float(np.sqrt(1+np.sum((pos[k]-ue[i])**2)))
                    theta=math.asin(1/d)
                    p=1/(1+c['A_ENV']*math.exp(-c['B_ENV']*(theta*180/math.pi-c['A_ENV'])))
                    se=math.log2(1+c['P_TX']*c['ALPHA0']/((H*d)**c['ALPHA_PL']*c['SIGMA2']*(p*c['MU_LOS']+(1-p)*c['MU_NLOS'])))
                    if 'd' in e:
                        for name,value in [('d',d),('theta',theta),('se',se),('loss',p*c['MU_LOS']+(1-p)*c['MU_NLOS'])]: m.setSolVal(sol,e[name],value)
                    x=float(rows['x_ik'][i][k]) if mode=='dars' else 0.
                    b=float(rows['bwshare'][i][k]) if mode=='dars' else 0.
                    f=float(rows['fraction_cpu'][i][k]) if mode=='dars' else 0.
                    ut=c['F_UE']/(c['S_CYCLES']*c['BANDWIDTH'])*x/(b*se) if x>0 else 0.
                    uc=c['F_UE']/c['F_UAV']*x/f if x>0 else 0.
                    m.setSolVal(sol,e['ut'],ut);m.setSolVal(sol,e['uc'],uc)
        accepted=m.addSol(sol,free=True)
        warm.append(dict(mode=mode,stored=bool(accepted)))
    solve_started=time.perf_counter()
    m.optimize()
    solve_wall_seconds=time.perf_counter()-solve_started
    solution=m.getBestSol()
    if solution is None:
        return dict(status=str(m.getStatus()),solving_seconds=m.getSolvingTime(),has_solution=False)
    value=lambda v:float(m.getSolVal(solution,v))
    best=[]; validations=[]; battery=np.array(case['initial_battery'],dtype=float)
    prior=np.array(case['initial_positions'])
    for t in range(T):
        pos=np.array([[value(q[t,k,h])*H for h in range(2)] for k in range(K)])
        matrices={field:[[value(data[t,i,k][v]) for k in range(K)] for i in range(A)]
                  for v,field in [('a','association'),('x','x_ik'),('f','fraction_cpu'),('b','bwshare')]}
        row=dict(**case['stages'][t],**matrices,positions_before=prior,positions_after=pos,
                 battery=battery.copy(),time_normalized=[value(data[t,i,0]['z']) for i in range(A)])
        audit=validate_solution(row,c)
        validations.append(audit);best.append(row)
        battery-=np.array(audit['energy_fly_j'])+np.array(audit['energy_hover_j'])+np.array(audit['energy_comp_j'])
        prior=pos
    dars_audits=[validate_solution(r,c) for r in case['dars']]
    primal=float(m.getPrimalbound());dual=float(m.getDualbound())
    verified=sum(v['total_slrr'] for v in validations)
    return dict(status=str(m.getStatus()),solver='SCIP',pyscipopt_version=pyscipopt.__version__,
                scip_version=f'{m.getMajorVersion()}.{m.getMinorVersion()}.{m.getTechVersion()}',
                solving_seconds=m.getSolvingTime(),solve_wall_seconds=solve_wall_seconds,
                nodes=m.getNNodes(),has_solution=True,
                J_feasible=primal,J_lower_bound=dual,solver_relative_gap=m.getGap(),
                total_slrr_feasible=A*T-primal,total_slrr_upper_bound=A*T-dual,
                verified_total_slrr=verified,slrr_absolute_certificate_width=primal-dual,
                independently_feasible=all(v['feasible'] for v in validations),
                dars_total_slrr=case['dars_total_slrr'],
                dars_to_upper_bound_percent=100*case['dars_total_slrr']/(A*T-dual) if A*T-dual>0 else None,
                energy_redundancy_proof=proof,warm_starts=warm,solution=best,
                independent_validation=validations,dars_validation=dars_audits,
                scope='Continuous joint motion/association/offloading/CPU/bandwidth, short horizon; all future exogenous user/task inputs known to optimizer. High battery makes energy constraints redundant by rigorous worst-case bound. Not a long-horizon or low-energy oracle.',
                population='30 input users retained for frozen policy; only stated active_per_slot users have tasks. No reduced-user retraining.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=960000)
    p.add_argument('--training-seed',type=int,choices=[42,43,44],default=42)
    p.add_argument('--active',type=int,default=3)
    p.add_argument('--horizon',type=int,default=1)
    p.add_argument('--time-limit',type=float,default=120)
    p.add_argument('--gap',type=float,default=1e-4)
    p.add_argument('--instance',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    args.output=args.output.resolve()
    if args.prepare_only:
        write(args.output/'instance.json',prepare(args))
        return
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'protocol.json',dict(arguments=vars(args),started_epoch=time.time(),
          script_sha256=sha(Path(__file__)),
          selection='Consecutive independent seeds; uniform random active-user IDs and task loads drawn before solve; no result-based state selection.',
          metric='Actual sum SLRR, not training reward.',
          purpose='Solver validation pilot. Formal paired benchmark design follows numerical validation.'))
    if args.instance:
        case=json.loads(args.instance.read_text(encoding='utf-8'))
        write(args.output/'instance.json',case)
    else:
        # Torch and SCIP ship different Intel OpenMP libraries on Windows.
        # Evaluate the policy in a separate process, never load both runtimes.
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--prepare-only',
            '--output',str(args.output),'--seed',str(args.seed),'--training-seed',str(args.training_seed),
            '--active',str(args.active),'--horizon',str(args.horizon)],check=True)
        case=json.loads((args.output/'instance.json').read_text(encoding='utf-8'))
    result=build_and_solve(case,args)
    write(args.output/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['solution','independent_validation','dars_validation']},default=serial),flush=True)


if __name__=='__main__':
    main()
