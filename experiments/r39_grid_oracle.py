"""Clairvoyant candidate-graph joint optimization with paired terminal battery.

Physical constants and exogenous inputs are loaded from a frozen-policy window.
Node positions and flight edges are discrete; association, offloading, CPU and
bandwidth are optimized jointly. Original DARS edges are included and checked.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']='-1'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,os.environ.get('R39_SOLVER_DEPS',str(HERE/'r39_solver_deps')))
import numpy as np


def serial(x):
    if hasattr(x,'tolist'): return x.tolist()
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)


def write(p,x):
    p.write_text(json.dumps(x,default=serial,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fly(speed,c):
    if speed<1e-6: return (c['P0']+c['PI'])*c['T_FLY']
    induced=math.sqrt(max(math.sqrt(1+speed**4/(4*c['V0']**4))-speed**2/(2*c['V0']**2),0))
    return c['T_FLY']*(c['P0']*(1+3*speed**2/c['U_TIP']**2)+c['PI']*induced+
        .5*c['D0']*c['RHO']*c['ROTOR_SOLIDITY']*c['ROTOR_AREA']*speed**3)


def charge(pair,case):
    c=case['constants']; powers=np.zeros(2)
    for tower in case['wct_positions']:
        row=[]
        for pos in pair:
            d=math.sqrt(sum((pos[h]-tower[h])**2 for h in range(2))+(c['H']-c['H_WCT'])**2)
            row.append(c['ETA_WCT']/(c['THETA_WCT']*d+c['MU_WCT'])**2 if d<=c['L_C'] else 0.)
        row=np.asarray(row); total=float(row.sum())
        if total>case['environment']['wct_power']: row*=case['environment']['wct_power']/total
        powers+=row
    return np.minimum(powers,c['P_TH'])*c['T_COM']


def efficiency(pos,users,c):
    vals=[]
    for user in users:
        d=math.sqrt(sum((pos[h]-user[h])**2 for h in range(2))+c['H']**2)
        theta=math.degrees(math.asin(c['H']/d))
        los=1/(1+c['A_ENV']*math.exp(-c['B_ENV']*(theta-c['A_ENV'])))
        g=c['ALPHA0']/d**c['ALPHA_PL']/(los*c['MU_LOS']+(1-los)*c['MU_NLOS'])
        vals.append(math.log2(1+c['P_TX']*g/c['SIGMA2']))
    return np.asarray(vals)


def graph(case,spacing):
    c=case['constants']; horizon=case['horizon']; reach=c['V_MAX']*c['T_FLY']
    initial=np.asarray(case['initial_positions']); nodes={}; edges={}; baseline={}
    for k in range(2):
        prev=[initial[k].copy()]; last_dars=0
        for t in range(horizon):
            points=[]
            def add(point):
                p=np.asarray(point,dtype=float)
                if np.any(p<0) or p[0]>c['GROUND_LENGTH'] or p[1]>c['GROUND_WIDTH']: return
                if not any(np.linalg.norm(p-q)<=reach+1e-6 for q in prev): return
                if not any(np.linalg.norm(p-q)<=1e-8 for q in points): points.append(p)
            extent=math.ceil((t+1)*reach/spacing)
            for dx in range(-extent,extent+1):
                for dy in range(-extent,extent+1):
                    if math.hypot(dx,dy)*spacing <= (t+1)*reach+1e-6:
                        add(initial[k]+spacing*np.array([dx,dy]))
            for tower in case['wct_positions']:
                delta=np.asarray(tower)-initial[k]; length=np.linalg.norm(delta)
                if length>0:
                    for step in range(1,t+2): add(initial[k]+delta/length*min(length,step*reach))
                else: add(initial[k])
            # Original DARS waypoints and earlier waypoints (hover candidates).
            for step in range(t+1): add(case['dars'][step]['positions_after'][k])
            current=np.array(points)
            wanted=np.asarray(case['dars'][t]['positions_after'][k])
            dars_node=int(np.argmin(np.linalg.norm(current-wanted,axis=1)))
            assert np.linalg.norm(current[dars_node]-wanted)<1e-7
            arcs=[]
            for i,start in enumerate(prev):
                for j,end in enumerate(current):
                    length=float(np.linalg.norm(end-start))
                    if length<=reach+1e-6:
                        speed=min(length/c['T_FLY'],c['V_MAX'])
                        arcs.append(dict(origin=i,target=j,speed=speed,flight_j=fly(speed,c),dars=False))
            # Preserve executed speed/energy even if a simulator wall reflection
            # makes Euclidean displacement shorter than speed * flight time.
            speed=case['dars'][t]['speeds'][k]
            energy=case['dars'][t]['energy_fly_j'][k]
            assert abs(fly(speed,c)-energy)<1e-6
            assert np.linalg.norm(prev[last_dars]-np.asarray(case['dars'][t]['positions_before'][k]))<1e-7
            arcs.append(dict(origin=last_dars,target=dars_node,speed=speed,flight_j=energy,dars=True))
            nodes[t,k]=current;edges[t,k]=arcs;baseline[t,k]=(dars_node,len(arcs)-1)
            prev=current;last_dars=dars_node
    return nodes,edges,baseline


def solve(case,args):
    import pyscipopt
    from pyscipopt import Model,quicksum
    from r39_grid_validate import validate_solution
    c=case['constants']; T=case['horizon']; scale=1000.
    assert int(c['K'])==2
    baseline_audit=validate_solution(case,case['dars'])
    if not baseline_audit['feasible']: raise ValueError('Original DARS physical replay failed independent validation')
    nodes,edges,dars_graph=graph(case,args.spacing)
    write(args.output/'graph.json',dict(spacing_m=args.spacing,
        nodes={f'{t},{k}':v for (t,k),v in nodes.items()},
        edges={f'{t},{k}':v for (t,k),v in edges.items()},
        dars_indices={f'{t},{k}':v for (t,k),v in dars_graph.items()}))
    m=Model('grid_assisted_terminal_paired_joint_reference')
    m.setRealParam('limits/time',args.time_limit)
    m.setRealParam('limits/gap',args.gap)
    m.setRealParam('limits/memory',4096.)
    m.setRealParam('numerics/feastol',1e-8)
    m.setIntParam('parallel/maxnthreads',1)
    m.setIntParam('display/verblevel',3)
    m.setIntParam('display/freq',1000)
    # Dense position-pair tables make exhaustive presolve probing expensive.
    # Disabling this optional reduction leaves all constraints and bounds valid.
    m.setIntParam('propagating/probing/maxprerounds',0)
    y={}; edgevar={}; battery={}; joint={}; received={}; flyexpr={}; link={}; z={}; se={}
    build_started=time.perf_counter()
    cap=case['environment']['battery_capacity']/scale
    analytic_j_lower=0.
    for k in range(2):
        battery[0,k]=m.addVar(f'B_0_{k}',lb=case['initial_battery'][k]/scale,ub=case['initial_battery'][k]/scale)
    for t in range(T):
        for k in range(2):
            n=len(nodes[t,k]); arcs=edges[t,k]
            y[t,k]=[m.addVar(f'y_{t}_{k}_{j}',vtype='B') for j in range(n)]
            edgevar[t,k]=[m.addVar(f'edge_{t}_{k}_{e}',vtype='B') for e in range(len(arcs))]
            m.addCons(quicksum(y[t,k])==1)
            for j in range(n):
                m.addCons(quicksum(edgevar[t,k][e] for e,a in enumerate(arcs) if a['target']==j)==y[t,k][j])
            if t:
                for j in range(len(nodes[t-1,k])):
                    m.addCons(quicksum(edgevar[t,k][e] for e,a in enumerate(arcs) if a['origin']==j)==y[t-1,k][j])
            else: m.addCons(quicksum(edgevar[t,k])==1)
            flyexpr[t,k]=quicksum(v*a['flight_j']/scale for v,a in zip(edgevar[t,k],arcs))
            battery[t+1,k]=m.addVar(f'B_{t+1}_{k}',lb=0,ub=cap)
            m.addCons(flyexpr[t,k]<=battery[t,k])
        n0,n1=len(nodes[t,0]),len(nodes[t,1])
        # Marginals of binary single-node choices make this nonnegative joint
        # table exactly one-hot; extra joint-position binaries are unnecessary.
        harvest=np.empty((n0,n1,2))
        for i in range(n0):
            for j in range(n1):
                joint[t,i,j]=m.addVar(f'pair_{t}_{i}_{j}',lb=0,ub=1)
                harvest[i,j]=charge([nodes[t,0][i],nodes[t,1][j]],case)/scale
        for i in range(n0): m.addCons(quicksum(joint[t,i,j] for j in range(n1))==y[t,0][i])
        for j in range(n1): m.addCons(quicksum(joint[t,i,j] for i in range(n0))==y[t,1][j])
        for k in range(2): received[t,k]=quicksum(joint[t,i,j]*harvest[i,j,k] for i in range(n0) for j in range(n1))
        row=case['stages'][t]; ids=row['active_indices']; users=np.asarray(row['user_positions'])[ids]
        for k in range(2): se[t,k]=np.array([efficiency(pos,users,c) for pos in nodes[t,k]]).T
        for i,uid in enumerate(ids):
            zmax=max(1.,float(case['dars'][t]['time_normalized'][i]))+1e-5
            z[t,i]=m.addVar(f'z_{t}_{uid}',lb=0,ub=zmax)
            for k in range(2):
                eligible=[j for j,pos in enumerate(nodes[t,k]) if np.linalg.norm(pos-users[i])<=c['R_COV']+1e-7]
                a=m.addVar(f'a_{t}_{uid}_{k}',vtype='B',ub=int(bool(eligible)))
                x=m.addVar(f'x_{t}_{uid}_{k}',lb=0,ub=int(bool(eligible)))
                f=m.addVar(f'f_{t}_{uid}_{k}',lb=0,ub=int(bool(eligible)))
                b=m.addVar(f'b_{t}_{uid}_{k}',lb=0,ub=int(bool(eligible)))
                ut=m.addVar(f'ut_{t}_{uid}_{k}',lb=0,ub=zmax if eligible else 0)
                uc=m.addVar(f'uc_{t}_{uid}_{k}',lb=0,ub=zmax if eligible else 0)
                for var in (x,f,b): m.addCons(var<=a)
                m.addCons(ut<=zmax*a);m.addCons(uc<=zmax*a)
                m.addCons(a<=quicksum(y[t,k][j] for j in eligible))
                # Disaggregate x by selected position, replacing x / SE(q)
                # with an exact linear sum over precomputed coefficients.
                w={j:m.addVar(f'w_{t}_{uid}_{k}_{j}',lb=0,ub=1) for j in eligible}
                for j,v in w.items(): m.addCons(v<=y[t,k][j])
                m.addCons(quicksum(w.values())==x)
                m.addCons(ut*b>=c['F_UE']/(c['S_CYCLES']*c['BANDWIDTH'])*quicksum(v/se[t,k][i,j] for j,v in w.items()))
                m.addCons(uc*f>=c['F_UE']/c['F_UAV']*x)
                entry=dict(a=a,x=x,f=f,b=b,ut=ut,uc=uc,w=w)
                if eligible:
                    # Valid conic performance bound, proved in theory_note.tex.
                    # It strengthens the continuous relaxation and preserves
                    # every original feasible control (including negative SLRR).
                    bp={j:m.addVar(f'bp_{t}_{uid}_{k}_{j}',lb=0,ub=1) for j in eligible}
                    for j,v in bp.items(): m.addCons(v<=y[t,k][j])
                    m.addCons(quicksum(bp.values())==b)
                    alpha=c['F_UE']/(c['S_CYCLES']*c['BANDWIDTH']);beta=c['F_UE']/c['F_UAV']
                    h1=m.addVar(f'h1_{t}_{uid}_{k}',lb=0,ub=max(se[t,k][i])/alpha)
                    h2=m.addVar(f'h2_{t}_{uid}_{k}',lb=0,ub=1/beta)
                    m.addCons(h1==quicksum(v*se[t,k][i,j]/alpha for j,v in bp.items()))
                    m.addCons(h2==f/beta)
                    u=m.addVar(f'u_{t}_{uid}_{k}',lb=0,ub=1)
                    r=m.addVar(f'r_{t}_{uid}_{k}',lb=0,ub=1);m.addCons(r<=a)
                    v1=m.addVar(f'v1_{t}_{uid}_{k}',lb=0,ub=1)
                    w1=m.addVar(f'w1_{t}_{uid}_{k}',lb=0,ub=max(se[t,k][i])/alpha)
                    v2=m.addVar(f'v2_{t}_{uid}_{k}',lb=0,ub=1)
                    w2=m.addVar(f'w2_{t}_{uid}_{k}',lb=0,ub=1/beta)
                    m.addCons(v1==1-u);m.addCons(w1==h1-u)
                    m.addCons(v2==u-r);m.addCons(w2==h2-r)
                    m.addCons(u*u<=v1*w1);m.addCons(r*r<=v2*w2)
                    entry.update(bp=bp,h1=h1,h2=h2,u=u,r=r,v1=v1,w1=w1,v2=v2,w2=w2)
                link[t,i,k]=entry
            m.addCons(link[t,i,0]['a']+link[t,i,1]['a']<=1)
            m.addCons(z[t,i]>=1-link[t,i,0]['x']-link[t,i,1]['x'])
            m.addCons(z[t,i]>=quicksum(link[t,i,k]['ut']+link[t,i,k]['uc'] for k in range(2)))
            m.addCons(quicksum(link[t,i,k]['r'] for k in range(2) if 'r' in link[t,i,k])>=1-z[t,i])
            available=[se[t,k][i,j] for k in range(2) for j in link[t,i,k]['w']]
            if available:
                ratio=c['F_UE']/c['F_UAV']+c['F_UE']/(c['S_CYCLES']*c['BANDWIDTH']*max(available))
                lower=ratio/(1+ratio)
            else: lower=1.
            # Relaxation: every user independently receives an entire UAV's
            # CPU/BW and its best candidate channel. It is a valid bound even
            # if the solver reaches a time limit before producing a dual bound.
            m.addCons(z[t,i]>=lower)
            analytic_j_lower+=lower
        for k in range(2):
            m.addCons(quicksum(link[t,i,k]['f'] for i in range(len(ids)))<=1)
            m.addCons(quicksum(link[t,i,k]['b'] for i in range(len(ids)))<=1)
            m.addCons(quicksum(link[t,i,k]['a'] for i in range(len(ids)))<=c['N_MAX'])
            comp=quicksum(c['KAPPA']*row['tasks'][uid]*c['S_CYCLES']*c['F_UAV']**2/scale*
                link[t,i,k]['x']*link[t,i,k]['f']**2 for i,uid in enumerate(ids))
            hover=(c['P0']+c['PI'])*c['T_COM']/scale
            m.addCons(hover+comp<=battery[t,k]-flyexpr[t,k]+received[t,k])
            m.addCons(battery[t+1,k]<=battery[t,k]-flyexpr[t,k]-hover-comp+received[t,k])
    for k in range(2): m.addCons(battery[T,k]>=case['terminal_battery_dars'][k]/scale)
    objective=quicksum(z.values());m.setObjective(objective,'minimize')
    # Insert the exact executed baseline, including its true capped batteries.
    warm=m.createSol()
    for k in range(2): m.setSolVal(warm,battery[0,k],case['initial_battery'][k]/scale)
    for t in range(T):
        d=case['dars'][t]
        for k in range(2):
            node,arc=dars_graph[t,k]
            for j,v in enumerate(y[t,k]): m.setSolVal(warm,v,float(j==node))
            for j,v in enumerate(edgevar[t,k]): m.setSolVal(warm,v,float(j==arc))
            m.setSolVal(warm,battery[t+1,k],d['battery_after'][k]/scale)
        for i in range(len(nodes[t,0])):
            for j in range(len(nodes[t,1])): m.setSolVal(warm,joint[t,i,j],float(i==dars_graph[t,0][0] and j==dars_graph[t,1][0]))
        for i,uid in enumerate(case['stages'][t]['active_indices']):
            m.setSolVal(warm,z[t,i],d['time_normalized'][i])
            for k in range(2):
                e=link[t,i,k];node=dars_graph[t,k][0]
                for var,field in [('a','association'),('x','x_ik'),('f','fraction_cpu'),('b','bwshare')]: m.setSolVal(warm,e[var],d[field][i][k])
                x=d['x_ik'][i][k];b=d['bwshare'][i][k];f=d['fraction_cpu'][i][k]
                for j,v in e['w'].items(): m.setSolVal(warm,v,x if j==node else 0.)
                ut=c['F_UE']/(c['S_CYCLES']*c['BANDWIDTH'])*x/(b*se[t,k][i,node]) if x>0 else 0.
                uc=c['F_UE']/c['F_UAV']*x/f if x>0 else 0.
                m.setSolVal(warm,e['ut'],ut);m.setSolVal(warm,e['uc'],uc)
                if 'r' in e:
                    alpha=c['F_UE']/(c['S_CYCLES']*c['BANDWIDTH']);beta=c['F_UE']/c['F_UAV']
                    h1=b*se[t,k][i,node]/alpha;h2=f/beta;u=h1/(1+h1)
                    r=max(0.,1-d['time_normalized'][i]) if d['association'][i][k]>.5 else 0.
                    for j,v in e['bp'].items(): m.setSolVal(warm,v,b if j==node else 0.)
                    for name,value in [('h1',h1),('h2',h2),('u',u),('r',r),('v1',1-u),('w1',h1-u),('v2',u-r),('w2',h2-r)]:
                        m.setSolVal(warm,e[name],value)
    warm_valid=bool(m.checkSol(warm,printreason=True,completely=True))
    if not warm_valid: raise ValueError('DARS candidate is not feasible in the reference formulation')
    m.addSol(warm,free=True)
    counts=dict(variables=m.getNVars(),constraints=m.getNConss(),
        nodes_per_slot=[[len(nodes[t,k]) for k in range(2)] for t in range(T)],
        edges_per_slot=[[len(edges[t,k]) for k in range(2)] for t in range(T)],
        active_users_per_slot=[len(s['active_indices']) for s in case['stages']])
    write(args.output/'status.json',dict(status='solving',pid=os.getpid(),warm_start_verified=warm_valid,
         counts=counts,build_seconds=time.perf_counter()-build_started,started_epoch=time.time()))
    start=time.perf_counter();m.optimize();wall=time.perf_counter()-start
    sol=m.getBestSol()
    if sol is None: raise RuntimeError('Verified baseline disappeared from solver')
    val=lambda v:float(m.getSolVal(sol,v))
    rows=[]; physical_battery=np.asarray(case['initial_battery'],dtype=float); prior=np.asarray(case['initial_positions'])
    for t in range(T):
        chosen=[int(np.argmax([val(v) for v in y[t,k]])) for k in range(2)]
        arcs=[edges[t,k][int(np.argmax([val(v) for v in edgevar[t,k]]))] for k in range(2)]
        pos=np.array([nodes[t,k][chosen[k]] for k in range(2)])
        ids=case['stages'][t]['active_indices'];A=len(ids)
        matrices={field:[[val(link[t,i,k][var]) for k in range(2)] for i in range(A)]
                  for var,field in [('a','association'),('x','x_ik'),('f','fraction_cpu'),('b','bwshare')]}
        ef=np.array([a['flight_j'] for a in arcs]);eh=np.full(2,(c['P0']+c['PI'])*c['T_COM'])
        comp=np.sum(np.asarray(matrices['x_ik'])*np.asarray(case['stages'][t]['tasks'])[ids,None]*c['S_CYCLES']*
                    (np.asarray(matrices['fraction_cpu'])*c['F_UAV'])**2*c['KAPPA'],axis=0)
        incoming=charge(pos,case)
        after=np.minimum(case['environment']['battery_capacity'],physical_battery-ef-eh-comp+incoming)
        rows.append(dict(**case['stages'][t],**matrices,positions_before=prior,positions_after=pos,
            speeds=[a['speed'] for a in arcs],battery_before=physical_battery.copy(),battery_after=after.copy(),
            auxiliary_battery_before_j=[val(battery[t,k])*scale for k in range(2)],
            auxiliary_battery_after_j=[val(battery[t+1,k])*scale for k in range(2)],
            energy_fly_j=ef,energy_hover_j=eh,energy_comp_j=comp,energy_in_j=incoming,
            time_normalized=[val(z[t,i]) for i in range(A)],selected_nodes=chosen))
        physical_battery=after;prior=pos
    raw_audit=validate_solution(case,rows)
    # The epigraph is a reporting auxiliary, not an executed control. Rebuild it
    # from physical max latency to turn a numerically understated z into a true
    # feasible epigraph; retain both raw values and the initial audit.
    epigraph_adjustment=0.;epigraph_bound_violation=0.
    for t,(row,check) in enumerate(zip(rows,raw_audit['stages'])):
        raw=np.asarray(row['time_normalized']);physical=np.asarray(check['normalized_delay'])
        corrected=np.maximum(raw,physical)
        upper_z=np.maximum(1.,np.asarray(case['dars'][t]['time_normalized']))+1e-5
        epigraph_bound_violation=max(epigraph_bound_violation,float(np.max(corrected-upper_z,initial=0.)))
        row['solver_time_normalized_raw']=raw.tolist()
        row['time_normalized']=corrected.tolist()
        epigraph_adjustment=max(epigraph_adjustment,float(np.max(corrected-raw,initial=0.)))
    audit=validate_solution(case,rows)
    n=sum(len(s['active_indices']) for s in case['stages']); primal=float(m.getPrimalbound())
    raw_dual=float(m.getDualbound());dual=max(analytic_j_lower,raw_dual)
    ref=audit['total_slrr'];dars=baseline_audit['total_slrr'];upper=n-dual
    verified_j=audit['objective_j']
    verified_j_gap=max(0.,verified_j-dual)/max(min(abs(verified_j),abs(dual)),1e-12)
    result=dict(status=str(m.getStatus()),case_id=case['case_id'],network_users=case['network_users'],
        training_seed=case['training_seed'],test_seed=case['test_seed'],start_slot=case['start_slot'],
        horizon=T,spacing_m=args.spacing,low_battery=case['low_battery'],
        solver_version=f'SCIP {m.getMajorVersion()}.{m.getMinorVersion()}.{m.getTechVersion()}',
        pyscipopt_version=pyscipopt.__version__,counts=counts,warm_start_verified=warm_valid,
        solve_seconds=wall,solver_gap=m.getGap(),J_primal=primal,J_dual=dual,
        SCIP_J_dual_raw=raw_dual,analytic_J_lower_bound=analytic_j_lower,
        lower_bound_source='SCIP' if raw_dual>=analytic_j_lower else 'independent-user analytic relaxation',
        J_independently_verified=verified_j,J_epigraph_checked=sum(sum(row['time_normalized']) for row in rows),
        verified_J_gap=verified_j_gap,
        oracle_total_slrr=ref,oracle_slrr_upper_bound=upper,dars_total_slrr=dars,
        slrr_gain=ref-dars,slrr_certificate_width=upper-ref,
        relative_slrr_certificate_width=(upper-ref)/max(abs(upper),1e-12),
        terminal_battery_oracle=physical_battery,terminal_battery_dars=case['terminal_battery_dars'],
        independent_audit=audit,raw_solver_audit=raw_audit,
        max_epigraph_adjustment=epigraph_adjustment,max_epigraph_bound_violation=epigraph_bound_violation,
        baseline_audit=baseline_audit,solution=rows,
        scope='Finite candidate graph (25m local lattice, reachable WCT-direction points, original DARS trajectory); full actual load; clairvoyant; continuous offload/CPU/bandwidth and exact association choices; original phase energy and per-UAV DARS terminal battery floors.')
    result['valid_reference']=bool(audit['feasible'] and epigraph_bound_violation<=1e-7 and ref+1e-6>=dars and upper+1e-5>=ref)
    result['certified_within_requested_gap']=bool(result['valid_reference'] and verified_j_gap<=args.gap)
    write(args.output/'result.json',result)
    write(args.output/'status.json',dict(status='completed',solver_status=result['status'],valid_reference=result['valid_reference'],ended_epoch=time.time()))
    print(json.dumps({k:v for k,v in result.items() if k not in ('solution','independent_audit','raw_solver_audit','baseline_audit')},default=serial),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--case-id')
    p.add_argument('--case-index',type=int,default=0)
    p.add_argument('--horizon',type=int,choices=(1,2,3,5),default=3)
    p.add_argument('--spacing',type=float,default=25.)
    p.add_argument('--time-limit',type=float,default=120.)
    p.add_argument('--gap',type=float,default=1e-3)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    cases=[json.loads(line) for line in args.cases.read_text(encoding='utf-8').splitlines()]
    case=copy.deepcopy(next(c for c in cases if c['case_id']==args.case_id) if args.case_id else cases[args.case_index])
    assert args.horizon<=case['horizon']
    case['source_window_id']=case['case_id'];case['horizon']=args.horizon
    case['case_id']=case['case_id'].rsplit('_h',1)[0]+f'_h{args.horizon}'
    case['stages']=case['stages'][:args.horizon];case['dars']=case['dars'][:args.horizon]
    case['terminal_battery_dars']=case['dars'][-1]['battery_after']
    case['dars_total_slrr']=sum(r['total_slrr'] for r in case['dars'])
    write(args.output/'instance.json',case)
    write(args.output/'protocol.json',dict(arguments=vars(args),source_cases_sha256=sha(args.cases),
        source_hashes={name:sha(HERE/name) for name in ('r39_grid_oracle.py','r39_grid_validate.py')},
        objective='maximize actual total SLRR over fixed exogenous window',
        battery_formulation='Auxiliary inventory under-estimates physical capped battery; monotonicity makes projection onto physical controls exact. Actual reported batteries are recomputed.',
        started_epoch=time.time()))
    solve(case,args)


if __name__=='__main__': main()
