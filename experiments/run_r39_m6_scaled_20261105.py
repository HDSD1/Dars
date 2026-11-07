"""Isolated, author-authorized M6 resource-scaled DARS and long-window oracle pilot.

No automatic retries. Existing experiments and learned parameters are read-only.
"""
import argparse
import copy
import ctypes
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'Dars-final')
OUT = ROOT/'results/reviewer_r39_m6_scaled_seed42_20261105'
SOURCE = OUT/'source_snapshot'
BASE = ROOT/'results/reviewer_users_m20_m25_m30_seed42_5000_20261102/m20/source_snapshot'
FROZEN = OUT/'oracle_source'
ALGO = 'per_sac_cvx'

def adapt_empty_active_users(path):
    # JSON [] loses the (0,K) matrix shape. Preserve it when reporting energy;
    # this changes no variables, constraints, actions, or physical equations.
    s=path.read_text(encoding='utf-8')
    for name in ('x_ik','fraction_cpu'):
        old=f"np.asarray(matrices['{name}'])"
        assert s.count(old)==1,(name,s.count(old))
        s=s.replace(old,old+'.reshape((A,2))')
    path.write_text(s,encoding='utf-8')

def now(): return datetime.now().astimezone().isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n'
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(raw,encoding='utf-8')
    for attempt in range(12):
        try: tmp.replace(p);return
        except PermissionError:
            if attempt==11:raise
            time.sleep(.25*(attempt+1))

def tag(smoke=False): return ('smoke' if smoke else 'formal')+'_r39_m6_scaled_seed42'
def result(smoke=False): return SOURCE/'results'/tag(smoke)
def controller(): return FROZEN/Path(__file__).name

def available_gib():
    class Mem(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','avail','totalPage','availPage','totalVirtual','availVirtual','extended')]
    m=Mem();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return m.avail/2**30

def source_guard():
    proof=read(OUT/'source_provenance.json')
    for name,digest in proof['source_sha256'].items():assert sha(SOURCE/name)==digest,name
    for name,digest in proof['base_source_sha256'].items():assert sha(BASE/name)==digest,name
    for name,digest in proof['oracle_source_sha256'].items():assert sha(FROZEN/name)==digest,name

def prepare():
    OUT.mkdir(exist_ok=False)
    SOURCE.mkdir();(SOURCE/'experiments').mkdir();FROZEN.mkdir()
    base_meta=read(BASE/'results/formal_users_m20_seed42/per_sac_cvx.manifest.json')
    assert base_meta['status']=='completed' and base_meta['training']==dict(episodes=5000,warmup_steps=8000,batch_size=512)
    old_hash={p.name:sha(p) for p in BASE.glob('*.py')}
    assert old_hash==base_meta['source_sha256']
    for p in BASE.glob('*.py'):shutil.copy2(p,SOURCE/p.name)
    shutil.copy2(BASE/'experiments/train_diverse.json',SOURCE/'experiments/train_diverse.json')
    config=(SOURCE/'config.py').read_text(encoding='utf-8')
    for name,old,new in [('M','20','6'),('F_UAV','3e9','9e8'),('BANDWIDTH','1e6','3e5')]:
        config,n=re.subn(r'^'+name+r'\s*=\s*'+re.escape(old)+r'(?=\s*(?:#|$))',name+' = '+new,config,flags=re.M)
        assert n==1,(name,n)
    config=config.replace('BANDWIDTH = 3e5             # 系统总带宽 B = 1 MHz','BANDWIDTH = 3e5             # M6: 0.3 MHz, scaled by 6/20')
    config=config.replace('F_UAV = 9e8                 # UAV MEC 服务器 CPU 频率 (Hz) = 3 GHz','F_UAV = 9e8                 # M6: 0.9 GHz per UAV, scaled by 6/20')
    (SOURCE/'config.py').write_text(config,encoding='utf-8')
    for name in ('r39_capture_windows.py','r39_grid_oracle.py','r39_grid_validate.py'):
        shutil.copy2(ROOT/'experiments'/name,FROZEN/name)
    # Preserve every executed full-rollout step and the physical full-horizon metric.
    cp=FROZEN/'r39_capture_windows.py';s=cp.read_text(encoding='utf-8')
    old='attempts=attempts, survived=len(traces)==cfg.horizon),'
    new="attempts=attempts, survived=len(traces)==cfg.horizon, traces=traces, cumulative_total_slrr=sum(t['total_slrr'] for t in traces), mean_total_slrr_fixed_horizon=sum(t['total_slrr'] for t in traces)/cfg.horizon),"
    assert s.count(old)==1;s=s.replace(old,new);cp.write_text(s,encoding='utf-8')
    adapt_empty_active_users(FROZEN/'r39_grid_oracle.py')
    shutil.copy2(Path(__file__),controller())
    write(OUT/'source_provenance.json',dict(created_at=now(),base=str(BASE),base_manifest_sha256=sha(BASE/'results/formal_users_m20_seed42/per_sac_cvx.manifest.json'),base_source_sha256=old_hash,source_sha256={p.name:sha(p) for p in SOURCE.glob('*.py')},changed_learning_files=['config.py'],changes={'M':[20,6],'F_UAV':[3e9,9e8],'BANDWIDTH':[1e6,3e5]},oracle_source_sha256={p.name:sha(p) for p in FROZEN.glob('*.py')},capture_change='Full episode trace and fixed50 physical SLRR logging only; catalog supplied by isolated wrapper. Original graph, solver and physical validator unchanged.'))
    write(OUT/'protocol.json',dict(created_at=now(),authorization='Author proposed M6 and CPU/BW scaled by6/20, then explicitly said start calculating in this experiment thread on2026-11-05.',
        training=dict(M=6,K=2,seed=42,episodes=5000,warmup_steps=8000,batch_size=512,threads=2,validation_episodes=50,F_UAV=9e8,BANDWIDTH=3e5,N_MAX=10,checkpoint_every=500),
        reward='Original M * mean-active rho with original battery penalty; not relabeled as sum-rho. Evaluate physical total SLRR separately.',
        controls='Original Greedy association, hidden layers, optimizer, four-cluster diverse hotspots0.9/0.1, local UE CPU, tasks, noise power, K2,4WCT, battery, flight/hover/charging physics, safety and 50-slot rollout unchanged. Input37/output12 from original formulas.',
        caveats=['One training seed only','Resource-per-nominal-user ratio preserved, not actual equal allocation','N_MAX10 preserved and nonbinding at M6','Spatial density and computing energy differ from M20','No exact optimizer/replay/RNG continuation','Concurrent workstation timing diagnostic only'],
        assessment=dict(test_seeds=list(range(982000,982010)),capture_horizon=20,start_slots=[0,10,20,30],full_rollout_slots=50,selection='Predefined windows; all incomplete windows and all executed full episodes retained; no outcome-based selection.'),
        oracle_pilot=dict(horizons=[10,20],test_seed=982000,start_slot=0,spacing_m=25,solve_limit_seconds=900,job_wall_limit_seconds=3600,SCIP_memory_limit_MiB=4096,minimum_free_RAM_GiB=5,maximum_estimated_variables=1500000,gap_J=.001,selection='Same prespecified first test episode, start0; if missing record unavailable, do not replace by favorable outcome.',scope='Original full reachable25m candidate graph, WCT-directed paths and DARS path; clairvoyant joint motion/association/offload/CPU/BW with each terminal battery >= paired DARS. Certification only within candidate graph; no continuous P1 optimality claim.',next_step='Two scalability pilots only. Preserve failures, bounds and charging traces; do not silently truncate horizon, prune graph, or retry. Formal sample size chosen after pilot feasibility is known.'),
        automatic_stages=['smoke22train+2validation and smoke capture/physical oracle validation','fresh5000train+50validation','10 full50-slot frozen test episodes with20-slot captures','H10 and H20 sequential bounded pilot attempts','pilot summary; no extra training/automatic retries']))
    write(OUT/'batch_status.json',dict(status='prepared',created_at=now(),completed_stages=[]))

def train_command(smoke=False):
    return [sys.executable,'-B','-u',str(SOURCE/'train_unified.py'),'--algo',ALGO,'--episodes','22' if smoke else '5000','--seed','42','--tag',tag(smoke),'--env-config',str(SOURCE/'experiments/train_diverse.json'),'--warmup-steps','0' if smoke else '8000','--eval-episodes','2' if smoke else '50','--threads','2']

def child(command,name,timeout):
    env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',R39_SOLVER_DEPS=str(ROOT/'experiments/r39_solver_deps'))
    env.pop('CUDA_VISIBLE_DEVICES',None)
    record=dict(status='running',started_at=now(),command=command)
    with (OUT/f'{name}.log').open('x',encoding='utf-8') as log,(OUT/f'{name}.errors.log').open('x',encoding='utf-8') as err:
        p=subprocess.Popen(command,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=err)
        record['pid']=p.pid;write(OUT/f'{name}.run_status.json',record)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Only this controller's newly owned child, at its declared hard limit.
            record.update(status='hard_timeout',timeout_seconds=timeout,noted_at=now());write(OUT/f'{name}.run_status.json',record)
            p.kill();p.wait();raise RuntimeError(name+' reached declared wall-time limit')
    record.update(status='completed' if code==0 else 'failed',exit_code=code,ended_at=now());write(OUT/f'{name}.run_status.json',record)
    if code:raise RuntimeError(f'{name} exit {code}')

def verify_training(smoke=False):
    r=result(smoke);m=read(r/f'{ALGO}.manifest.json');rows=[json.loads(l) for l in (r/f'{ALGO}.train.jsonl').read_text(encoding='utf-8').splitlines()];val=read(r/f'{ALGO}.validation.json')
    assert len(rows)==(22 if smoke else 5000) and len(val)==(2 if smoke else 50)
    assert m['status']=='completed' and m['training_seed']==42
    assert m['training']==dict(episodes=22 if smoke else 5000,warmup_steps=0 if smoke else 8000,batch_size=512)
    for k,v in dict(M=6,K=2,F_UAV=9e8,BANDWIDTH=3e5,N_MAX=10,STATE_DIM=37,ACTION_DIM=12).items():assert m['system_parameters'][k]==v,(k,m['system_parameters'][k])
    assert m['source_sha256']==read(OUT/'source_provenance.json')['source_sha256']
    assert m['runtime']['device']!='CPU' and m['gradient_updates']>0
    assert all(x['energy_violation_steps']==0 for x in rows+val)
    # CPU-only verification in this separate worker, never alter the GPU run.
    import torch
    weights=torch.load(r/f'{ALGO}.pth',map_location='cpu',weights_only=False)
    def finite(v):
        if torch.is_tensor(v):assert torch.isfinite(v).all()
        elif isinstance(v,dict):
            for x in v.values():finite(x)
        elif isinstance(v,(list,tuple)):
            for x in v:finite(x)
    finite(weights)
    write(r/'completion_verification.json',dict(verified_at=now(),training_rows=len(rows),validation_rows=len(val),gradient_updates=m['gradient_updates'],checkpoint_sha256=sha(r/f'{ALGO}.pth'),source_sha256=m['source_sha256'],formal_evidence=not smoke))

def capture(smoke=False):
    sys.path.insert(0,str(FROZEN));module=importlib.import_module('r39_capture_windows')
    module.catalog=lambda:{(6,42):dict(source=SOURCE,checkpoint=result(smoke)/f'{ALGO}.pth',manifest=result(smoke)/f'{ALGO}.manifest.json')}
    dest=OUT/('smoke_capture' if smoke else 'capture');dest.mkdir(exist_ok=False)
    args=SimpleNamespace(users=6,training_seed=42,episodes=2 if smoke else 10,seed_start=981900 if smoke else 982000,horizon=20,start_slots=[0] if smoke else [0,10,20,30],output=dest)
    module.capture(args)

def pilot(horizon,smoke=False):
    os.environ['R39_SOLVER_DEPS']=str(ROOT/'experiments/r39_solver_deps')
    sys.path.insert(0,str(FROZEN));module=importlib.import_module('r39_grid_oracle')
    dest=OUT/('smoke_oracle' if smoke else f'pilot_h{horizon}');dest.mkdir(exist_ok=False)
    cap=OUT/('smoke_capture' if smoke else 'capture')/'cases.jsonl'
    cases=[json.loads(l) for l in cap.read_text(encoding='utf-8').splitlines()]
    seed=981900 if smoke else 982000
    matches=[c for c in cases if c['test_seed']==seed and c['start_slot']==0]
    if not matches:
        write(dest/'status.json',dict(status='unavailable_prespecified_window',reason='DARS did not execute the complete20-slot source window; no substitute selected.'));return
    case=copy.deepcopy(matches[0]);case['source_window_id']=case['case_id'];case['horizon']=horizon;case['case_id']=case['case_id'].rsplit('_h',1)[0]+f'_h{horizon}'
    case['stages']=case['stages'][:horizon];case['dars']=case['dars'][:horizon];case['terminal_battery_dars']=case['dars'][-1]['battery_after'];case['dars_total_slrr']=sum(s['total_slrr'] for s in case['dars'])
    write(dest/'instance.json',case)
    write(dest/'status.json',dict(status='graph_preflight',pid=os.getpid(),started_at=now()))
    nodes,edges,baseline=module.graph(case,25.)
    pairs=sum(len(nodes[t,0])*len(nodes[t,1]) for t in range(horizon))
    estimate=pairs+sum(len(v) for v in nodes.values())+sum(len(v) for v in edges.values())+2*(horizon+1)
    for t in range(horizon):
        n=sum(len(nodes[t,k]) for k in range(2));a=len(case['stages'][t]['active_indices']);estimate+=a*(50+2*n)
    write(dest/'graph_preflight.json',dict(nodes_per_slot=[[len(nodes[t,k]) for k in range(2)] for t in range(horizon)],joint_position_variables=pairs,conservative_variable_estimate=estimate,estimated_variable_cap=1500000,available_RAM_GiB=available_gib(),scope='Unpruned original25m graph. Size check only, not optimization evidence.'))
    if estimate>1500000:
        write(dest/'status.json',dict(status='resource_guard_not_solved',reason='Conservative model estimate exceeds pilot allocation; original graph retained and not silently pruned.',estimated_variables=estimate));return
    # Reuse the exact preflight graph; skip duplicate expensive graph generation.
    module.graph=lambda case,spacing:(nodes,edges,baseline)
    args=SimpleNamespace(spacing=25.,time_limit=10. if smoke else 900.,gap=.001,output=dest)
    module.write(dest/'protocol.json',dict(arguments=vars(args),source_cases_sha256=sha(cap),source_hashes={p.name:sha(p) for p in FROZEN.glob('*.py')},selection='Fixed first test seed/start0',formal_evidence=not smoke,scope=read(OUT/'protocol.json')['oracle_pilot']['scope']))
    solved=module.solve(case,args)
    if smoke:assert solved['valid_reference'],'Smoke physical oracle audit failed'

def queue(smoke=False):
    source_guard()
    sf=OUT/('smoke_status.json' if smoke else 'batch_status.json')
    if smoke:assert not sf.exists();state=dict(status='prepared',completed_stages=[])
    else:
        state=read(sf);assert state['status']=='prepared','Do not duplicate an existing launch'
        assert read(OUT/'smoke_status.json')['status']=='completed'
    state.update(status='running',runner_pid=os.getpid(),started_at=now());write(sf,state)
    try:
        while available_gib()<1.25:
            state.update(status='waiting_for_training_memory',available_RAM_GiB=available_gib(),updated_at=now());write(sf,state);time.sleep(30)
        state.update(status='training',updated_at=now());write(sf,state)
        child(train_command(smoke),'smoke_training' if smoke else 'training',3600 if smoke else 172800)
        child([sys.executable,'-B',str(controller()),'--verify']+(['--smoke'] if smoke else []),'smoke_verify' if smoke else 'verify',300)
        state['completed_stages'].append('training_and_validation');state.update(status='capturing',updated_at=now());write(sf,state)
        child([sys.executable,'-B','-u',str(controller()),'--capture']+(['--smoke'] if smoke else []),'smoke_capture' if smoke else 'capture',600 if smoke else 3600)
        state['completed_stages'].append('frozen_full_rollout_capture')
        for h in ([3] if smoke else [10,20]):
            if not smoke:
                while available_gib()<5:
                    state.update(status='waiting_for_oracle_memory',horizon=h,available_RAM_GiB=available_gib(),updated_at=now());write(sf,state);time.sleep(30)
            state.update(status='oracle_pilot',horizon=h,updated_at=now());write(sf,state)
            try:
                child([sys.executable,'-B','-u',str(controller()),'--pilot','--horizon',str(h)]+(['--smoke'] if smoke else []),'smoke_oracle' if smoke else f'pilot_h{h}',600 if smoke else 3600)
            except Exception as exc:
                if smoke:raise
                state.setdefault('pilot_failures',[]).append(dict(horizon=h,error=repr(exc)))
            state['completed_stages'].append(f'pilot_h{h}_attempt_finished');write(sf,state)
        source_guard()
        summaries=[]
        for h in ([3] if smoke else [10,20]):
            d=OUT/('smoke_oracle' if smoke else f'pilot_h{h}')
            v=read(d/'result.json') if (d/'result.json').exists() else read(d/'status.json') if (d/'status.json').exists() else {}
            summaries.append(dict(horizon=h,**{k:v[k] for k in ('status','valid_reference','certified_within_requested_gap','dars_total_slrr','oracle_total_slrr','oracle_slrr_upper_bound','verified_J_gap','solve_seconds') if k in v}))
        write(OUT/('smoke_summary.json' if smoke else 'pilot_summary.json'),dict(created_at=now(),results=summaries,formal_batch_not_started=True,training_seed=42,full_rollout_metrics='capture/episodes.jsonl includes all executed steps and fixed50 total SLRR. Incomplete windows retained separately.',scope=read(OUT/'protocol.json')['oracle_pilot']['scope']))
        state.update(status='completed' if smoke else 'pilot_finished_review_required',ended_at=now(),pilot_results=summaries);write(sf,state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),ended_at=now());write(sf,state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--queue',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--capture',action='store_true');p.add_argument('--verify',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--horizon',type=int,choices=[3,10,20],default=10);a=p.parse_args()
    if a.prepare:prepare()
    elif a.verify:verify_training(a.smoke)
    elif a.capture:capture(a.smoke)
    elif a.pilot:pilot(a.horizon,a.smoke)
    elif a.queue:queue(a.smoke)
