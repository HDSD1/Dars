"""Author-authorized hotspot probability sweep: five algorithms, seed42, 10000 episodes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'results/reviewer_users_m20_m25_m30_seed42_5000_20261102/m30/source_snapshot'
OUT = ROOT/'results/reviewer_hot_p60_p100_seed42_10000_20261103'
ALGOS = ['per_sac_cvx','esac_per_cvx','sac_cvx','td3_cvx','maddpg_cvx']
HOT = [60,70,80,90,100]
def now(): return datetime.now().astimezone().isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,obj):
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(40):
        try: tmp.replace(p); return
        except PermissionError:
            if attempt==39: raise
            time.sleep(.25)
def src(h): return OUT/f'h{h}/source_snapshot'
def tag(h,smoke=False): return f'{"smoke" if smoke else "formal"}_hot_h{h}_seed42'
def result(h,algo,smoke=False): return src(h)/'results'/tag(h,smoke)
def command(h,a,smoke=False):
    return [sys.executable,'-B','-u',str(src(h)/'train_unified.py'),'--algo',a,'--episodes','22' if smoke else '10000','--seed','42','--tag',tag(h,smoke),'--env-config',str(src(h)/'experiments/train_diverse.json'),'--warmup-steps','0' if smoke else '8000','--eval-episodes','2' if smoke else '50','--threads','2']
def prepare():
    OUT.mkdir(exist_ok=False)
    for h in HOT:
        s=src(h);s.mkdir(parents=True);(s/'experiments').mkdir()
        for p in OLD.glob('*.py'):shutil.copy2(p,s/p.name)
        shutil.copy2(OLD/'experiments/train_diverse.json',s/'experiments/train_diverse.json')
        p=s/'config.py';text=p.read_text(encoding='utf-8')
        text,n=re.subn(r'^HOTSPOT_HOT_PROB\s*=\s*0\.9[^\n]*',f'HOTSPOT_HOT_PROB = {h/100:.1f}  # Author-authorized probability sweep',text,flags=re.M);assert n==1
        p.write_text(text,encoding='utf-8')
        assert re.search(r'^M\s*=\s*30\b',text,re.M) and re.search(r'^K\s*=\s*2\b',text,re.M)
        assert re.search(r'^HOTSPOT_COLD_PROB\s*=\s*0\.1\b',text,re.M)
        write(OUT/f'h{h}/source_provenance.json',dict(source=str(OLD),hot=h/100,cold=.1,source_sha256={p.name:sha(p) for p in s.glob('*.py')},changed_from_source=[p.name for p in s.glob('*.py') if sha(p)!=sha(OLD/p.name)]))
    jobs=[dict(hot=h,algorithm=a,status='queued') for h in HOT for a in ALGOS]
    write(OUT/'protocol.json',dict(created_at=now(),authorization='Author requested continuing previous hotspot values with10000 training episodes; prior five-algorithm/single-seed scope retained.',hot_probabilities=[h/100 for h in HOT],cold_probability=.1,M=30,K=2,algorithms=ALGOS,training_seed=42,episodes=10000,warmup_steps=8000,batch_size=512,threads=2,training_order=jobs,
        training_definition='Original Greedy, four-cluster diverse environment, same networks/reward/safety. Only hotspot probability varies. All25 models train fresh10000, including0.9; no15000 checkpoint substitution.',
        reward='Original M times mean-active-user rho plus battery penalty; separately log physical cumulative totalSLRR and fixed50 mean totalSLRR.',
        evaluation='100 frozen episodes/model, new seeds941000-941099, same-hot external trajectories checked across algorithms. Across hot probabilities demand distributions differ.',
        timing='Sequential evaluation after this entire training queue; synchronized GPU policy+preprocessing+safety/association/allocation, including rejected attempts; local workstation, not airborne. Check unrelated overlap before using as clean benchmark.',
        energy='Actual simulated flight/hover/task-computation/total and charging up to termination; excludes neural inference and control radio. Future totalSLRR zero after termination, fixed50 denominator.',
        figures='Only(a) evaluation return and(b) completion delay; separate energy and slot timing. Preserve charging/completion/totalSLRR. Times New Roman, manuscript palette, PDF/600dpiPNG/SVG, independent directory; inspect before delivery.',
        limitations='Single training seed42; test-episode confidence intervals are conditional on each policy. Probability sensitivity is not unseen-pattern generalization.'))
    write(OUT/'batch_status.json',dict(status='prepared',jobs=jobs,created_at=now()))
def verify(h,a,smoke):
    r=result(h,a,smoke);meta=read(r/f'{a}.manifest.json');rows=[json.loads(l) for l in (r/f'{a}.train.jsonl').read_text(encoding='utf-8').splitlines()];val=read(r/f'{a}.validation.json')
    assert len(rows)==(22 if smoke else 10000) and len(val)==(2 if smoke else 50)
    assert meta['status']=='completed' and meta['system_parameters']['HOTSPOT_HOT_PROB']==h/100 and meta['system_parameters']['HOTSPOT_COLD_PROB']==.1
    assert meta['system_parameters']['M']==30 and meta['system_parameters']['K']==2 and meta['runtime']['device']!='CPU'
    assert all(x['energy_violation_steps']==0 for x in rows+val)
    import torch
    def finite(x):
        if torch.is_tensor(x): assert torch.isfinite(x).all()
        elif isinstance(x,dict):
            for v in x.values():finite(v)
        elif isinstance(x,(list,tuple)):
            for v in x:finite(v)
    finite(torch.load(r/f'{a}.pth',map_location='cpu',weights_only=False))
    write(r/f'{a}.verification.json',dict(training_rows=len(rows),validation_rows=len(val),checkpoint_sha256=sha(r/f'{a}.pth'),verified_at=now()))
def run_train(h,a,smoke=False):
    dest=OUT/f'h{h}';prefix=('smoke_' if smoke else '')+a;env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (dest/f'{prefix}.log').open('ab') as log,(dest/f'{prefix}.errors.log').open('ab') as err:
        p=subprocess.Popen(command(h,a,smoke),cwd=src(h),env=env,stdout=log,stderr=err,stdin=subprocess.DEVNULL)
        write(dest/f'{prefix}.run_status.json',dict(status='running',pid=p.pid,started_at=now(),hot=h/100,algorithm=a))
        code=p.wait()
    if code:
        write(dest/f'{prefix}.run_status.json',dict(status='failed',pid=p.pid,exit_code=code,ended_at=now()));raise RuntimeError(f'{h}/{a} exit{code}')
    verify(h,a,smoke);write(dest/f'{prefix}.run_status.json',dict(status='completed',pid=p.pid,ended_at=now(),hot=h/100,algorithm=a))
def queue(smoke=False):
    sf=OUT/('smoke_status.json' if smoke else 'batch_status.json')
    state=dict(status='prepared',jobs=[dict(hot=60,algorithm=a,status='queued') for a in ALGOS]+[dict(hot=100,algorithm='per_sac_cvx',status='queued')]) if smoke else read(sf)
    assert state['status']=='prepared','Never duplicate/relaunch an existing queue'
    state.update(status='running',runner_pid=os.getpid(),started_at=now());write(sf,state)
    try:
        for j in state['jobs']:
            j.update(status='running',started_at=now());write(sf,state);run_train(j['hot'],j['algorithm'],smoke);j.update(status='completed',ended_at=now());write(sf,state)
        if not smoke:
            state.update(status='evaluating');write(sf,state)
            for h in HOT:
                for a in ALGOS:
                    state.update(current_evaluation=dict(hot=h,algorithm=a));write(sf,state)
                    subprocess.run([sys.executable,'-B','-u',str(ROOT/'experiments/evaluate_hot_sweep_20261103.py'),'--hot',str(h),'--algo',a],check=True)
            state.update(status='plotting');write(sf,state)
            subprocess.run([sys.executable,'-B',str(ROOT/'experiments/plot_hot_sweep_20261103.py')],check=True)
        state.update(status='completed',ended_at=now());write(sf,state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc),ended_at=now());write(sf,state);raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--queue',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.queue:queue(a.smoke)
