"""User-requested5000 episodes for M20/25; explicitly reuse15000-episode M30 models; paused."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/formal_all_five_seeds43_44_restart_20261031/source_snapshot'
OUT=ROOT/'results/reviewer_users_m20_m25_m30_seed42_5000_20261102'
ALGOS=['per_sac_cvx','esac_per_cvx','sac_cvx','td3_cvx','maddpg_cvx']
def now():return datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def src(m):return OUT/f'm{m}/source_snapshot'
def tag(m,smoke=False):return f'{"smoke" if smoke else "formal"}_users_m{m}_seed42'
def reused(algo):
    if algo in ('td3_cvx','maddpg_cvx'):
        return ROOT/'results/remaining_td3_maddpg_seed42_20261030/source_snapshot/results/remaining_td3_maddpg_seed42_20261030'
    return ROOT/'results/original_main_seed42_gpu_retry_20261029_234000'
def result(m,algo,smoke=False):return reused(algo) if m==30 else src(m)/'results'/tag(m,smoke)
def cmd(m,algo,smoke=False):
    return [sys.executable,'-B','-u',str(src(m)/'train_unified.py'),'--algo',algo,'--episodes','22' if smoke else '5000',
            '--seed','42','--tag',tag(m,smoke),'--env-config',str(src(m)/'experiments/train_diverse.json'),
            '--warmup-steps','0' if smoke else '8000','--eval-episodes','2' if smoke else '50','--threads','2']
def replace(p,a,b):
    s=p.read_text(encoding='utf-8');assert s.count(a)==1,(p,a);p.write_text(s.replace(a,b),encoding='utf-8')

def prepare():
    OUT.mkdir(exist_ok=False)
    old_hash={p.name:sha(p) for p in OLD.glob('*.py')};models=[]
    for algo in ALGOS:
        p=reused(algo);meta=read(p/f'{algo}.manifest.json')
        assert meta['status']=='completed' and meta['training_seed']==42
        assert meta['training']==dict(episodes=15000,warmup_steps=8000,batch_size=512)
        assert meta['source_sha256']==old_hash
        assert meta['system_parameters']['M']==30 and meta['system_parameters']['K']==2
        assert (meta['system_parameters']['HOTSPOT_HOT_PROB'],meta['system_parameters']['HOTSPOT_COLD_PROB'])==(.9,.1)
        models.append(dict(M=30,algorithm=algo,manifest=str(p/f'{algo}.manifest.json'),checkpoint=str(p/f'{algo}.pth'),sha256=sha(p/f'{algo}.pth'),status='reused_completed'))
    for m in (20,25,30):
        s=src(m);s.mkdir(parents=True);(s/'experiments').mkdir()
        for p in OLD.glob('*.py'):shutil.copy2(p,s/p.name)
        shutil.copy2(OLD/'experiments/train_diverse.json',s/'experiments/train_diverse.json')
        if m!=30:
            p=s/'config.py';text=p.read_text(encoding='utf-8');text,n=re.subn(r'^M\s*=\s*30(?=\s*(?:#|$))',f'M = {m}',text,flags=re.M);assert n==1;p.write_text(text,encoding='utf-8')
        p=s/'evaluation.py'
        replace(p,'        self.reward = 0.0','        self.reward = 0.0\n        self.total_slrr = 0.0')
        replace(p,"            self.rhos.append(float(info['rho_mean']))","            self.rhos.append(float(info['rho_mean']))\n            self.total_slrr += float(np.sum(info['rho']))")
        replace(p,'                   termination_reason=self.reason, reward=self.reward,','                   termination_reason=self.reason, reward=self.reward,\n                   cumulative_total_slrr=self.total_slrr,\n                   mean_total_slrr_fixed_horizon=self.total_slrr / self.horizon,')
        p=s/'train_unified.py'
        replace(p,'        ep_rewards.append(ep_reward)',"        if needs_training and (ep + 1) % 500 == 0:\n            agent.save(str(result_dir / f'{algo_name}.episode{ep+1}.pth'))\n        ep_rewards.append(ep_reward)")
        hashes={p.name:sha(p) for p in s.glob('*.py')}
        write(OUT/f'm{m}/source_provenance.json',dict(original_source=str(OLD),source_sha256=hashes,
              changed_files=[n for n,h in hashes.items() if old_hash[n]!=h],M=m,K=2,state_dim=13+4*m,action_dim=6+m))
    jobs=[dict(M=m,algorithm=a,status='queued',command=cmd(m,a)) for a in ALGOS for m in (20,25)]
    write(OUT/'protocol.json',dict(created_at=now(),authorization='Author paused user-count sweep and changed every training budget to5000 on2026-11-02. Author explicitly retained existing15000-episode M30 models without retraining. Only M20/25 use5000; prepared only, not resumed.',
        algorithms=ALGOS,user_counts=[20,25,30],training_seed=42,K=2,hot=.9,cold=.1,episodes=5000,warmup_steps=8000,batch_size=512,threads=2,
        reused_models=models,training_order=jobs,
        changes='Only M changes (including original formula-based dimensions and M-dependent reward scaling). Original Greedy association, diverse hotspots, energy rules and reward preserved. Added totalSLRR logging and500-episode weight snapshots. No exact resume state.',
        assessment='Independent100 frozen-policy episodes per model with new seeds931000-931099. Same external states across algorithms within M checked by hashes; across M compare distributions, not identical populations.',
        outputs='Only(a) evaluation return and(b) mean active-user completion delay for user sensitivity; no(c). Separate fly/hover/compute/total task-service energy and mean/P95 end-to-end decision latency plots. Preserve totalSLRR and survival in data.',
        timing='Fresh policy evaluation timed on local workstation GPU,2 CPU threads, after known K3/K4 and this queue are finished; exclude training/backprop/logging. Includes preprocessing+policy+safety+association+allocation, includes rejected attempts. No onboard or neural-inference energy measurement claimed.',
        limitations='One training seed; uncertainty across test episodes is conditional on that model, not training-seed robustness. Early-terminated episodes retain actual energy, fixed50 totalSLRR denominator and zero future contribution. No fabricated post-termination energy.',
        plotting=dict(font='Times New Roman',pdf_fonttype=42,png_dpi=600,overwrite_manuscript_figures=False)))
    write(OUT/'batch_status.json',dict(status='prepared',jobs=jobs,reused_models=models,created_at=now()))

def verify(m,algo,smoke=False):
    r=result(m,algo,smoke);meta=read(r/f'{algo}.manifest.json')
    rows=[json.loads(l) for l in (r/f'{algo}.train.jsonl').read_text(encoding='utf-8').splitlines()]
    val=read(r/f'{algo}.validation.json')
    assert len(rows)==(22 if smoke else 5000) and len(val)==(2 if smoke else 50)
    assert meta['status']=='completed' and meta['system_parameters']['M']==m and meta['system_parameters']['K']==2
    assert meta['runtime']['device']!='CPU' and meta['training']['batch_size']==512
    assert all(x['energy_violation_steps']==0 for x in rows+val)
    import torch
    weights=torch.load(r/f'{algo}.pth',map_location='cpu',weights_only=False)
    def finite(x):
        if torch.is_tensor(x):assert torch.isfinite(x).all()
        elif isinstance(x,dict):
            for v in x.values():finite(v)
        elif isinstance(x,(tuple,list)):
            for v in x:finite(v)
    finite(weights)
    write(r/'verification.json',dict(verified_at=now(),training_rows=len(rows),validation_rows=len(val),checkpoint_sha256=sha(r/f'{algo}.pth'),training=meta['training'],runtime=meta['runtime']))

def run_train(m,algo,smoke=False):
    dest=OUT/f'm{m}';prefix=('smoke_' if smoke else '')+algo
    env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (dest/f'{prefix}.log').open('ab') as log,(dest/f'{prefix}.errors.log').open('ab') as err:
        p=subprocess.Popen(cmd(m,algo,smoke),cwd=src(m),env=env,stdout=log,stderr=err,stdin=subprocess.DEVNULL)
        write(dest/f'{prefix}.run_status.json',dict(status='running',pid=p.pid,started_at=now(),M=m,algorithm=algo))
        code=p.wait()
    if code:raise RuntimeError(f'{m}/{algo} exit{code}')
    verify(m,algo,smoke)
    write(dest/f'{prefix}.run_status.json',dict(status='completed',pid=p.pid,ended_at=now(),M=m,algorithm=algo))

def known_scaling_active():
    jobs=[]
    for k in (3,4):
        p=ROOT/f'results/reviewer_uav_scaling_k3_k4_seed42_20261102/k{k}/run_status.json'
        if p.exists() and read(p)['status'] not in ('completed','failed'):jobs.append(f'K{k}')
    return jobs

def queue(smoke=False):
    sf=OUT/('smoke_status.json' if smoke else 'batch_status.json')
    state=dict(status='running',jobs=read(OUT/'protocol.json')['training_order'],started_at=now(),runner_pid=os.getpid()) if smoke else read(sf)
    if not smoke:assert state['status']=='prepared','Never relaunch an existing batch'
    state.update(status='running',runner_pid=os.getpid(),started_at=now());write(sf,state)
    try:
        for j in state['jobs']:
            j.update(status='running',started_at=now());write(sf,state)
            run_train(j['M'],j['algorithm'],smoke)
            j.update(status='completed',ended_at=now());write(sf,state)
        if not smoke:
            while known_scaling_active():
                state.update(status='waiting_for_timing_window',waiting_for=known_scaling_active(),updated_at=now());write(sf,state);time.sleep(30)
            state.update(status='evaluating',waiting_for=[]);write(sf,state)
            for m in (20,25,30):
                for a in ALGOS:
                    state.update(current_evaluation=dict(M=m,algorithm=a),updated_at=now());write(sf,state)
                    subprocess.run([sys.executable,'-B','-u',str(ROOT/'experiments/evaluate_user_sweep_5000_20261102.py'),'--m',str(m),'--algo',a],check=True)
            state.update(status='plotting');write(sf,state)
            subprocess.run([sys.executable,'-B',str(ROOT/'experiments/plot_user_sweep_5000_20261102.py')],check=True)
        state.update(status='completed',ended_at=now());write(sf,state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc),ended_at=now());write(sf,state);raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--queue',action='store_true');a=ap.parse_args()
    if a.prepare:prepare()
    elif a.queue:queue(a.smoke)
