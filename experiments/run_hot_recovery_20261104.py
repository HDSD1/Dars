"""Recover interrupted hotspot queue, preserving four completed models and all old records."""
import argparse
import os
import shutil
import subprocess
import sys
import run_hot_sweep_20261103 as core

ROOT=core.ROOT
ORIGINAL=core.OUT
OUT=ROOT/'results/reviewer_hot_p60_p100_seed42_10000_recovery_20261104'
ALGOS=core.ALGOS
HOT=core.HOT
read,write,sha,now=core.read,core.write,core.sha,core.now
REUSED={(60,a) for a in ALGOS if a!='maddpg_cvx'}
core.OUT=OUT
src=core.src
def result(h,a,smoke=False):
    if (h,a) in REUSED and not smoke:
        return ORIGINAL/f'h{h}/source_snapshot/results'/core.tag(h)
    return src(h)/'results'/core.tag(h,smoke)
core.result=result

def prepare():
    OUT.mkdir(exist_ok=False)
    reused=[]
    for h,a in sorted(REUSED):
        folder=result(h,a);meta=read(folder/f'{a}.manifest.json');validation=read(folder/f'{a}.validation.json');verification=read(folder/f'{a}.verification.json')
        assert meta['status']=='completed' and meta['training']['episodes']==10000 and len(validation)==50
        assert verification['checkpoint_sha256']==sha(folder/f'{a}.pth')
        reused.append(dict(hot=h,algorithm=a,source=str(folder),checkpoint_sha256=sha(folder/f'{a}.pth')))
    for h in HOT:
        old=ORIGINAL/f'h{h}/source_snapshot';s=src(h);s.mkdir(parents=True);(s/'experiments').mkdir()
        for p in old.glob('*.py'):shutil.copy2(p,s/p.name)
        shutil.copy2(old/'experiments/train_diverse.json',s/'experiments/train_diverse.json')
        hashes={p.name:sha(p) for p in s.glob('*.py')}
        assert hashes=={p.name:sha(p) for p in old.glob('*.py')}
        write(OUT/f'h{h}/source_provenance.json',dict(source=str(old),source_sha256=hashes,identical_learning_source=True))
    jobs=[dict(hot=h,algorithm=a,status='reused_completed' if (h,a) in REUSED else 'queued') for h in HOT for a in ALGOS]
    protocol=read(ORIGINAL/'protocol.json');protocol.update(created_at=now(),recovery_authorization='Prior explicit author request to restart failed experiments; Swap cancellation remains respected.',original_root=str(ORIGINAL),reused_completed_models=reused,training_order=jobs,
        interruption='Original supervisor39580 and worker61960 absent (OpenProcess error87); managed session51260 gone;MADDPG0.6 logs froze9564,stderr empty. Cause unknown. Original records/status preserved.',
        recovery='Only interrupted MADDPG0.6 restarts fresh10000 with unchanged seed42/budget/source;20unstarted jobs follow. Four completed0.6 models reused. Weight-only checkpoint9500 lacks full replay/RNG training state;not exact continuation. No new Swap run.')
    write(OUT/'protocol.json',protocol);write(OUT/'batch_status.json',dict(status='prepared',jobs=jobs,reused_completed_models=reused,created_at=now()))

def queue():
    sf=OUT/'batch_status.json';state=read(sf);assert state['status']=='prepared','Do not duplicate queue'
    state.update(status='running',runner_pid=os.getpid(),started_at=now());write(sf,state)
    try:
        for j in state['jobs']:
            if j['status']=='reused_completed':continue
            j.update(status='running',started_at=now());write(sf,state)
            core.run_train(j['hot'],j['algorithm'])
            j.update(status='completed',ended_at=now());write(sf,state)
        state.update(status='evaluating');write(sf,state)
        for h in HOT:
            for a in ALGOS:
                state.update(current_evaluation=dict(hot=h,algorithm=a));write(sf,state)
                subprocess.run([sys.executable,'-B','-u',str(ROOT/'experiments/evaluate_hot_recovery_20261104.py'),'--hot',str(h),'--algo',a],check=True)
        state.update(status='plotting');write(sf,state)
        subprocess.run([sys.executable,'-B',str(ROOT/'experiments/plot_hot_recovery_20261104.py')],check=True)
        state.update(status='completed',ended_at=now());write(sf,state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc),ended_at=now());write(sf,state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--queue',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.queue:queue()
