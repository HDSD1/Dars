"""Second authorized recovery: reuse ten completed models; preserve both interrupted roots."""
import argparse
import os
import shutil
import subprocess
import sys
import run_hot_sweep_20261103 as core

ROOT=core.ROOT
ORIGINAL=core.OUT
PREVIOUS=ROOT/'results/reviewer_hot_p60_p100_seed42_10000_recovery_20261104'
OUT=ROOT/'results/reviewer_hot_p60_p100_seed42_10000_recovery2_20261104'
ALGOS=core.ALGOS
HOT=core.HOT
read,write,sha,now=core.read,core.write,core.sha,core.now
REUSED={(h,a) for h in (60,70) for a in ALGOS}
core.OUT=OUT
src=core.src
def result(h,a,smoke=False):
    if (h,a) in REUSED and not smoke:
        source=ORIGINAL if h==60 and a!='maddpg_cvx' else PREVIOUS
        return source/f'h{h}/source_snapshot/results'/core.tag(h)
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
        old=PREVIOUS/f'h{h}/source_snapshot';s=src(h);s.mkdir(parents=True);(s/'experiments').mkdir()
        for p in old.glob('*.py'):shutil.copy2(p,s/p.name)
        shutil.copy2(old/'experiments/train_diverse.json',s/'experiments/train_diverse.json')
        hashes={p.name:sha(p) for p in s.glob('*.py')}
        assert hashes=={p.name:sha(p) for p in old.glob('*.py')}
        write(OUT/f'h{h}/source_provenance.json',dict(source=str(old),source_sha256=hashes,identical_learning_source=True))
    jobs=[dict(hot=h,algorithm=a,status='reused_completed' if (h,a) in REUSED else 'queued') for h in HOT for a in ALGOS]
    protocol=read(ORIGINAL/'protocol.json');protocol.update(created_at=now(),recovery_authorization='Author explicitly requested immediate hotspot restart on2026-11-04, not waiting for next check; Swap remains cancelled.',previous_recovery_root=str(PREVIOUS),original_root=str(ORIGINAL),reused_completed_models=reused,training_order=jobs,
        interruption='Previous recovery supervisor14348 andworker26380 exited; h80 DARS3661 records frozen, stderr empty. Root cause unknown. Both prior roots preserved.',
        recovery='h80 DARS restarts fresh10000 seed42, warmup8000,batch512,threads2;14unstarted jobs follow. Reuse ten completed h60/h70 models from explicit original/previous mapping. Weight checkpoints lack optimizer/replay/RNG;not exact continuation. Independent hidden background launch with dedicated supervisor logs. Learning source byte-identical. No Swap run.')
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
                subprocess.run([sys.executable,'-B','-u',str(ROOT/'experiments/evaluate_hot_recovery2_20261104.py'),'--hot',str(h),'--algo',a],check=True)
        state.update(status='plotting');write(sf,state)
        subprocess.run([sys.executable,'-B',str(ROOT/'experiments/plot_hot_recovery2_20261104.py')],check=True)
        state.update(status='completed',ended_at=now());write(sf,state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc),ended_at=now());write(sf,state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--queue',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.queue:queue()
