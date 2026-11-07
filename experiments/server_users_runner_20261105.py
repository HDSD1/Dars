"""Isolated Slurm task; fresh training only; never overwrite an existing task."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT
ALGOS=['per_sac_cvx','esac_per_cvx','sac_cvx','td3_cvx','maddpg_cvx']
def now():return datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def src(m):return ROOT/f'sources/m{m}'
def tag(m,seed,smoke=False):return f'{"smoke" if smoke else "formal"}_users_m{m}_seed{seed}_10000_server'
def result(m,algo,smoke=False):return src(m)/'results'/tag(m,int(os.environ['IOTJ_TRAIN_SEED']),smoke)

def verify(m,algo,seed,smoke):
    import torch
    folder=result(m,algo,smoke)
    meta=read(folder/f'{algo}.manifest.json')
    rows=[json.loads(l) for l in (folder/f'{algo}.train.jsonl').read_text(encoding='utf-8').splitlines()]
    val=read(folder/f'{algo}.validation.json')
    assert len(rows)==(22 if smoke else 10000) and len(val)==(2 if smoke else 50)
    assert meta['status']=='completed' and meta['training_seed']==seed
    assert meta['training']==dict(episodes=22 if smoke else 10000,warmup_steps=0 if smoke else 8000,batch_size=512)
    assert meta['system_parameters']['M']==m and meta['system_parameters']['K']==2
    assert meta['runtime']['device']!='CPU'
    assert all(r['energy_violation_steps']==0 for r in rows+val)
    if smoke:assert sum(r['executed_steps'] for r in rows)>512
    def finite(x):
        if torch.is_tensor(x):assert torch.isfinite(x).all()
        elif isinstance(x,dict):
            for v in x.values():finite(v)
        elif isinstance(x,(list,tuple)):
            for v in x:finite(v)
    finite(torch.load(folder/f'{algo}.pth',map_location='cpu',weights_only=False))
    write(folder/f'{algo}.verification.json',dict(verified_at=now(),training_rows=len(rows),validation_rows=len(val),checkpoint_sha256=sha(folder/f'{algo}.pth'),training=meta['training'],runtime=meta['runtime']))

def run(index):
    protocol=read(ROOT/'protocol.json');job=protocol['jobs'][index]
    m,seed,algo=job['M'],job['seed'],job['algorithm']
    os.environ['IOTJ_TRAIN_SEED']=str(seed)
    folder=ROOT/f'tasks/{index:03d}_m{m}_seed{seed}_{algo}'
    folder.mkdir(parents=True,exist_ok=False)
    state=dict(job,status='preflight',started_at=now(),pid=os.getpid(),slurm_job=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'))
    write(folder/'run_status.json',state)
    try:
        # Detect edits or damaged upload before importing training code.
        for rel,digest in read(ROOT/'package_hashes.json').items():assert sha(ROOT/rel)==digest,rel
        import torch
        assert torch.cuda.is_available(),'GPU is required; never silently fall back to CPU'
        gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True,encoding='utf-8',check=True).stdout
        (folder/'gpu_start.txt').write_text(gpu,encoding='utf-8')
        write(folder/'runtime.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),node=state['node'],slurm_job=state['slurm_job'],visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
        for smoke in [True,False]:
            stage='smoke' if smoke else 'training'
            state.update(status=stage,updated_at=now());write(folder/'run_status.json',state)
            cmd=[sys.executable,'-B','-u',str(src(m)/'train_unified.py'),'--algo',algo,'--episodes','22' if smoke else '10000','--seed',str(seed),'--tag',tag(m,seed,smoke),'--env-config',str(src(m)/'experiments/train_diverse.json'),'--warmup-steps','0' if smoke else '8000','--eval-episodes','2' if smoke else '50','--threads','2']
            with (folder/f'{stage}.log').open('xb') as out,(folder/f'{stage}.errors.log').open('xb') as err:
                child=subprocess.Popen(cmd,cwd=src(m),stdout=out,stderr=err)
                state.update(child_pid=child.pid,command=cmd);write(folder/'run_status.json',state)
                code=child.wait()
            if code:raise RuntimeError(f'{stage} exited {code}')
            verify(m,algo,seed,smoke)
        state.update(status='evaluating',updated_at=now());write(folder/'run_status.json',state)
        with (folder/'evaluation.log').open('xb') as out,(folder/'evaluation.errors.log').open('xb') as err:
            subprocess.run([sys.executable,'-B','-u',str(ROOT/'evaluate_server_users.py'),'--m',str(m),'--algo',algo],stdout=out,stderr=err,check=True)
        assert read(ROOT/f'evaluation/seed{seed}/m{m}/{algo}/manifest.json')['status']=='completed'
        state.update(status='completed',ended_at=now());write(folder/'run_status.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),ended_at=now());write(folder/'run_status.json',state)
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True)
    run(parser.parse_args().index)
