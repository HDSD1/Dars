"""Persistent single-run supervisor. No automatic retries or model overwrite."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT
ALGOS=['per_sac_cvx']
def now():return datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,obj):
    tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(20):
        try:tmp.replace(p);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.25)
def src(m):
    assert m==30
    return ROOT/'source_snapshot'
def tag(smoke=False):return f'{"smoke" if smoke else "formal"}_wct1_center_dars_seed42'
def result(m,algo,smoke=False):
    assert algo=='per_sac_cvx'
    return src(m)/'results'/tag(smoke)

def verify(smoke):
    import torch
    folder=result(30,'per_sac_cvx',smoke);algo='per_sac_cvx'
    meta=read(folder/f'{algo}.manifest.json')
    rows=[json.loads(l) for l in (folder/f'{algo}.train.jsonl').read_text(encoding='utf-8').splitlines()]
    val=read(folder/f'{algo}.validation.json')
    assert len(rows)==(22 if smoke else 10000) and len(val)==(2 if smoke else 50)
    assert meta['status']=='completed' and meta['training_seed']==42
    assert meta['training']==dict(episodes=22 if smoke else 10000,warmup_steps=0 if smoke else 8000,batch_size=512)
    assert meta['system_parameters']['M']==30 and meta['system_parameters']['K']==2
    assert meta['environment']['wct_indices']==[0] and meta['runtime']['device']!='CPU'
    assert all(r['energy_violation_steps']==0 for r in rows+val)
    for r in rows+val:assert abs(r['mean_total_slrr_fixed_horizon']*50-r['cumulative_total_slrr'])<1e-8
    assert meta['gradient_updates']>0
    if smoke:assert sum(r['executed_steps'] for r in rows)>512
    def finite(x):
        if torch.is_tensor(x):assert torch.isfinite(x).all()
        elif isinstance(x,dict):
            for v in x.values():finite(v)
        elif isinstance(x,(list,tuple)):
            for v in x:finite(v)
    finite(torch.load(folder/f'{algo}.pth',map_location='cpu',weights_only=False))
    write(folder/f'{algo}.verification.json',dict(verified_at=now(),training_rows=len(rows),validation_rows=len(val),checkpoint_sha256=sha(folder/f'{algo}.pth'),gradient_updates=meta['gradient_updates'],executed_training_steps=sum(r['executed_steps'] for r in rows),runtime=meta['runtime']))

def run():
    assert read(ROOT/'run_status.json')['status']=='prepared'
    with (ROOT/'launch.lock').open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    state=dict(status='preflight',started_at=now(),supervisor_pid=os.getpid(),training_seed=42,episodes=10000)
    write(ROOT/'run_status.json',state)
    env=os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8',PYTHONUTF8='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def child(cmd,stage,cwd=ROOT):
        with (ROOT/f'{stage}.log').open('xb') as out,(ROOT/f'{stage}.errors.log').open('xb') as err:
            proc=subprocess.Popen(cmd,cwd=cwd,stdout=out,stderr=err,stdin=subprocess.DEVNULL,env=env)
            state.update(status=stage,child_pid=proc.pid,command=cmd,updated_at=now());write(ROOT/'run_status.json',state)
            code=proc.wait()
        if code:raise RuntimeError(f'{stage} exited {code}')
    try:
        for rel,digest in read(ROOT/'package_hashes.json').items():assert sha(ROOT/rel)==digest,rel
        import torch
        assert torch.cuda.is_available(),'CUDA required; no silent CPU fallback'
        gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True,encoding='utf-8',check=True).stdout
        (ROOT/'gpu_start.txt').write_text(gpu,encoding='utf-8')
        write(ROOT/'runtime.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),scope='Local diagnostic timing; concurrent workloads not excluded'))
        old=read(ROOT/'protocol.json')['original_source']
        for label,path in [('original',old),('central',str(src(30)))]:
            child([sys.executable,'-B',str(ROOT/'probe_wct1_center_20261107.py'),path,str(ROOT/f'preflight_{label}.json')],f'probe_{label}')
        old_probe=read(ROOT/'preflight_original.json');new=read(ROOT/'preflight_central.json')
        assert new['records']==old_probe['records'],'Hotspots/initial users/exogenous trajectories changed'
        assert new['wct_positions']==[[250.,250.]] and new['state_dim']==133 and new['action_dim']==36
        assert sum(new['center_charging_w'])<=200+1e-8 and min(new['center_charging_w'])>0
        for k in ('K','M','state_dim','action_dim','F_UAV','BANDWIDTH','hot_prob','cold_prob'):assert new[k]==old_probe[k]
        write(ROOT/'preflight_verification.json',dict(verified_at=now(),external_trajectories_identical=True,slots_checked=150,initial_uav_identical=True,hotspot_centers=new['records'][0]['centers'],wct_positions=new['wct_positions'],center_shared_charging_w=new['center_charging_w'],only_learning_source_change='config.py N_WCT/WCT_POSITIONS',passed=True))
        for smoke in (True,False):
            stage='smoke' if smoke else 'training'
            cmd=[sys.executable,'-B','-u',str(src(30)/'train_unified.py'),'--algo','per_sac_cvx','--episodes','22' if smoke else '10000','--seed','42','--tag',tag(smoke),'--env-config',str(src(30)/'experiments/train_diverse.json'),'--warmup-steps','0' if smoke else '8000','--batch-size','512','--eval-episodes','2' if smoke else '50','--threads','2']
            child(cmd,stage,src(30));verify(smoke)
            evcmd=[sys.executable,'-B','-u',str(ROOT/'evaluate_wct1_center_20261107.py'),'--m','30','--algo','per_sac_cvx']
            if smoke:evcmd+=['--smoke']
            child(evcmd,'smoke_evaluation' if smoke else 'evaluation')
            dest=ROOT/('smoke_evaluation' if smoke else 'evaluation')/'seed42/m30/per_sac_cvx'
            assert read(dest/'manifest.json')['status']=='completed'
        state.update(status='completed',ended_at=now(),test_episodes=100);write(ROOT/'run_status.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),ended_at=now());write(ROOT/'run_status.json',state)
        (ROOT/'supervisor_exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise

if __name__=='__main__':run()
