"""Run only seed 42, sequentially, with original experiment settings."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
TZ = timezone(timedelta(hours=8))
def now():
    return datetime.now(TZ).isoformat()
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

tag = sys.argv[1]
out = ROOT / 'results' / tag
out.mkdir(parents=True, exist_ok=False)
status = dict(status='starting', tag=tag, started=now(), runner_pid=os.getpid(),
              python=sys.executable, platform=platform.platform(), cpu=platform.processor(),
              cpu_logical_count=os.cpu_count(), runs=[])
def save():
    tmp = out / 'batch_status.tmp'
    tmp.write_text(json.dumps(status, indent=2), encoding='utf-8')
    tmp.replace(out / 'batch_status.json')
save()
try:
    status['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    (out/'git_status.txt').write_text(subprocess.check_output(['git','status','--short','--branch'],text=True),encoding='utf-8')
    (out/'git_commit.txt').write_text(status['git_commit']+'\n',encoding='utf-8')
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    paths = list(ROOT.glob('*.py')) + [ROOT/'experiments/train_diverse.json']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    for path in paths:
        dest = snapshot / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    shutil.copy2(__file__, out/'launch_formal.py')
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    import numpy as np
    import torch
    import config as c
    from experiment_config import EnvironmentConfig
    from train_unified import ALGO_REGISTRY
    assert torch.cuda.is_available(), 'GPU selected but CUDA is unavailable'
    expected = dict(K=2,M=30,N_WCT=4,E_BATTERY_CAPACITY=40000,E_BATTERY_INIT=40000,
                    SLOT_NUM=50,T_FLY=1,T_COM=7,P_MAX_WCT=200,P_TH=200,L_C=100,
                    BATCH_SIZE=512,WARMUP_STEPS=8000)
    for key,value in expected.items():
        assert getattr(c,key)==value, (key,getattr(c,key),value)
    cfg = EnvironmentConfig.from_json(ROOT/'experiments/train_diverse.json')
    assert cfg == EnvironmentConfig(hotspot_mode='diverse')
    status.update(status='running', device=torch.cuda.get_device_name(0), torch=torch.__version__,numpy=np.__version__)
    for algo in ['per_sac_cvx','esac_per_cvx','sac_cvx','td3_cvx','maddpg_cvx']:
        assert algo in ALGO_REGISTRY
        assert all(digest(ROOT/path)==value for path,value in hashes.items()), 'Source/config changed during batch'
        command = [sys.executable,'-u','train_unified.py','--algo',algo,'--episodes','15000',
                   '--seed','42','--tag',tag,'--env-config','experiments/train_diverse.json','--eval-episodes','50']
        entry = dict(algorithm=algo,status='running',started=now(),command=command)
        status['runs'].append(entry)
        status['current_algorithm'] = algo
        save()
        started = time.perf_counter()
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        with (out/(algo+'.console.log')).open('wb') as log:
            proc = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            entry['pid'] = proc.pid
            save()
            entry['exit_code'] = proc.wait()
        entry.update(ended=now(),elapsed_seconds=time.perf_counter()-started)
        save()
        if entry['exit_code'] != 0:
            entry['status'] = 'failed'
            raise RuntimeError(f'{algo} exited with {entry["exit_code"]}; inspect console log')
        for suffix in ['.pth','.manifest.json','.train.jsonl','.validation.json','.npz']:
            assert (out/(algo+suffix)).stat().st_size > 0, (algo,suffix)
        manifest = json.loads((out/(algo+'.manifest.json')).read_text(encoding='utf-8'))
        assert manifest['status']=='completed' and manifest['training']['episodes']==15000
        episodes = executed = attempted = 0
        with (out/(algo+'.train.jsonl')).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                episodes += 1
                executed += row['executed_steps']
                attempted += row['attempted_steps']
        assert episodes==15000
        validation = json.loads((out/(algo+'.validation.json')).read_text(encoding='utf-8'))
        assert len(validation)==50
        entry.update(status='completed',episodes=episodes,executed_steps=executed,attempted_steps=attempted,
                     training_seconds=manifest['training_seconds'],gradient_updates=manifest.get('gradient_updates'),
                     checkpoint_sha256=digest(out/(algo+'.pth')),checkpoint_bytes=(out/(algo+'.pth')).stat().st_size,
                     validation_episodes=len(validation))
        if entry['gradient_updates'] is None:
            entry['gradient_update_note']='Original manifest does not expose this agent counter; no algorithm code was changed.'
        save()
    status.update(status='completed',ended=now(),current_algorithm=None)
    save()
except BaseException as exc:
    status.update(status='failed',ended=now(),error=str(exc))
    save()
    (out/'batch_error.log').write_text(traceback.format_exc(),encoding='utf-8')
    raise
