"""Isolated, explicitly authorized restart of failed TD3 seed44."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/formal_all_five_seeds43_44_restart_20261031/source_snapshot'
OUT=ROOT/'results/td3_seed44_retry_20261102'
TAG='formal_td3_seed44_retry_20261102'

def write(obj):
    p=OUT/'run_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(p)

def main(worker):
    if not worker:
        OUT.mkdir(exist_ok=False)
        src=OUT/'source_snapshot';src.mkdir();(src/'experiments').mkdir()
        for p in OLD.glob('*.py'):shutil.copy2(p,src/p.name)
        shutil.copy2(OLD/'experiments/train_diverse.json',src/'experiments/train_diverse.json')
        oldmanifest=OLD/'results/formal_all_five_seeds43_44_session2_20261031_seed44/td3_cvx.manifest.json'
        m=json.loads(oldmanifest.read_text(encoding='utf-8'))
        for name,h in m['source_sha256'].items():
            assert hashlib.sha256((src/name).read_bytes()).hexdigest()==h,name
        cmd=[sys.executable,'-B','-u',str(src/'train_unified.py'),'--algo','td3_cvx',
             '--episodes','15000','--seed','44','--tag',TAG,'--env-config',str(src/'experiments/train_diverse.json'),
             '--eval-episodes','50','--threads','2']
        protocol=dict(algorithm='td3_cvx',training_seed=44,command=cmd,source=str(src),
            old_failed_manifest=str(oldmanifest),old_failed_records=557,training=m['training'],
            environment=m['environment'],source_sha256=m['source_sha256'],
            execution_change='CPU intra-op threads 24 to 2 to share host with ongoing Swap; learning parameters unchanged.',
            restart='Fresh run from seed44: no recoverable weights, optimizer or replay checkpoint exists. Old failed files preserved.',
            authorized='User explicitly requested restarting TD3 in experiment thread on 2026-11-02.')
        (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        write(dict(status='starting',phase='training',target_episodes=15000,training_seed=44))
        si=subprocess.STARTUPINFO();si.dwFlags|=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
        with (OUT/'runner.log').open('ab') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,startupinfo=si,
                creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP,close_fds=True)
        print(json.dumps({'runner_pid':p.pid,'root':str(OUT)}));return
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    status=dict(status='running',phase='training_and_validation',training_seed=44,target_episodes=15000,
                runner_pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),source=protocol['source'])
    with (OUT/'training.log').open('ab') as out,(OUT/'training.errors.log').open('ab') as err:
        p=subprocess.Popen(protocol['command'],cwd=protocol['source'],env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL)
        status['training_pid']=p.pid;write(status);code=p.wait()
    status.update(status='completed' if code==0 else 'failed',exit_code=code,ended_at=datetime.now().astimezone().isoformat())
    result=Path(protocol['source'])/'results'/TAG
    status['result_directory']=str(result)
    if (result/'td3_cvx.train.jsonl').exists():status['episodes_completed']=len((result/'td3_cvx.train.jsonl').read_bytes().splitlines())
    write(status)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');main(parser.parse_args().worker)
