"""Supervise the user-authorized fresh retry without changing the failed run."""
from pathlib import Path
import subprocess
import os
import sys
import json
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]/'results/reviewer_swap_retrain_seed42_retry_20261103'
if __name__=='__main__':
    sf=ROOT/'supervisor.json'
    with sf.open('x',encoding='utf-8') as f:
        json.dump({'status':'starting','pid':os.getpid(),'time':datetime.now().astimezone().isoformat()},f)
    assert not (ROOT/'run_status.json').exists(), 'Never duplicate the retry'
    env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (ROOT/'training.console.log').open('ab') as log,(ROOT/'training.errors.log').open('ab') as err:
        worker=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'source_snapshot/run_swap_training.py')],cwd=ROOT/'source_snapshot',env=env,stdout=log,stderr=err)
        sf.write_text(json.dumps({'status':'running','pid':os.getpid(),'worker_pid':worker.pid},indent=2)+'\n',encoding='utf-8')
        print('Swap retry worker PID '+str(worker.pid),flush=True)
        code=worker.wait()
    sf.write_text(json.dumps({'status':'completed' if code==0 else 'failed','pid':os.getpid(),'worker_pid':worker.pid,'exit_code':code},indent=2)+'\n',encoding='utf-8')
    if code:
        status=ROOT/'run_status.json'
        record=json.loads(status.read_text(encoding='utf-8')) if status.exists() else {}
        record.update(status='failed',exit_code=code,supervisor_detected_exit=True)
        status.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    sys.exit(code)
