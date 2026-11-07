"""Submit only the next authorized task after success, respecting account queue limits."""
import json
import os
from pathlib import Path
import subprocess
from server_users_runner_20261105 import ROOT,read,write,now

def main():
    index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    protocol=read(ROOT/'protocol.json')
    job=protocol['jobs'][index]
    task=ROOT/f"tasks/{index:03d}_m{job['M']}_seed{job['seed']}_{job['algorithm']}"
    assert read(task/'run_status.json')['status']=='completed'
    if index==len(protocol['jobs'])-1:
        write(ROOT/'queue_status.json',dict(status='all_tasks_completed_aggregation_pending',completed=75,updated_at=now()))
        return
    lock=ROOT/f'chain_submission_{index:03d}'
    lock.mkdir(exist_ok=False)
    current=os.environ['SLURM_JOB_ID']
    command=['sbatch','--parsable',f'--array={index+1}',f'--dependency=afterok:{current}','--output=logs/%A_%a.out','--error=logs/%A_%a.err','submit.sbatch']
    write(lock/'intent.json',dict(command=command,created_at=now(),next_index=index+1))
    try:
        proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=60)
        write(lock/'receipt.json',dict(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,at=now()))
        if proc.returncode:raise RuntimeError('Next-task submission failed; no automatic retry')
        next_job=proc.stdout.strip().split(';')[0]
        assert next_job.isdigit(),proc.stdout
        write(ROOT/'queue_status.json',dict(status='next_task_submitted',last_completed_index=index,next_index=index+1,next_job_id=next_job,updated_at=now()))
    except BaseException as exc:
        write(ROOT/'queue_status.json',dict(status='submission_failed_or_unknown',last_completed_index=index,error=repr(exc),updated_at=now()))
        raise

if __name__=='__main__':main()
