"""Frozen-source sequential CPU queue for the R3.9 terminal-energy reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/reviewer_r39_grid_terminal_20261104'
GROUPS=[(20,42),(25,42),(30,42),(30,43),(30,44)]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,x):
    temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temp.replace(p)


def plan(out,h3_limit,h5_limit):
    out.mkdir(parents=True,exist_ok=False)
    archive=out/'source_snapshot';archive.mkdir()
    for name in ['r39_grid_oracle.py','r39_grid_validate.py','r39_capture_windows.py','run_r39_grid_batch.py']:
        shutil.copy2(ROOT/'experiments'/name,archive/name)
    shutil.copy2(DATA/'theory_note.tex',archive/'theory_note.tex')
    inputs={};groups=[]
    for M,seed in GROUPS:
        directory=DATA/f'capture_m{M}_seed{seed}';cases=directory/'cases.jsonl'
        meta=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        assert meta['status']=='completed'
        rows=[json.loads(line) for line in cases.read_text(encoding='utf-8').splitlines()]
        for row in rows: row['_source']=str(cases.resolve())
        groups.append(rows)
        inputs[f'm{M}_s{seed}']=dict(cases=str(cases.resolve()),sha256=sha(cases),
             count=len(rows),low_battery=sum(c['low_battery'] for c in rows))
    jobs=[]
    def add(c,horizon,limit,weight):
        tag=c['case_id'].rsplit('_h',1)[0]+f'_h{horizon}'
        jobs.append(dict(name=tag,cases=c['_source'],case_id=c['case_id'],horizon=horizon,
            network_users=c['network_users'],training_seed=c['training_seed'],test_seed=c['test_seed'],
            start_slot=c['start_slot'],low_battery=c['low_battery'],sampling_weight=weight,time_limit=limit))
    # H3: all 250 prespecified windows; round-robin over available models.
    for idx in range(max(map(len,groups))):
        for rows in groups:
            if idx<len(rows): add(rows[idx],3,h3_limit,1.)
    # H5: every naturally low-battery window plus 26 ordinary windows.
    # Select ordinary windows by a fixed hash, before any H5 solver results.
    normal_selected=[]
    for rows in groups:
        low=[r for r in rows if r['low_battery']]
        normal=sorted([r for r in rows if not r['low_battery']],key=lambda r:hashlib.sha256(('r39_h5_v1:'+r['case_id']).encode()).hexdigest())
        normal_selected.append((normal,normal[:5]))
        for row in low:add(row,5,h5_limit,1.)
    leftovers=[r for allrows,selected in normal_selected for r in allrows[5:]]
    extra=min(leftovers,key=lambda r:hashlib.sha256(('r39_h5_extra:'+r['case_id']).encode()).hexdigest())
    for allrows,selected in normal_selected:
        if any(r['case_id']==extra['case_id'] for r in allrows): selected.append(extra)
        for row in selected:add(row,5,h5_limit,len(allrows)/len(selected))
    protocol=dict(status='planned',created_epoch=time.time(),inputs=inputs,jobs=jobs,
        total_jobs=len(jobs),h3_jobs=sum(j['horizon']==3 for j in jobs),h5_jobs=sum(j['horizon']==5 for j in jobs),
        gap_target_J=1e-3,spacing_m=25.,terminal='Each UAV physical terminal battery >= frozen DARS terminal battery in the same window.',
        population='K=2; all actual M20/25/30 users and active tasks retained; scenario unchanged.',
        models='M30 training seeds42/43/44; M20 and M25 have completed seed42 only. No new training.',
        sampling='H3 uses every captured window. H5 includes all low-battery starts (min battery/capacity<=0.2) and 26 hash-selected ordinary starts. Report strata separately; supplied weights reconstruct the captured-window population.',
        inference='Windows from one test episode are correlated. Use paired episode-level resampling; distinguish independent training seeds from test seeds.',
        bounds='Global bounds apply to the specified candidate graph with paired terminal floors; output physical feasible J and max(SCIP lower bound, independent-user analytic lower bound), retaining raw solver values.',
        source_hashes={p.name:sha(p) for p in archive.iterdir() if p.is_file()},
        solver_dependencies=str((ROOT/'experiments/r39_solver_deps').resolve()))
    write(out/'protocol.json',protocol)
    write(out/'batch_status.json',dict(status='planned',completed_jobs=0,total_jobs=len(jobs),jobs=[]))
    return protocol


def run(out):
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    previous=json.loads((out/'batch_status.json').read_text(encoding='utf-8'))
    if previous['status']!='planned':
        raise RuntimeError('This queue has already been started; inspect its recorded process/status instead of launching it again.')
    archive=out/'source_snapshot'
    for name,h in protocol['source_hashes'].items():assert sha(archive/name)==h
    for item in protocol['inputs'].values():assert sha(item['cases'])==item['sha256']
    env=os.environ.copy();env['R39_SOLVER_DEPS']=protocol['solver_dependencies'];env['CUDA_VISIBLE_DEVICES']='-1'
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]='1'
    status=dict(status='running',pid=os.getpid(),started_epoch=time.time(),completed_jobs=0,
         total_jobs=len(protocol['jobs']),valid_references=0,certified_references=0,failed_jobs=0,jobs=[])
    write(out/'batch_status.json',status)
    records=[]
    for spec in protocol['jobs']:
        destination=out/'runs'/spec['name'];destination.parent.mkdir(exist_ok=True)
        command=[sys.executable,'-B','-u',str(archive/'r39_grid_oracle.py'),'--cases',spec['cases'],
            '--case-id',spec['case_id'],'--horizon',str(spec['horizon']),'--spacing',str(protocol['spacing_m']),
            '--gap',str(protocol['gap_target_J']),'--time-limit',str(spec['time_limit']),'--output',str(destination)]
        record=dict(**spec,status='running',output=str(destination),started_epoch=time.time())
        status['current_job']=spec['name'];status['jobs'].append(record)
        with (out/(spec['name']+'.log')).open('w',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                 creationflags=0x08004000 if os.name=='nt' else 0)
            record['pid']=child.pid;write(out/'batch_status.json',status)
            code=child.wait()
        record.update(exit_code=code,ended_epoch=time.time())
        if code:
            record['status']='process_failed';status['failed_jobs']+=1
        else:
            r=json.loads((destination/'result.json').read_text(encoding='utf-8'))
            record.update(status='completed',solver_status=r['status'],valid_reference=r['valid_reference'],
                certified=r['certified_within_requested_gap'],dars_slrr=r['dars_total_slrr'],reference_slrr=r['oracle_total_slrr'],
                reference_upper=r['oracle_slrr_upper_bound'],verified_J_gap=r['verified_J_gap'],solve_seconds=r['solve_seconds'])
            status['valid_references']+=int(r['valid_reference'])
            status['certified_references']+=int(r['certified_within_requested_gap'])
        status['completed_jobs']+=1;records.append(dict(record));write(out/'batch_status.json',status)
        write(out/'completed_results.json',records)
        print(f"{status['completed_jobs']}/{status['total_jobs']} {spec['name']} {record['status']}",flush=True)
    status.update(status='completed',current_job=None,ended_epoch=time.time());write(out/'batch_status.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DATA/'formal_v1')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--run-existing',action='store_true')
    p.add_argument('--h3-limit',type=float,default=180.)
    p.add_argument('--h5-limit',type=float,default=300.)
    a=p.parse_args();a.output=a.output.resolve()
    if not a.run_existing:
        protocol=plan(a.output,a.h3_limit,a.h5_limit)
        print(json.dumps({k:protocol[k] for k in ['total_jobs','h3_jobs','h5_jobs','models']}),flush=True)
    if not a.prepare_only:run(a.output)


if __name__=='__main__':main()
