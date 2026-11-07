"""Sequential CPU-only paired association/resource audit. No training mutations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'experiments'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,value):
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    tmp.replace(p)


def run(out):
    if os.name=='nt':
        import ctypes
        k=ctypes.windll.kernel32
        k.GetCurrentProcess.restype=ctypes.c_void_p
        k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):
            raise OSError('Could not set below-normal priority')
    out.mkdir(parents=True,exist_ok=False)
    oldroot=ROOT/'results/original_main_seed42_gpu_retry_20261029_234000'
    queue=ROOT/'results/formal_all_five_seeds43_44_restart_20261031/source_snapshot'
    paths={42:oldroot,**{s:queue/f'results/formal_all_five_seeds43_44_session2_20261031_seed{s}' for s in (43,44)}}
    cfg=SCRIPTS/'train_diverse.json'
    selected={'base':(100,811000),'high_load':(50,821000),'low_battery':(50,841000)}
    files=['compare_association.py','association_benchmarks.py','compare_direct_objective.py',
           'conic_resource_reference.py','audit_resource_from_association.py','run_phase2_three_seeds.py','train_diverse.json']
    hashes={f:sha(SCRIPTS/f) for f in files}
    archive=out/'source_snapshot';archive.mkdir()
    for f in files: shutil.copy2(SCRIPTS/f,archive/f)
    models={}
    for seed,folder in paths.items():
        m=json.loads((folder/'per_sac_cvx.manifest.json').read_text())
        assert m['status']=='completed' and m['training_seed']==seed
        for key,value in json.loads(cfg.read_text()).items():
            assert m['environment'][key]==value, ('training environment mismatch',seed,key)
        for episodes,start in selected.values():
            used=set(m['training_episode_seeds'])|set(m['validation_seeds'])
            assert not used.intersection(range(start,start+episodes))
        models[seed]=dict(checkpoint=str(folder/'per_sac_cvx.pth'),sha256=sha(folder/'per_sac_cvx.pth'),
                          manifest=str(folder/'per_sac_cvx.manifest.json'),
                          source=str(oldroot/'source_snapshot' if seed==42 else queue))
    protocol=dict(created_epoch=time.time(),test_design='Three frozen trained DARS seeds, paired snapshots; only original Greedy DARS advances each environment.',
                  models=models,conditions=selected,total_episodes=600,planned_sampling_slots=[0,10,20,30,40],
                  inclusion='Executed attempted slots at fixed stride with active tasks; rejected attempts logged separately. No branch/outcome filtering.',
                  association_methods=['distance_greedy','adapted_deferred_acceptance','adapted_matching_plus_swaps'],
                  association_fixed='Positions/tasks/channels/raw Actor offload proposals/budgets; each candidate gets the same safety/resource routine.',
                  resource_fixed='Exact same sampled state; original Greedy association and its safety-corrected offload fixed; only CPU/BW change.',
                  reference='Clarabel SOCP with numerical primal-dual checks; independent SLSQP cross-check on every paired state.',
                  version='Existing four-cluster diverse training; not the proposed five-cluster/new-reward version.',
                  timing='Concurrent-training CPU diagnostic only; formal uncontended timing remains pending.',
                  source_hashes=hashes)
    write(out/'protocol.json',protocol)
    batch=dict(status='running',pid=os.getpid(),started_epoch=time.time(),jobs=[],current_job=None,completed_jobs=0,total_jobs=18)
    write(out/'batch_status.json',batch)
    try:
        for scenario,(episodes,start) in selected.items():
            for seed in (42,43,44):
                info=models[seed];tag=f'seed{seed}_{scenario}'
                association=out/'association'/tag; reference=out/'resource'/tag
                assoc_cmd=[sys.executable,'-B','-u',str(SCRIPTS/'compare_association.py'),
                           '--model',info['checkpoint'],'--model-manifest',info['manifest'],
                           '--source-dir',info['source'],'--env-config',str(cfg),
                           '--scenario',scenario,'--episodes',str(episodes),'--seed-start',str(start),
                           '--stride','10','--timing-repeats','1','--swap-evaluations','1000','--swap-rounds','20',
                           '--output-dir',str(association)]
                ref_cmd=[sys.executable,'-B','-u',str(SCRIPTS/'audit_resource_from_association.py'),
                         '--input',str(association),'--output',str(reference)]
                for kind,cmd,result in (('association',assoc_cmd,association),('resource',ref_cmd,reference)):
                    assert all(sha(SCRIPTS/f)==h for f,h in hashes.items()), 'Evaluation code changed during batch'
                    job=dict(name=f'{kind}_{tag}',kind=kind,seed=seed,scenario=scenario,
                             status='running',started_epoch=time.time(),output=str(result),command=cmd)
                    batch['jobs'].append(job);batch['current_job']=job['name']
                    write(out/'batch_status.json',batch)
                    print('Starting '+job['name'],flush=True)
                    with (out/(job['name']+'.log')).open('w',encoding='utf-8') as log:
                        process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                                 creationflags=0x08004000 if os.name=='nt' else 0)
                        job['pid']=process.pid;write(out/'batch_status.json',batch)
                        code=process.wait()
                    job.update(exit_code=code,ended_epoch=time.time(),status='completed' if code==0 else 'failed')
                    if code: raise RuntimeError('Failed '+job['name']+'; inspect its log')
                    batch['completed_jobs']+=1;write(out/'batch_status.json',batch)
                    print('Completed '+job['name'],flush=True)
        assert all(sha(info['checkpoint'])==info['sha256'] for info in models.values())
        batch.update(status='completed',current_job=None,ended_epoch=time.time())
        write(out/'batch_status.json',batch)
    except Exception:
        batch.update(status='failed',error=traceback.format_exc(),ended_epoch=time.time())
        write(out/'batch_status.json',batch)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    run(Path(p.parse_args().output).resolve())
