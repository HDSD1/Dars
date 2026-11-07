"""Package author-approved 5 x 5 x 3, fresh 10000-episode server experiments."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import tarfile

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'results/reviewer_users_m15_m35_three_seeds_10000_server_20261105'
SOURCE = BASE / 'results/reviewer_users_m20_m25_m30_seed42_5000_20261102/m20/source_snapshot'
ALGOS = ['per_sac_cvx','esac_per_cvx','sac_cvx','td3_cvx','maddpg_cvx']
REMOTE = '/data/run01/scze797/reviewer_users_m15_m35_three_seeds_10000_server_20261105'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, data): p.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=False)
    bundle = OUT / 'bundle'
    bundle.mkdir()
    originals = {p.name:sha(p) for p in SOURCE.glob('*.py')}
    sources = {}
    for m in [15,20,25,30,35]:
        target = bundle / f'sources/m{m}'
        (target/'experiments').mkdir(parents=True)
        for p in SOURCE.glob('*.py'): shutil.copy2(p,target/p.name)
        shutil.copy2(SOURCE/'experiments/train_diverse.json',target/'experiments/train_diverse.json')
        p = target/'config.py'
        if m != 20:
            text, n = re.subn(r'^M\s*=\s*20(?=\s*(?:#|$))',f'M = {m}',p.read_text(encoding='utf-8'),flags=re.M)
            assert n == 1
            p.write_text(text,encoding='utf-8')
        hashes={p.name:sha(p) for p in target.glob('*.py')}
        assert [n for n in hashes if hashes[n]!=originals[n]] == ([] if m==20 else ['config.py'])
        sources[str(m)] = dict(files=hashes, environment_sha256=sha(target/'experiments/train_diverse.json'))
    jobs=[]
    # Cover each count at seed42 before seeds43/44; no outcomes determine order.
    for seed in [42,43,44]:
        for m in [15,20,25,30,35]:
            for algo in ALGOS:
                jobs.append(dict(index=len(jobs),M=m,seed=seed,algorithm=algo,episodes=10000))
    protocol=dict(authorization='Author requested M15/35 and seeds43/44 for M15/20/25/30/35, then explicitly changed all training budgets to10000. All75 models are fresh; no old model is relabeled or overwritten.',
        remote_root=REMOTE, jobs=jobs, algorithms=ALGOS, user_counts=[15,20,25,30,35],training_seeds=[42,43,44],
        training=dict(episodes=10000,warmup_steps=8000,batch_size=512,threads=2,validation_episodes=50),
        original_source=str(SOURCE),source_sha256=sources,
        learning_changes='Only config.M relative to instrumented M20 source; original formula-derived dimensions and M reward scaling. No CPU/bandwidth scaling. K2,4WCT,hot0.9/cold0.1,Greedy,networks,optimizer,physics,safety unchanged.',
        reward='Original M * mean-active rho plus battery penalty; distinct from physical total SLRR.',
        evaluation=dict(episodes_per_model=100,seed_start=931000,horizon=50,primary_metric='mean_total_slrr_fixed_horizon',early_termination='zero future SLRR; actual executed energy retained',paired='same-M external trajectories independently checked; final cross-model hashes must agree',timing='Server synchronized end-to-end decision diagnostic; record node/GPU and contention. Do not combine as matched hardware with old workstation timings.'),
        statistics='First compute each trained seed mean over100 test episodes, then3-seed mean and sample SD. Episode bootstrap intervals are conditional, not training-seed uncertainty.',
        execution='One GPU maximum, Slurm serial array. First task gates remaining tasks via afterok. Every task smoke22+2 before formal10000+50 and100 frozen tests. No auto retry; failed originals retained; no optimizer/replay/RNG continuation claimed.',
        exclusions='No main manuscript/Bib/referenced figures altered; local hotspot and M6 processes untouched; Swap remains cancelled.')
    save(bundle/'protocol.json',protocol)
    for name in ['server_users_runner_20261105.py']:
        shutil.copy2(BASE/'experiments'/name,bundle/name)
    # Adapt only evaluator orchestration/metadata; physical rollout calculations unchanged.
    text=(BASE/'experiments/evaluate_user_sweep_5000_20261102.py').read_text(encoding='utf-8')
    text=text.replace('from run_user_sweep_5000_20261102 import OUT, ROOT, ALGOS, src, result, read, sha, now',
        'from server_users_runner_20261105 import OUT, ROOT, ALGOS, src, result, read, sha, now\nTRAIN_SEED = int(os.environ["IOTJ_TRAIN_SEED"])')
    start=text.index('    if not smoke:\n')
    end=text.index("    for n in ('OMP_NUM_THREADS'",start)
    text=text[:start]+text[end:]
    text=text.replace('training_seed=42','training_seed=TRAIN_SEED')
    text=text.replace("/f'm{m}'/algo", "/f'seed{TRAIN_SEED}'/f'm{m}'/algo")
    text=text.replace("hardware_scope='Local workstation, not airborne measurements. Known experimental training queue, K-scaling jobs and authorized Swap retry idle for formal evaluation; unrelated workload still requires monitoring.'", "hardware_scope='Slurm server GPU; own training ended before this evaluation. Other-node processes not excluded. Diagnostic timings; not airborne or matched to workstation. Hardware recorded in task runtime.'")
    text=text.replace("statistics='Single training seed42. CIs across independent test episodes conditional on this policy; no training-seed uncertainty.'", "statistics='One of the registered training seeds42/43/44; per-model episode bootstrap CI remains conditional. Aggregate3 training-seed means and sample SD separately.'")
    start=text.index('    # Same user/task trajectories')
    end=text.index('    assert before==model_digest(agent)',start)
    text=text[:start]+'    # Every rollout was compared to an independently generated external trajectory.\n'+text[end:]
    (bundle/'evaluate_server_users.py').write_text(text,encoding='utf-8')
    submit='''#!/bin/bash
#SBATCH --job-name=iotj_users_10000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
cd REMOTE_ROOT
source /etc/profile.d/modules.sh
module load dp_lammps_pytorch/3.1.1_29Aug2024
export PYTHONPATH=/data/run01/scze797/uav_ddpg_revision_20261021/numpy126_overlay
export PYTHONNOUSERSITE=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MPLBACKEND=Agg
python -B -u server_users_runner_20261105.py --index "${SLURM_ARRAY_TASK_ID:-0}"
'''.replace('REMOTE_ROOT',REMOTE)
    (bundle/'submit.sbatch').write_text(submit,encoding='utf-8',newline='\n')
    hashes={str(p.relative_to(bundle)).replace('\\','/'):sha(p) for p in bundle.rglob('*') if p.is_file()}
    save(bundle/'package_hashes.json',hashes)
    with tarfile.open(OUT/'bundle.tar.gz','w:gz') as archive:
        for p in bundle.rglob('*'):
            if p.is_file():archive.add(p,arcname=str(p.relative_to(bundle)),recursive=False)
    save(OUT/'preparation.json',dict(status='prepared_not_submitted',jobs=len(jobs),remote_root=REMOTE,archive_sha256=sha(OUT/'bundle.tar.gz'),package_files=len(hashes)))
    print(json.dumps(dict(root=str(OUT),jobs=len(jobs),archive_bytes=(OUT/'bundle.tar.gz').stat().st_size),ensure_ascii=True))

if __name__=='__main__': main()
