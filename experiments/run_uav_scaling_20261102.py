"""Authorized K=3/4 DARS seed42 runs, isolated from published/older sources."""
import argparse
from datetime import datetime
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'results/formal_all_five_seeds43_44_restart_20261031/source_snapshot'
OUT = ROOT / 'results/reviewer_uav_scaling_k3_k4_seed42_20261102'

def now(): return datetime.now().astimezone().isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, obj):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)
def edit(p, old, new):
    s = p.read_text(encoding='utf-8')
    assert s.count(old) == 1, (p.name, old)
    p.write_text(s.replace(old, new), encoding='utf-8')
def folder(k): return OUT / f'k{k}'
def source(k): return folder(k) / 'source_snapshot'
def tag(k, smoke=False): return f'{"smoke" if smoke else "formal"}_dars_k{k}_seed42'
def command(k, smoke=False):
    src = source(k)
    return [sys.executable, '-B', '-u', str(src/'train_unified.py'), '--algo', 'per_sac_cvx',
            '--episodes', '3' if smoke else '15000', '--seed', '42', '--tag', tag(k, smoke),
            '--env-config', str(src/'experiments/train_diverse.json'), '--eval-episodes', '2' if smoke else '50',
            '--warmup-steps', '0' if smoke else '8000', '--batch-size', '16' if smoke else '512', '--threads', '2']

def prepare():
    OUT.mkdir(exist_ok=False)
    for k in (3, 4):
        src = source(k); src.mkdir(parents=True); (src/'experiments').mkdir()
        for p in OLD.glob('*.py'): shutil.copy2(p, src/p.name)
        shutil.copy2(OLD/'experiments/train_diverse.json', src/'experiments/train_diverse.json')
        p = src/'config.py'; s = p.read_text(encoding='utf-8')
        s, n = re.subn(r'^K = 2(?=\s+#)', f'K = {k}', s, flags=re.M); assert n == 1
        p.write_text(s, encoding='utf-8')
        p = src/'evaluation.py'
        edit(p, '        self.reward = 0.0', '        self.reward = 0.0\n        self.total_slrr = 0.0')
        edit(p, "            self.rhos.append(float(info['rho_mean']))", "            self.rhos.append(float(info['rho_mean']))\n            self.total_slrr += float(np.sum(info['rho']))")
        edit(p, "                   termination_reason=self.reason, reward=self.reward,", "                   termination_reason=self.reason, reward=self.reward,\n                   cumulative_total_slrr=self.total_slrr,\n                   mean_total_slrr_fixed_horizon=self.total_slrr / self.horizon,")
        edit(p, "    for key in ('executed_steps', 'min_battery_j', 'slrr_executed', 'slrr_fixed_horizon',", "    for key in ('cumulative_total_slrr', 'mean_total_slrr_fixed_horizon', 'executed_steps', 'min_battery_j', 'slrr_executed', 'slrr_fixed_horizon',")
        p = src/'train_unified.py'
        edit(p, '        ep_rewards.append(ep_reward)', "        if needs_training and (ep + 1) % 500 == 0:\n            agent.save(str(result_dir / f'{algo_name}.episode{ep+1}.pth'))\n        ep_rewards.append(ep_reward)")
        changes = {p.name: {'original':sha(OLD/p.name), 'new':sha(p)} for p in src.glob('*.py') if sha(p)!=sha(OLD/p.name)}
        assert set(changes) == {'config.py', 'evaluation.py', 'train_unified.py'}
        write(folder(k)/'protocol.json', dict(created_at=now(), authorized='Author selected simultaneous K=3 and K=4 on 2026-11-02.',
            K=k, M=30, training_seed=42, episodes=15000, warmup_steps=8000, batch_size=512, threads=2,
            state_dim=6*k+121, action_dim=3*k+30, training_wct_indices=[0,1,2,3],
            command=command(k), original_source=str(OLD), source=str(src), source_changes=changes,
            source_sha256={p.name:sha(p) for p in src.glob('*.py')},
            learning_definition='Original Greedy DARS, four-cluster diverse hotspots. Only K changes; existing dimension-dependent network/entropy scales automatically. Original reward M*mean-active rho plus original battery penalty unchanged.',
            logging_changes='Log physical sum(rho) over executed slots and fixed-50 mean; zero future contribution after termination. Save weights every500 episodes; checkpoints do NOT contain optimizer/replay/RNG and are not exact-resume snapshots.',
            test=dict(seeds=list(range(921000,921100)), wct_counts=[4,2,1], subsets='all 6 pairs and all 4 single stations',
                      scope='Frozen policy trained with4 WCT. Within each K identical seed schedule; cross-K user positions are distribution matched, not identical due to original init RNG consumption. Does not prove retrained sparse-WCT performance.',
                      timing='Diagnostic workstation timing; other concurrent work may be active. Not formal uncontended/airborne timing.'),
            evidence='One training seed per K, not three-seed robustness. Added UAVs also add compute, bandwidth and batteries; not a constant-total-resource comparison.'))
        write(folder(k)/'run_status.json', dict(status='prepared', K=k, training_seed=42, updated_at=now()))
    write(OUT/'protocol.json', dict(created_at=now(), Ks=[3,4], training_seeds=[42], model='DARS Greedy',
          reviewer_mapping={'R3.7':'K2 existing plus new K3/K4', 'R1.6':'Fixed K3/K4 policies tested with4/2/1 active WCT'},
          energy_bound='K4/1WCT: maximum initial+harvest230000J < hover-only235886J over50 slots. Full survival physically impossible under current parameters.'))

def verify(k, smoke=False):
    result = source(k)/'results'/tag(k, smoke)
    rows = [json.loads(s) for s in (result/'per_sac_cvx.train.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest = json.loads((result/'per_sac_cvx.manifest.json').read_text(encoding='utf-8'))
    validation = json.loads((result/'per_sac_cvx.validation.json').read_text(encoding='utf-8'))
    assert len(rows)==(3 if smoke else 15000) and len(validation)==(2 if smoke else 50)
    assert manifest['status']=='completed' and manifest['system_parameters']['K']==k
    assert manifest['runtime']['device'] != 'CPU', manifest['runtime']
    for r in rows+validation:
        assert len(r['final_battery_j'])==k
        assert r['energy_violation_steps']==0
        assert abs(r['mean_total_slrr_fixed_horizon']*50-r['cumulative_total_slrr'])<1e-8
    import torch
    weights = torch.load(result/'per_sac_cvx.pth', map_location='cpu', weights_only=False)
    count=0
    for key in ('actor','critic','critic_target'):
        for tensor in weights[key].values(): assert torch.isfinite(tensor).all(); count+=1
    record=dict(checked_at=now(),K=k,training_rows=len(rows),validation_rows=len(validation),
                checkpoint_sha256=sha(result/'per_sac_cvx.pth'),finite_tensors=count,
                runtime=manifest['runtime'],training=manifest['training'])
    write(folder(k)/('smoke_verification.json' if smoke else 'completion_verification.json'),record)
    return record

def evaluate(k):
    sys.path.insert(0,str(source(k)))
    import torch
    torch.set_num_threads(2)
    from dataclasses import replace
    from DARS import PERSACAgent
    from experiment_config import EnvironmentConfig
    from evaluation import evaluate_policy, summarize, write_json, model_digest
    model=source(k)/'results'/tag(k)/'per_sac_cvx.pth'
    manifest=json.loads(model.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    cfg=EnvironmentConfig(**manifest['environment']); agent=PERSACAgent();agent.load(str(model)); digest=model_digest(agent)
    dest=folder(k)/'test';dest.mkdir(exist_ok=False)
    cases=[tuple(range(4))]+list(combinations(range(4),2))+list(combinations(range(4),1))
    summaries=[]
    for indices in cases:
        name='wct_'+''.join(map(str,indices))
        write(dest/'progress.json',dict(status='running',case=name,cases_completed=len(summaries),total_cases=len(cases),updated_at=now()))
        rows=evaluate_policy(agent,replace(cfg,wct_indices=indices),manifest['env_kwargs'],range(921000,921100),keep_trace=True)
        assert all(r['energy_violation_steps']==0 for r in rows)
        write_json(dest/f'{name}.episodes.json',rows)
        summaries.append(dict(case=name,active_wct_count=len(indices),wct_indices=indices,summary=summarize(rows)))
        write(dest/'case_summaries.json',summaries)
    assert model_digest(agent)==digest
    write(dest/'progress.json',dict(status='completed',cases_completed=len(cases),episodes=1100,updated_at=now()))

def worker(k, smoke=False):
    dest=folder(k)
    statusfile=dest/('smoke_status.json' if smoke else 'run_status.json')
    existing=json.loads(statusfile.read_text(encoding='utf-8')) if statusfile.exists() else {}
    assert existing.get('status') in (None,'prepared'), 'Never restart an existing worker/run automatically'
    status=dict(status='running',phase='smoke' if smoke else 'training_and_validation',K=k,training_seed=42,
                runner_pid=os.getpid(),started_at=now(),target_episodes=3 if smoke else 15000)
    env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    prefix='smoke' if smoke else 'training'
    try:
        with (dest/f'{prefix}.log').open('ab') as log,(dest/f'{prefix}.errors.log').open('ab') as err:
            p=subprocess.Popen(command(k,smoke),cwd=source(k),env=env,stdout=log,stderr=err,stdin=subprocess.DEVNULL)
            status['training_pid']=p.pid;write(statusfile,status)
            code=p.wait()
        if code: raise RuntimeError(f'Training exited {code}')
        verify(k,smoke)
        if not smoke:
            status.update(phase='frozen_wct_evaluation',training_episodes_completed=15000,updated_at=now());write(statusfile,status)
            evaluate(k)
        status.update(status='completed',phase='completed',ended_at=now());write(statusfile,status)
    except Exception as exc:
        status.update(status='failed',error=repr(exc),ended_at=now());write(statusfile,status);raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--k',type=int,choices=[3,4]);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if a.prepare: prepare()
    else: worker(a.k,a.smoke)
