"""Prepare the author-approved two-stage DARS single-central-WCT experiment.

Creates a new immutable source snapshot; refuses existing output directories.
Reuses the verified original supervisor/evaluator with explicit routing changes.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import hashlib
import shutil
import subprocess

CODE = Path(__file__).resolve().parents[1]
WORKSPACE = CODE.parent
BASE = WORKSPACE/'Dars-final/results/reviewer_wct1_center_dars_seed42_10000_20261107'
OUT = WORKSPACE/'Dars-final/results/reviewer_conditional_wct1_center_seed42_10000_20261107'
ALGO = 'per_sac_cvx_conditional'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)

def prepare():
    assert read(BASE/'run_status.json')['status'] == 'completed'
    assert read(BASE/'evaluation/seed42/m30/per_sac_cvx/manifest.json')['status'] == 'completed'
    OUT.mkdir(exist_ok=False)
    src=OUT/'source_snapshot';src.mkdir();(src/'experiments').mkdir()
    for p in CODE.glob('*.py'):shutil.copy2(p,src/p.name)
    # Exact same central-WCT config as the finished original comparison model.
    shutil.copy2(BASE/'source_snapshot/config.py',src/'config.py')
    shutil.copy2(BASE/'source_snapshot/experiments/train_diverse.json',src/'experiments/train_diverse.json')
    changes={p.name:{'original':sha(BASE/'source_snapshot'/p.name),'new':sha(p)}
             for p in src.glob('*.py') if (BASE/'source_snapshot'/p.name).exists() and sha(p)!=sha(BASE/'source_snapshot'/p.name)}
    assert set(changes)=={'evaluation.py','train_unified.py'},changes
    for name in ['UAV_env.py','DARS.py','config.py','experiment_config.py','state_normalization.py']:
        assert sha(src/name)==sha(BASE/'source_snapshot'/name)
    runner=(BASE/'run_wct1_center_20261107.py').read_text(encoding='utf-8')
    runner=runner.replace('per_sac_cvx',ALGO).replace('_wct1_center_dars_seed42','_conditional_wct1_center_seed42')
    runner=runner.replace('evaluate_wct1_center_20261107.py','evaluate_conditional_wct1_20261107.py')
    runner=runner.replace("only_learning_source_change='config.py N_WCT/WCT_POSITIONS'","only_learning_source_change='New association-conditioned decision structure; physical configuration identical to original central-WCT run'")
    runner=replace_once(runner,"assert meta['gradient_updates']>0", "assert meta['gradient_updates']>0\n    assert meta['architecture']=='dars_association_conditioned_two_stage_v1'")
    runner=replace_once(runner,"state.update(status='completed',ended_at=now(),test_episodes=100);write(ROOT/'run_status.json',state)","compare_baseline()\n        state.update(status='completed',ended_at=now(),test_episodes=100);write(ROOT/'run_status.json',state)")
    compare='''
def compare_baseline():
    old=Path(read(ROOT/'protocol.json')['baseline_root'])/'evaluation/seed42/m30/per_sac_cvx'
    new=ROOT/'evaluation/seed42/m30/per_sac_cvx_conditional'
    a=[json.loads(s) for s in (old/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    b=[json.loads(s) for s in (new/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(a)==len(b)==100
    assert all(x['seed']==y['seed'] and x['external_state_hashes']==y['external_state_hashes'] for x,y in zip(a,b))
    fields=['mean_total_slrr_fixed_horizon','cumulative_total_slrr','survived','executed_steps','energy_out_j','energy_in_j']
    metrics={k:dict(original_mean=sum(x[k] for x in a)/100,new_mean=sum(x[k] for x in b)/100,paired_mean_difference=sum(y[k]-x[k] for x,y in zip(a,b))/100) for k in fields}
    write(ROOT/'paired_comparison.json',dict(verified_at=now(),models=2,test_episodes_per_model=100,external_states_identical=True,training_seeds=[42],training_budget_each=10000,metrics=metrics,scope='Single trained seed per architecture; paired test differences conditional on these policies. No training-seed robustness or hardware timing equivalence claim. Delay means are conditional on execution; compare survival alongside delay.'))

'''
    runner=replace_once(runner,"if __name__=='__main__':run()",compare+"if __name__=='__main__':run()")
    (OUT/'run_conditional_wct1_20261107.py').write_text(runner,encoding='utf-8')
    ev=(BASE/'evaluate_wct1_center_20261107.py').read_text(encoding='utf-8')
    ev=ev.replace('from run_wct1_center_20261107 import','from run_conditional_wct1_20261107 import')
    ev=replace_once(ev,'env=UAVEnv(experiment_config=cfg,seed=seed,**kwargs);state=env.reset(seed=seed);configure_policy_environment(agent,env)',"env=getattr(agent,'environment_class',UAVEnv)(experiment_config=cfg,seed=seed,**kwargs);state=env.reset(seed=seed);configure_policy_environment(agent,env)")
    ev=replace_once(ev,'sync_device();policy_seconds=time.perf_counter()-start',"sync_device();policy_seconds=max(0.,time.perf_counter()-start-getattr(agent,'last_preparation_seconds',0.))")
    ev=replace_once(ev,"active=env.active_mask.copy();tasks=env.task_list.copy()","active=env.active_mask.copy();tasks=env.task_list.copy()\n                positions_before=env.loc_uav.copy();users_before=env.loc_ue.copy();phase=int(env._get_current_phase())")
    ev=replace_once(ev,"slots.append(dict(slot=slot,executed=executed,total_slrr=total,","slots.append(dict(slot=slot,executed=executed,total_slrr=total,uav_before=positions_before,uav_after=env.loc_uav.copy(),users_before=users_before,phase=phase,")
    (OUT/'evaluate_conditional_wct1_20261107.py').write_text(ev,encoding='utf-8')
    shutil.copy2(BASE/'probe_wct1_center_20261107.py',OUT/'probe_wct1_center_20261107.py')
    git_commit=subprocess.run(['git','-C',str(CODE),'rev-parse','HEAD'],capture_output=True,text=True,encoding='utf-8',check=True).stdout.strip()
    protocol=dict(created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        authorization='Author explicitly requested trying the new algorithm with two UAVs and one WCT, after reviewing original training and trajectories. Retain latest seed42/10000 settings.',
        algorithm=ALGO,architecture='dars_association_conditioned_two_stage_v1',training_seed=42,episodes=10000,warmup_steps=8000,batch_size=512,threads=2,
        K=2,M=30,wct_positions=[[250.,250.]],wct_indices=[0],wct_power_total_w=200.,
        original_source=str(BASE/'source_snapshot'),baseline_root=str(BASE),code_branch='feature/association-conditioned-offload',code_commit=git_commit,
        source_changes_vs_original=changes,added_learning_modules=['conditional_dars.py','sequential_environment.py'],
        reward='Unchanged original M*mean-active rho plus battery penalty; not manuscript sum-rho reward',
        physical_objective='Fixed50-slot total SLRR with zero future contribution after early termination',
        decision_structure='Movement -> existing energy safety -> Greedy association -> conditional offload -> unchanged KKT and energy safety -> one physical slot',
        learning='Separate movement/offload SAC actors, twin critics/targets, PER and temperatures; intra-slot discount1 and physical GAMMA. Inherited3-physical-slot return approximation.',
        unchanged='Exact original central-WCT config, physics, Greedy, square-root KKT, hotspots0.9/0.1, user RNG, K2 M30 CPU3GHz bandwidth1MHz 40kJ/UAV.',
        smoke=dict(episodes=22,validation=2,warmup_steps=0,batch_size=512,test_episodes=2,formal_evidence=False),validation_episodes=50,
        test=dict(episodes=100,seeds=list(range(931000,931100)),horizon=50,same_baseline_external_states=True),
        caveats=['One training seed per architecture, not three-seed robustness','More network parameters/replay and sequential computation; not parameter-matched ablation','Weight-only checkpoints every500; not optimizer/replay/RNG continuation','Timing diagnostic workstation only; concurrent workload not excluded','Server user-count remains paused; Swap remains cancelled; do not resume either'],
        source_sha256={str(p.relative_to(src)).replace('\\','/'):sha(p) for p in src.rglob('*') if p.is_file()})
    write(OUT/'protocol.json',protocol)
    write(OUT/'package_hashes.json',{str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file()})
    write(OUT/'run_status.json',dict(status='prepared',updated_at=protocol['created_at']))
    print(str(OUT))

if __name__=='__main__':prepare()
