import argparse
import csv
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
args = parser.parse_args()
out = args.directory.resolve()
status = json.loads((out/'batch_status.json').read_text(encoding='utf-8'))
assert status['status']=='completed'
protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
methods = protocol['methods']
per_seed = []; paired = []; checks = []; all_rows = []
for seed in [42,43,44]:
    folder=out/f'seed{seed}'
    rows=[json.loads(s) for s in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    steps=[json.loads(s) for s in (folder/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==3*len(protocol['test_seeds'])
    assert len({(r['test_seed'],r['method']) for r in rows})==len(rows)
    assert json.loads((folder/'progress.json').read_text(encoding='utf-8'))['model_unchanged']
    verification=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
    if 'executed_slots' in verification:
        # Original verification counter included a rejected last attempt.
        verification['attempted_slots']=verification.pop('executed_slots')
    checks.append(verification)
    all_rows.extend(rows)
    reference={r['test_seed']:r for r in rows if r['method']=='greedy'}
    for method in methods:
        rr=[r for r in rows if r['method']==method]
        ss=[s for s in steps if s['method']==method and s['executed']]
        values=np.array([r['mean_total_slrr_fixed_horizon'] for r in rr])
        runtime=np.array([s['association_allocation_ms'] for s in ss])
        record=dict(training_seed=seed,method=method,episodes=len(rr),survived_episodes=sum(r['survived'] for r in rr),
            survival_percent=100*np.mean([r['survived'] for r in rr]),
            mean_total_slrr_fixed_horizon=float(values.mean()),episode_total_slrr_sd=float(values.std(ddof=1)),
            mean_cumulative_total_slrr=float(np.mean([r['cumulative_total_slrr'] for r in rr])),
            mean_executed_slots=float(np.mean([r['executed_slots'] for r in rr])),
            mean_min_battery_kj=float(np.mean([r['min_battery_j'] for r in rr]))/1000,
            mean_comp_energy_j=float(np.mean([r['comp_energy_j'] for r in rr])),
            mean_total_energy_kj=float(np.mean([r['comp_energy_j']+r['flight_energy_j']+r['hover_energy_j'] for r in rr]))/1000,
            association_ms_mean=float(runtime.mean()),association_ms_p95=float(np.percentile(runtime,95)),
            prototype_decision_ms_mean=float(np.mean([s['prototype_decision_ms'] for s in steps if s['method']==method])),
            max_energy_violation_j=max(r['max_energy_violation_j'] for r in rr),
            max_adapter_error=max(r['max_adapter_error'] for r in rr),swap_budget_hits=sum(r['swap_budget_hits'] for r in rr))
        per_seed.append(record)
        if method!='greedy':
            differences=np.array([r['mean_total_slrr_fixed_horizon']-reference[r['test_seed']]['mean_total_slrr_fixed_horizon'] for r in rr])
            paired.append(dict(training_seed=seed,method=method,mean_paired_total_slrr_gain=float(differences.mean()),
                median_paired_total_slrr_gain=float(np.median(differences)),
                improved_episodes=int(np.sum(differences>1e-8)),worse_episodes=int(np.sum(differences < -1e-8)),
                unchanged_episodes=int(np.sum(np.abs(differences)<=1e-8)),
                survived_only_alternative=sum(r['survived'] and not reference[r['test_seed']]['survived'] for r in rr),
                survived_only_greedy=sum(not r['survived'] and reference[r['test_seed']]['survived'] for r in rr)))

aggregate=[]
keys=['mean_total_slrr_fixed_horizon','mean_cumulative_total_slrr','survival_percent','mean_executed_slots',
      'mean_min_battery_kj','mean_comp_energy_j','mean_total_energy_kj','association_ms_mean','association_ms_p95','prototype_decision_ms_mean']
baseline=np.mean([r['mean_total_slrr_fixed_horizon'] for r in per_seed if r['method']=='greedy'])
baseline_runtime=np.mean([r['association_ms_mean'] for r in per_seed if r['method']=='greedy'])
for method in methods:
    rr=[r for r in per_seed if r['method']==method]
    entry=dict(method=method,episodes=sum(r['episodes'] for r in rr),survived_episodes=sum(r['survived_episodes'] for r in rr))
    for key in keys:
        values=[r[key] for r in rr]
        entry[key]=dict(mean=float(np.mean(values)),seed_sd=float(np.std(values,ddof=1)))
    entry['relative_total_slrr_gain_percent']=100*(entry['mean_total_slrr_fixed_horizon']['mean']/baseline-1)
    entry['runtime_ratio_to_greedy']=entry['association_ms_mean']['mean']/baseline_runtime
    aggregate.append(entry)

for name,records in [('per_seed_summary.csv',per_seed),('paired_seed_summary.csv',paired)]:
    with (out/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
result=dict(status='completed',total_episodes=len(all_rows),aggregation='100 test episodes per method per trained seed; three training seeds equally weighted. SD is across training-seed means, not a confidence interval.',
    methods=aggregate,paired=paired,validation=dict(native_greedy_verifications=checks,max_energy_violation_j=max(r['max_energy_violation_j'] for r in per_seed),
        max_adapter_error=max(r['max_adapter_error'] for r in per_seed),all_models_unchanged=True),
    scope='Frozen Greedy-trained policies with closed-loop plug-in association replacements. No retraining. CPU runtime on evolving method-specific states; not an onboard measurement or isolated timing benchmark.',
    statistical_review=dict(coverage='11/11 statistical fallacy categories assessed',
        seed_aggregation='Equal training-seed weights; all per-seed results and paired episodes retained.',
        selection='All prespecified episodes retained including early termination. Conditional delay is secondary, not the headline objective.',
        inference='Descriptive evaluation of three fixed models and nominal scenarios; no significance testing or hardware/generalization claim.',
        intervention='Association alone changed in the simulator; this does not establish that higher battery or CPU individually caused any outcome.'))
(out/'aggregate_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
report=['Frozen-policy association replacement: nominal, 3 training seeds x 100 test seeds x 3 methods.',
        'Primary metric: cumulative sum_i rho_i / fixed 50-slot horizon; zero after rejection/termination.',
        'Method | mean total SLRR/slot | across-seed SD | relative gain % | full-horizon survival | association ms | P95 ms (mean of seed P95s)']
for r in aggregate:
    report.append(f"{r['method']} | {r['mean_total_slrr_fixed_horizon']['mean']:.6f} | {r['mean_total_slrr_fixed_horizon']['seed_sd']:.6f} | {r['relative_total_slrr_gain_percent']:+.3f} | {r['survived_episodes']}/{r['episodes']} | {r['association_ms_mean']['mean']:.4f} | {r['association_ms_p95']['mean']:.4f}")
report.extend(['',result['scope'],'All methods include the same adapter and final environment allocation pass; the association-module timing excludes Actor inference.',
               'Prototype decision timing includes Actor, movement safety and common final allocation pass; it is not optimized deployment timing.'])
(out/'RESULTS.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
print('\n'.join(report))
print(json.dumps(result['validation'],indent=2))
