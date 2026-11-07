"""Equal-placement/equal-training-seed aggregation of WCT stress outcomes."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main(root):
    batch=read(root/'batch_status.json');assert batch['status']=='completed'
    protocol=read(root/'protocol.json');seeds=protocol['test_seeds']
    per_seed=[];by_subset=[];all_rows={};curves=[]
    for model in (42,43,44):
        for case in protocol['cases']:
            folder=root/f'seed{model}'/case['name'];assert read(folder/'manifest.json')['status']=='completed'
            rows=[json.loads(x) for x in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
            assert [r['seed'] for r in rows]==seeds
            for r in rows:
                assert abs(sum(t['total_slrr'] for t in r['trace'])-r['cumulative_total_slrr'])<1e-8
                assert abs(r['cumulative_total_slrr']/r['horizon']-r['mean_total_slrr_fixed_horizon'])<1e-9
                assert r['survived']==(r['executed_steps']==r['horizon'])
                if case['active_wct_count']==0:assert r['energy_in_j']==0
            all_rows[(model,case['name'])]=rows
            by_subset.append(dict(training_seed=model,case=case['name'],active_wct_count=case['active_wct_count'],
                episodes=len(rows),survived=sum(r['survived'] for r in rows),
                survival_probability=float(np.mean([r['survived'] for r in rows])),
                mean_total_slrr_fixed_horizon=float(np.mean([r['mean_total_slrr_fixed_horizon'] for r in rows])),
                mean_min_battery_j=float(np.mean([r['min_battery_j'] for r in rows])),
                mean_executed_steps=float(np.mean([r['executed_steps'] for r in rows]))))
        for count in (4,2,1,0):
            cases=[c for c in protocol['cases'] if c['active_wct_count']==count]
            rows=[r for c in cases for r in all_rows[(model,c['name'])]]
            r=dict(training_seed=model,active_wct_count=count,placements=len(cases),episodes=len(rows),
                survived=sum(x['survived'] for x in rows),
                survival_probability=float(np.mean([x['survived'] for x in rows])),
                mean_total_slrr_fixed_horizon=float(np.mean([x['mean_total_slrr_fixed_horizon'] for x in rows])),
                mean_cumulative_total_slrr=float(np.mean([x['cumulative_total_slrr'] for x in rows])),
                mean_min_battery_j=float(np.mean([x['min_battery_j'] for x in rows])),
                worst_min_battery_j=min(x['min_battery_j'] for x in rows),
                mean_executed_steps=float(np.mean([x['executed_steps'] for x in rows])),
                mean_energy_in_j=float(np.mean([x['energy_in_j'] for x in rows])),
                max_energy_violation_j=max(x['max_energy_violation_j'] for x in rows),
                energy_violation_steps=sum(x['energy_violation_steps'] for x in rows),
                termination_reasons=dict(Counter(x['termination_reason'] for x in rows)))
            per_seed.append(r)
            for t in range(51):
                completed=[x for x in rows if x['executed_steps']>=t]
                batteries=[min(x['initial_battery_j']) if t==0 else min(x['trace'][t-1]['battery_after_service']) for x in completed]
                # Only actual continuing trajectories contribute battery values;
                # terminated episodes contribute zero to service, not fake battery measurements.
                slrr=[x['trace'][t-1]['total_slrr'] if t>0 and x['executed_steps']>=t else 0. for x in rows]
                curves.append(dict(training_seed=model,active_wct_count=count,completed_slots=t,
                    survival_fraction=len(completed)/len(rows),observed_battery_episodes=len(batteries),
                    conditional_mean_min_uav_battery_j=float(np.mean(batteries)) if batteries else None,
                    mean_slot_total_slrr_zero_after_termination=float(np.mean(slrr)) if t else None))
    aggregate=[]
    for count in (4,2,1,0):
        group=[r for r in per_seed if r['active_wct_count']==count]
        r=dict(active_wct_count=count,training_seeds=[42,43,44],placements_per_model=group[0]['placements'],
            episodes=sum(x['episodes'] for x in group),survived=sum(x['survived'] for x in group),
            max_energy_violation_j=max(x['max_energy_violation_j'] for x in group),
            energy_violation_steps=sum(x['energy_violation_steps'] for x in group))
        for key in ('survival_probability','mean_total_slrr_fixed_horizon','mean_cumulative_total_slrr',
                    'mean_min_battery_j','mean_executed_steps','mean_energy_in_j'):
            v=np.array([x[key] for x in group]);r[key]=dict(mean=float(v.mean()),between_seed_sd=float(v.std(ddof=1)))
        aggregate.append(r)
    summary=dict(status='completed',total_episodes=sum(r['episodes'] for r in per_seed),by_wct_count=aggregate,
        statistics='Equal test episodes and equal placements within training seed; mean and sample SD across three training seeds. Counts are descriptive, repeated placements are not independent trained-policy replicates.',
        zero_wct=protocol['zero_wct'],per_training_seed=per_seed,
        scope='Frozen four-cluster DARS policy and original safety layer. Full-horizon termination retained; no retraining, no scenario exclusion.',
        battery_curve='Conditional on completing the slot; sample count and survival fraction accompany each point. No fabricated post-termination battery.',
        time='Recorded inference/scheduling times are diagnostic, not an uncontended benchmark.')
    (root/'aggregate_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    for name,rows in [('per_training_seed.csv',per_seed),('per_subset.csv',by_subset),('time_curves.csv',curves)]:
        with (root/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=['Frozen DARS WCT-availability stress evaluation',summary['statistics'],summary['zero_wct'],'']
    for r in aggregate:
        report.append(f"Active WCTs={r['active_wct_count']}: survival={r['survival_probability']['mean']:.4%}; full-horizon mean total SLRR={r['mean_total_slrr_fixed_horizon']['mean']:.6f}; mean completed slots={r['mean_executed_steps']['mean']:.3f}; mean minimum battery={r['mean_min_battery_j']['mean']:.3f} J; maximum executed-action energy violation={r['max_energy_violation_j']:.3g} J.")
    (root/'RESULTS.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
