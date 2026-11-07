"""Summarize completed paired evaluations without changing acceptance criteria."""
import json
import csv
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'results/reviewer_phase2_three_seeds_20261101/formal'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def main():
    batch = read(ROOT / 'batch_status.json')
    assert batch['status'] == 'completed' and batch['completed_jobs'] == 18
    per_seed, pooled = [], []
    for scenario in ('base', 'high_load', 'low_battery'):
        resources, episodes, associations = [], [], []
        for seed in (42, 43, 44):
            rpath = ROOT / 'resource' / f'seed{seed}_{scenario}'
            apath = ROOT / 'association' / f'seed{seed}_{scenario}'
            rr, aa = lines(rpath/'state_results.jsonl'), lines(apath/'state_results.jsonl')
            ee = read(apath/'summary.json')['episodes']
            assert [(r['test_seed'],r['slot']) for r in rr] == [(a['seed'],a['step']) for a in aa]
            valid = [r for r in rr if r['reference_valid']]
            row = dict(scenario=scenario, training_seed=seed, episodes=len(ee),
                       full_horizon_episodes=sum(e['survived_greedy_rollout'] for e in ee),
                       states=len(rr), valid_states=len(valid),
                       gap_mean_percent=float(np.mean([r['raw_gap_percent'] for r in valid])),
                       gap_p95_percent=float(np.percentile([r['raw_gap_percent'] for r in valid],95)),
                       total_slrr_gain=float(np.mean([r['total_slrr_gain'] for r in valid])),
                       mean_active_slrr_gain=float(np.mean([r['mean_active_slrr_gain'] for r in valid])))
            for method in ('matching','swap'):
                gains=[a['methods']['greedy']['J']-a['methods'][method]['J'] for a in aa]
                row[method+'_total_slrr_gain']=float(np.mean(gains))
                row[method+'_worse_states']=sum(x < -1e-8 for x in gains)
            per_seed.append(row)
            resources.extend(rr); episodes.extend(ee); associations.extend(aa)
        valid=[r for r in resources if r['reference_valid']]
        groups=[g for r in resources for g in r['conic_groups'] if not g.get('trivial')]
        row=dict(scenario=scenario, episodes=len(episodes),
                 full_horizon_episodes=sum(e['survived_greedy_rollout'] for e in episodes),
                 termination_reasons=dict(Counter(e['termination_reason'] for e in episodes)),
                 attempted_states=sum(e['executed_steps']+e['rejected_attempts'] for e in episodes),
                 executed_states=sum(e['executed_steps'] for e in episodes),
                 sampled_states=len(resources), valid_states=len(valid),
                 validity_percent=100*len(valid)/len(resources),
                 gap_pooled_mean_percent=float(np.mean([r['raw_gap_percent'] for r in valid])),
                 gap_pooled_p95_percent=float(np.percentile([r['raw_gap_percent'] for r in valid],95)),
                 solver_group_status_counts=dict(Counter(g['status'] for g in groups)),
                 maximum_relative_numerical_bound_gap=max(g['relative_certificate_gap'] for g in groups),
                 maximum_energy_excess_j=max(g['energy_excess_j'] for g in groups),
                 maximum_resource_share_excess=max(g['resource_capacity_excess'] for g in groups),
                 solver_disagreement_states=sum(abs(r['slsqp_minus_conic_j'])>1e-5*(1+abs(r['opt_j'])) for r in valid if r['slsqp_success']),
                 cf_allocation_ms_diagnostic=float(np.mean([r['cf_allocation_ms'] for r in valid])),
                 conic_allocation_ms_diagnostic=float(np.mean([r['conic_allocation_ms'] for r in valid])))
        seedrows=[r for r in per_seed if r['scenario']==scenario]
        for metric in ('gap_mean_percent','total_slrr_gain','mean_active_slrr_gain','matching_total_slrr_gain','swap_total_slrr_gain'):
            values=[r[metric] for r in seedrows]
            row[metric+'_three_seed_mean']=float(np.mean(values))
            row[metric+'_three_seed_sd']=float(np.std(values,ddof=1))
        pooled.append(row)
    report=dict(status='completed', training_seeds=[42,43,44], per_seed=per_seed, scenarios=pooled,
                scope='Frozen four-cluster diverse policies; paired instantaneous module evaluation. Neither alternative is written back into trajectories.',
                acceptance='Original strict acceptance unchanged: Solved plus physical feasibility and numerical primal-dual bound. AlmostSolved excluded, status counts retained.',
                statistics='Equal weight across the three training seeds for mean and sample SD; pooled state P95 is descriptive, not an independent-replicate confidence interval.',
                low_battery='20 kJ initial battery; conditional pre-termination state comparison. Report full-horizon completion separately; never interpret conditional gaps as survival robustness.',
                timing='Concurrent-training CPU diagnostic only; no onboard measurement or formal speedup claim.')
    (ROOT/'aggregate_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (ROOT/'per_seed_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_seed[0]));writer.writeheader();writer.writerows(per_seed)
    text=['Second experiment: frozen DARS three-seed paired evaluation',
          report['scope'],report['acceptance'],report['statistics'],report['low_battery'],report['timing'],'']
    for r in pooled:
        text.append(f"{r['scenario']}: {r['valid_states']}/{r['sampled_states']} reference-valid states; weighted-J gap (equal-seed mean) {r['gap_mean_percent_three_seed_mean']:.6f}%; pooled P95 {r['gap_pooled_p95_percent']:.6f}%; full horizon {r['full_horizon_episodes']}/{r['episodes']}.")
        text.append(f"Total SLRR gain from resource optimization {r['total_slrr_gain_three_seed_mean']:.6f}; matching {r['matching_total_slrr_gain_three_seed_mean']:.6f}; swap {r['swap_total_slrr_gain_three_seed_mean']:.6f}.")
    (ROOT/'RESULTS.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(pooled,indent=2))

if __name__=='__main__':
    main()
