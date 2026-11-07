import argparse
import csv
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();out=args.directory
assert json.loads((out/'batch_status.json').read_text(encoding='utf-8'))['status']=='completed'
per_seed=[];rows=[]
for seed in [42,43,44]:
    rr=[json.loads(s) for s in (out/f'seed{seed}/states.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rr)==500;rows.extend(rr)
    for method in ['greedy','matching','swap','exact']:
        value=np.array([r[f'{method}_total_slrr'] for r in rr]);greedy=np.array([r['greedy_total_slrr'] for r in rr])
        exact=np.array([r['exact_total_slrr'] for r in rr])
        per_seed.append(dict(training_seed=seed,method=method,states=len(rr),mean_total_slrr=float(value.mean()),
            mean_gain_over_greedy=float((value-greedy).mean()),mean_shortfall_to_exact=float((exact-value).mean()),
            states_at_exact_value=int(np.sum(np.abs(exact-value)<=1e-8))))
with (out/'per_seed_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(per_seed[0]));w.writeheader();w.writerows(per_seed)
base=np.mean([r['mean_total_slrr'] for r in per_seed if r['method']=='greedy'])
aggregate=[]
for method in ['greedy','matching','swap','exact']:
    rr=[r for r in per_seed if r['method']==method]
    value=np.mean([r['mean_total_slrr'] for r in rr])
    aggregate.append(dict(method=method,mean_total_slrr=float(value),seed_sd=float(np.std([r['mean_total_slrr'] for r in rr],ddof=1)),
        mean_gain_over_greedy=float(np.mean([r['mean_gain_over_greedy'] for r in rr])),relative_gain_percent=float(100*(value/base-1)),
        states_at_exact_value=sum(r['states_at_exact_value'] for r in rr)))
report=dict(status='completed',states=len(rows),methods=aggregate,
    total_per_uav_subsets_evaluated=sum(r['counts']['per_uav_subsets_evaluated'] for r in rows),
    total_joint_mappings_represented=sum(r['counts']['coverage_capacity_feasible_joint_maps'] for r in rows),
    max_joint_mappings_in_one_state=max(r['counts']['coverage_capacity_feasible_joint_maps'] for r in rows),
    brute_force_validation_states=sum(r['bruteforce_checked'] for r in rows),
    improved_states=sum(r['exact_gain']>1e-8 for r in rows),
    changed_users_mean=float(np.mean([r['changed_associations'] for r in rows])),
    changed_users_max=max(r['changed_associations'] for r in rows),
    states_with_two_or_more_changed_users=sum(r['changed_associations']>=2 for r in rows),
    total_new_associations=sum(r['newly_associated'] for r in rows),total_returns_to_local=sum(r['returned_to_local'] for r in rows),
    mean_search_ms=float(np.mean([r['search_ms'] for r in rows])),p95_search_ms=float(np.percentile([r['search_ms'] for r in rows],95)),
    max_energy_excess_j=max(r['energy_excess_j'] for r in rows),
    scope='All associations optimized simultaneously at fixed sampled states, including local execution. Exact only under the existing closed-form resource/safety mapping and fixed positions/raw offload proposals. Not joint P1 optimality; no trajectory changes or retraining.',
    aggregation='Equal weights across three training-seed means; 500 snapshots each. Exact-value counts use absolute total-SLRR tolerance 1e-8.',
    example_first_state=rows[0])
(out/'aggregate_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
lines=['Exact association comparison: all 1500 nominal snapshots.',
       'Method | mean per-slot total SLRR | absolute gain over Greedy | relative gain % | states matching exact value']
for r in aggregate:
    lines.append(f"{r['method']} | {r['mean_total_slrr']:.6f} | {r['mean_gain_over_greedy']:+.6f} | {r['relative_gain_percent']:+.3f} | {r['states_at_exact_value']}/1500")
lines.extend(['',report['scope'],f"Subset evaluations: {report['total_per_uav_subsets_evaluated']}; joint assignments represented: {report['total_joint_mappings_represented']}; independent brute-force checks: {report['brute_force_validation_states']}.",
    f"Exact search mean runtime: {report['mean_search_ms']:.3f} ms; P95: {report['p95_search_ms']:.3f} ms (offline CPU, no onboard claim)."])
(out/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
print(json.dumps({k:v for k,v in report.items() if k not in ['example_first_state','methods']},indent=2))
