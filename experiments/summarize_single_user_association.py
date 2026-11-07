"""Aggregate relocation diagnostics without treating repeated states as seeds."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from diagnose_single_user_association import summarize

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def main(root):
    assert read(root/'batch_status.json')['status']=='completed'
    all_users=[];all_candidates=[];all_states=[];per_seed=[];scenarios=[];examples=[]
    for scenario in ('base','high_load','low_battery'):
        us=[];cs=[];ss=[];seed_summaries=[]
        for seed in (42,43,44):
            folder=root/f'seed{seed}_{scenario}'
            assert read(folder/'manifest.json')['status']=='completed'
            u,c,s=[rows(folder/f'{key}.jsonl') for key in ('users','candidates','states')]
            summary=summarize(u,c,s);original=read(folder/'summary.json')
            assert summary==original
            for candidate in c:
                candidate.update(training_seed=seed, source_directory=folder.name)
            summary.update(scenario=scenario,training_seed=seed);seed_summaries.append(summary)
            per_seed.append(dict(scenario=scenario,training_seed=seed,
                **{k:summary[k] for k in ('states','greedy_associated_user_states','with_geometric_alternative',
                    'with_capacity_eligible_alternative','with_feasible_alternative','worse_latency_user_states',
                    'system_improving_user_states','jointly_improving_user_states','eligible_fraction',
                    'worse_latency_fraction_among_eligible','system_improving_fraction_among_eligible',
                    'jointly_improving_fraction_among_eligible','faster_user_but_system_worse')},
                faster_user_mean_delay_reduction_percent=summary['user_delay_reduction_percent_improved']['mean']))
            us+=u;cs+=c;ss+=s
        pooled=summarize(us,cs,ss)
        pooled['worse_latency_fraction_among_all_associated']=pooled['worse_latency_user_states']/pooled['greedy_associated_user_states']
        means={}
        for key in ('eligible_fraction','worse_latency_fraction_among_eligible','system_improving_fraction_among_eligible','jointly_improving_fraction_among_eligible'):
            v=[x[key] for x in seed_summaries if x[key] is not None]
            means[key]=dict(mean=float(np.mean(v)) if v else None,sd=float(np.std(v,ddof=1)) if len(v)>1 else None,training_seed_count=len(v))
        for descriptor,condition in (
            ('more_cpu',lambda c:c['alternative_cpu_hz']>c['greedy_cpu_hz']+1e-8*max(1,c['greedy_cpu_hz'])),
            ('more_bandwidth',lambda c:c['alternative_bandwidth_hz']>c['greedy_bandwidth_hz']+1e-8*max(1,c['greedy_bandwidth_hz'])),
            ('more_battery',lambda c:c['target_battery_before_j']>c['source_battery_before_j']+1e-6),
            ('more_computation_budget',lambda c:c['target_computation_budget_j']>c['source_computation_budget_j']+1e-6)):
            selected=[c for c in cs if c['feasible'] and condition(c)]
            pooled['alternatives_with_'+descriptor]=dict(count=len(selected),user_improved=sum(c['user_delay_improved'] for c in selected),
                system_improved=sum(c['system_slrr_improved'] for c in selected))
        scenarios.append(dict(scenario=scenario,pooled_descriptive=pooled,three_training_seed_statistics=means))
        for name,predicate in [('user_and_system_improve',lambda c:c['feasible'] and c['user_delay_improved'] and c['system_slrr_improved']),
                               ('user_improves_system_worsens',lambda c:c['feasible'] and c['user_delay_improved'] and c['system_slrr_worsened'])]:
            # Fixed input order; never select the largest gain as a representative.
            example=next((c for c in cs if predicate(c)),None)
            if example:examples.append(dict(scenario=scenario,selection_rule='First qualifying row in training seed 42/43/44, test seed, slot, user order',category=name,candidate=example))
        all_users+=us;all_candidates+=cs;all_states+=ss
    total=summarize(all_users,all_candidates,all_states)
    assert total['states']==2576
    report=dict(status='completed',total=total,scenarios=scenarios,
        statistical_scope='User-state counts are descriptive conditional frequencies, not independent training replicates. Equal-weight training-seed means/SD reported separately.',
        population='Active users already assigned by Greedy. No-alternative users and unassociated active users are counted separately. Rejected trajectory attempts are outside the existing input snapshot population.',
        causal_scope='One association changes, but allocations and safety corrections can change for all users. CPU, bandwidth and battery descriptors co-occur; they are not isolated causal effects.',
        limitations='No swaps for capacity-full targets, no reassignment of unassociated users, no future trajectory writeback, no retraining; these results do not establish a global association optimum or survival improvement.')
    (root/'aggregate_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (root/'illustrative_cases.json').write_text(json.dumps(examples,indent=2)+'\n',encoding='utf-8')
    with (root/'per_training_seed.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(per_seed[0]));w.writeheader();w.writerows(per_seed)
    compact=[]
    for s in scenarios:
        x=s['pooled_descriptive']
        compact.append(dict(scenario=s['scenario'],associated_user_states=x['greedy_associated_user_states'],
            feasible_alternative_user_states=x['with_feasible_alternative'],faster_user_states=x['worse_latency_user_states'],
            faster_percent_among_eligible=100*x['worse_latency_fraction_among_eligible'],
            faster_percent_among_all_associated=100*x['worse_latency_fraction_among_all_associated'],
            system_improving_user_states=x['system_improving_user_states'],jointly_improving_user_states=x['jointly_improving_user_states'],
            jointly_improving_percent_among_eligible=100*x['jointly_improving_fraction_among_eligible'],
            faster_but_system_worse=x['faster_user_but_system_worse'],
            mean_delay_reduction_percent_among_faster=x['user_delay_reduction_percent_improved']['mean']))
    with (root/'scenario_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(compact[0]));w.writeheader();w.writerows(compact)
    text=['Single-user reassociation diagnostic',f"States: {total['states']}; associated user-state pairs: {total['greedy_associated_user_states']}; feasible alternative pairs: {total['with_feasible_alternative']}.",
        report['statistical_scope'],report['population'],report['causal_scope'],report['limitations'],'']
    for row in compact:text.append(json.dumps(row))
    text += ['',f"Greedy reconstruction maximum error: {total['greedy_reconstruction_max_error']}; executed candidate energy excess maximum: {total['energy_excess_j_max']} J.",
        'Raw per-user outcomes, candidate CPU/bandwidth/battery budgets, safety corrections, individual and system SLRR changes are retained under each seed/scenario directory.']
    (root/'RESULTS.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(total=total,compact=compact),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
