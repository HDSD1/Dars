"""Summarize existing exact nominal associations; no new rollout or training."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'results'
EXACT = ROOT / 'reviewer_exact_association_20261101/formal'
SNAPS = ROOT / 'reviewer_phase2_three_seeds_20261101/formal/association'
AUDIT = ROOT / 'reviewer_r31_single_user_20261101/formal'
OUT = ROOT / 'reviewer_r31_exact_gap_20261102'


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def save_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    ratios = np.array([r['greedy_to_exact_percent'] for r in rows if r['greedy_to_exact_percent'] is not None])
    tail_n = max(1, int(np.ceil(.05 * len(ratios))))
    greedy = sum(r['greedy_total_slrr'] for r in rows)
    exact = sum(r['exact_total_slrr'] for r in rows)
    return dict(
        states=len(rows), positive_optimum_states=len(ratios),
        excluded_nonpositive_optimum_states=len(rows)-len(ratios),
        mean_state_ratio_percent=float(ratios.mean()),
        median_state_ratio_percent=float(np.median(ratios)),
        p05_state_ratio_percent=float(np.percentile(ratios, 5)),
        worst_five_percent_mean_ratio=float(np.sort(ratios)[:tail_n].mean()),
        worst_five_percent_count=tail_n,
        minimum_state_ratio_percent=float(ratios.min()),
        p95_relative_shortfall_percent=float(np.percentile(100-ratios,95)),
        ratio_of_slrr_sums_percent=100*greedy/exact,
        greedy_mean_total_slrr=greedy/len(rows), exact_mean_total_slrr=exact/len(rows),
        mean_absolute_slrr_gap=(exact-greedy)/len(rows),
        exact_match_states=sum(r['exact_match'] for r in rows),
        exact_match_percent=100*np.mean([r['exact_match'] for r in rows]),
        greedy_state_mean_active_delay_s=float(np.mean([r['greedy_mean_active_delay_s'] for r in rows])),
        exact_state_mean_active_delay_s=float(np.mean([r['exact_mean_active_delay_s'] for r in rows])),
        mean_active_delay_reduction_s=float(np.mean([r['active_delay_reduction_s'] for r in rows])),
        greedy_user_weighted_delay_s=sum(r['greedy_sum_active_delay_s'] for r in rows)/sum(r['active_users'] for r in rows),
        exact_user_weighted_delay_s=sum(r['exact_sum_active_delay_s'] for r in rows)/sum(r['active_users'] for r in rows),
        exact_slrr_solution_has_worse_mean_delay_states=sum(r['active_delay_reduction_s'] < -1e-8 for r in rows),
    )


def sign(value, tolerance):
    return 'better' if value > tolerance else 'worse' if value < -tolerance else 'tied'


def main():
    OUT.mkdir(exist_ok=True)
    protocol=json.loads((EXACT/'protocol.json').read_text(encoding='utf-8'))
    state_rows=[]
    user_rows=[]
    candidate_rows=[]
    hashes={}
    for seed in [42,43,44]:
        src=SNAPS/f'seed{seed}_base/state_results.jsonl'
        digest=hashlib.sha256(src.read_bytes()).hexdigest()
        assert digest == protocol['inputs'][str(seed)]['states_sha256']
        paths=[src,EXACT/f'seed{seed}/states.jsonl',AUDIT/f'seed{seed}_base/users.jsonl',AUDIT/f'seed{seed}_base/candidates.jsonl']
        hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        originals={(r['seed'],r['step']):r for r in read_rows(src)}
        exacts=read_rows(paths[1])
        assert len(exacts)==len(originals)==500
        for r in exacts:
            s=originals[(r['test_seed'],r['slot'])]
            mask=np.array(s['active_mask'],dtype=bool)
            g=np.array(s['methods']['greedy']['delay'])[mask]
            o=np.array(r['exact_delays'])[mask]
            g_slrr=float(np.sum(s['methods']['greedy']['rho']))
            assert abs(g_slrr-r['greedy_total_slrr'])<1e-9
            assert r['exact_total_slrr']>=g_slrr-1e-9
            state_rows.append(dict(
                training_seed=seed,test_seed=r['test_seed'],slot=r['slot'],active_users=int(mask.sum()),
                greedy_total_slrr=g_slrr,exact_total_slrr=r['exact_total_slrr'],
                greedy_to_exact_percent=100*g_slrr/r['exact_total_slrr'] if r['exact_total_slrr']>0 else None,
                absolute_slrr_gap=r['exact_total_slrr']-g_slrr,
                exact_match=abs(r['exact_total_slrr']-g_slrr)<=1e-8,
                greedy_mean_active_delay_s=float(g.mean()),exact_mean_active_delay_s=float(o.mean()),
                active_delay_reduction_s=float((g-o).mean()),
                greedy_sum_active_delay_s=float(g.sum()),exact_sum_active_delay_s=float(o.sum()),
                shared_users=r['counts']['shared_users'],changed_associations=r['changed_associations']))
        user_rows.extend(read_rows(paths[2]))
        for r in read_rows(paths[3]):
            if not r['feasible']:
                continue
            candidate_rows.append(dict(
                training_seed=seed,test_seed=r['test_seed'],slot=r['slot'],user=r['user'],
                own_delay_change=sign(r['user_delay_reduction_s'],1e-8*max(1,r['greedy_delay_s'])),
                system_slrr_change=sign(r['system_total_slrr_gain'],1e-8),
                own_delay_reduction_s=r['user_delay_reduction_s'],system_slrr_gain=r['system_total_slrr_gain'],
                battery_relation=sign(r['target_battery_before_j']-r['source_battery_before_j'],1e-6)))
    counter=Counter((r['own_delay_change'],r['system_slrr_change']) for r in candidate_rows)
    three=[dict(own_delay=a,system_slrr=b,count=counter[a,b],percent=100*counter[a,b]/len(candidate_rows))
           for a in ['better','tied','worse'] for b in ['better','tied','worse']]
    two=[]
    for a in [True,False]:
        for b in [True,False]:
            n=sum((r['own_delay_change']=='better')==a and (r['system_slrr_change']=='better')==b for r in candidate_rows)
            two.append(dict(own_delay_improved=a,system_slrr_improved=b,count=n,percent=100*n/len(candidate_rows)))
    per_seed=[dict(training_seed=seed,**summarize([r for r in state_rows if r['training_seed']==seed])) for seed in [42,43,44]]
    out=dict(
        scope='All 1500 existing nominal snapshots, three frozen training seeds. Conditional exact association under original resource/safety mapping, fixed positions and raw offload proposals; includes local execution. Not joint P1 optimality or long-horizon evaluation.',
        definitions=dict(ratio='100 * Greedy total SLRR / conditional exact total SLRR, computed per state when exact value > 0.',
                         tail='Both P5 and mean of the lowest ceil(0.05*N) state ratios; P95 shortfall = P95(100-ratio).',
                         delay='Average completion delay over all active users, including local-only users; each sampled state equally weighted. User-weighted average reported separately.',
                         cross_table='All feasible single-user relocations; 2x2 uses improved vs not improved, preserving ties explicitly in 3x3.',
                         interpretation='A statewise SLRR optimum need not minimize unweighted mean delay. Snapshots from one rollout are correlated; counts are descriptive, not independent trials.'),
        aggregate=summarize(state_rows),per_training_seed=per_seed,
        relocation_population=dict(associated_user_state_records=len(user_rows),feasible_relocations=len(candidate_rows),
            no_single_relocation_reasons=dict(Counter(r['reason'] for r in user_rows if not r['feasible_alternatives'])),
            target_battery_comparison=dict(Counter(r['battery_relation'] for r in candidate_rows)),
            own_better_system_worse_among_own_better_percent=100*counter['better','worse']/sum(r['own_delay_change']=='better' for r in candidate_rows)),
        cross_table_2x2=two,cross_table_3x3=three,
        shared_users=dict(mean=float(np.mean([r['shared_users'] for r in state_rows])),maximum=max(r['shared_users'] for r in state_rows)),
        source_sha256=hashes)
    (OUT/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save_csv('per_state_metrics.csv',state_rows)
    save_csv('per_seed_metrics.csv',per_seed)
    save_csv('relocation_cross_table_2x2.csv',two)
    save_csv('relocation_cross_table_3x3.csv',three)
    save_csv('relocation_pairs.csv',candidate_rows)
    (OUT/'analysis_source.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ['source_sha256','definitions','per_training_seed']},ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
