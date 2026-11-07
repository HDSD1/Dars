"""Independent arithmetic and denominator audit; does not execute any policy."""
import csv
import json
from pathlib import Path
import runpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'results'
SOURCE = ROOT / 'reviewer_phase2_three_seeds_20261101/formal'
DIAG = ROOT / 'reviewer_r31_single_user_20261101/formal'
OUT = ROOT / 'reviewer_r31_metric_audit_20261101'
OUT.mkdir(exist_ok=True)

def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def independently_evaluate(state, association, scales, cfg):
    """Reimplement closed-form allocations/channel/delay in vector arithmetic."""
    a = np.array(association, dtype=int)
    active = np.array(state['active_mask'], dtype=bool)
    task = np.array(state['tasks'], dtype=float)
    x = np.array(state['raw_offload_proposal'], dtype=float)
    x[(a < 0) | ~active] = 0
    for k in range(cfg['K']):
        x[a == k] *= scales[k]
    x[x <= 1e-10] = 0
    ue = np.array(state['ue_positions'])
    uav = np.array(state['candidate_uav_positions'])
    d3 = np.sqrt(((ue[:, None] - uav[None, :]) ** 2).sum(axis=2) + cfg['H'] ** 2)
    theta = np.rad2deg(np.arcsin(cfg['H'] / d3))
    plos = 1 / (1 + cfg['A_ENV'] * np.exp(-cfg['B_ENV'] * (theta - cfg['A_ENV'])))
    gain = cfg['ALPHA0'] / d3 ** cfg['ALPHA_PL'] / (plos * cfg['MU_LOS'] + (1-plos) * cfg['MU_NLOS'])
    spectral = np.log2(1 + cfg['P_TX'] * gain / cfg['SIGMA2'])
    cpu = np.zeros(len(a)); bw = np.zeros(len(a)); edge = np.zeros(len(a))
    comp = np.zeros(cfg['K'])
    for k in range(cfg['K']):
        ids = np.flatnonzero(active & (a == k) & (x > 0))
        if not len(ids):
            continue
        work = x[ids] * task[ids] * cfg['S_CYCLES']
        weights = np.sqrt(work)
        cpu[ids] = cfg['F_UAV'] * weights / weights.sum()
        weights = np.sqrt(x[ids] * task[ids] / spectral[ids, k])
        bw[ids] = cfg['BANDWIDTH'] * weights / weights.sum()
        edge[ids] = x[ids] * task[ids] / (bw[ids] * spectral[ids,k]) + work / cpu[ids]
        comp[k] = np.sum(cfg['KAPPA'] * work * cpu[ids]**2)
    base = task * cfg['S_CYCLES'] / cfg['F_UE']
    delay = np.maximum((1-x)*base, edge)
    rho = np.zeros(len(a)); rho[active] = 1-delay[active]/base[active]
    return dict(cpu=cpu, bw=bw, delay=delay, rho=rho, x=x, comp=comp)

records = []; all_candidates = []; all_states = []; all_users = []
errors = dict(delay_seconds=0., normalized_rho=0., total_slrr=0., cpu_relative=0., bandwidth_relative=0., unit_conversion_delay_seconds=0.)
thresholds = [0, 0.1, 1, 5, 10]
for scene in ['base', 'high_load', 'low_battery']:
    for seed in [42,43,44]:
        name = f'seed{seed}_{scene}'
        manifest = json.loads((SOURCE/'association'/name/'manifest.json').read_text(encoding='utf-8'))
        cfg = runpy.run_path(str(Path(manifest['source_dir'])/'config.py'))
        rows = read_rows(SOURCE/'association'/name/'state_results.jsonl')
        users = read_rows(DIAG/name/'users.jsonl')
        candidates = read_rows(DIAG/name/'candidates.jsonl')
        lookup = {(r['seed'],r['step']):r for r in rows}
        bystate = {}
        for c in candidates:
            r = lookup[c['test_seed'],c['slot']]
            greedy = r['methods']['greedy']
            a = list(greedy['association']); a[c['user']] = c['target_uav']
            alt = independently_evaluate(r,a,c['alternative_safety_scales'],cfg)
            baseline = independently_evaluate(r,greedy['association'],greedy['scales'],cfg)
            i = c['user']
            errors['delay_seconds'] = max(errors['delay_seconds'], abs(alt['delay'][i]-c['alternative_delay_s']), np.max(abs(baseline['delay']-greedy['delay'])))
            delta = alt['rho']-baseline['rho']
            errors['normalized_rho'] = max(errors['normalized_rho'], np.max(abs(delta-c['all_user_slrr_delta'])))
            errors['total_slrr'] = max(errors['total_slrr'], abs(delta.sum()-c['system_total_slrr_gain']))
            errors['cpu_relative'] = max(errors['cpu_relative'], abs(alt['cpu'][i]-c['alternative_cpu_hz'])/max(1,c['alternative_cpu_hz']))
            errors['bandwidth_relative'] = max(errors['bandwidth_relative'], abs(alt['bw'][i]-c['alternative_bandwidth_hz'])/max(1,c['alternative_bandwidth_hz']))
            # Cycles/Hz and Gcycles/GHz must produce identical seconds.
            cycles = alt['x'][i]*r['tasks'][i]*cfg['S_CYCLES']
            if alt['cpu'][i] > 0:
                errors['unit_conversion_delay_seconds'] = max(errors['unit_conversion_delay_seconds'], abs(cycles/alt['cpu'][i] - (cycles/1e9)/(alt['cpu'][i]/1e9)))
            c = dict(c,scenario=scene,training_seed=seed)
            all_candidates.append(c)
            bystate.setdefault((c['test_seed'],c['slot']),[]).append(c)
        gains = []
        for r in rows:
            cs = bystate.get((r['seed'],r['step']),[])
            best = max([0.] + [c['system_total_slrr_gain'] for c in cs if c['feasible']])
            base_total = r['active_users']-r['methods']['greedy']['J']
            swapgain = r['methods']['greedy']['J']-r['methods']['swap']['J']
            st = dict(scenario=scene,training_seed=seed,test_seed=r['seed'],slot=r['step'],baseline_total_slrr=base_total,
                      best_single_transfer_gain=best,swap_gain=swapgain,feasible_candidates=len(cs))
            all_states.append(st);gains.append(best)
        feasible = [c for c in candidates if c['feasible']]
        faster = [c for c in feasible if c['user_delay_improved']]
        records.append(dict(scenario=scene,training_seed=seed,associated_user_states=len(users),eligible=len(feasible),faster=len(faster),
            faster_pct_all_associated=100*len(faster)/len(users),faster_pct_eligible=100*len(faster)/len(feasible),
            mean_best_single_transfer_total_slrr_gain=float(np.mean(gains))))
        all_users.extend(dict(u,scenario=scene,training_seed=seed) for u in users)

summary = dict(independent_reconstruction_max_errors=errors,scenarios=[],per_seed=records,
    scope='Existing snapshots only. Independent equations audit; no new trajectories, policy changes, or training. Best single transfer includes staying with Greedy; candidates are alternatives, not simultaneously executable actions.')
for scene in ['base','high_load','low_battery']:
    cs = [c for c in all_candidates if c['scenario']==scene and c['feasible']]
    us = [u for u in all_users if u['scenario']==scene]
    ss = [s for s in all_states if s['scenario']==scene]
    improved = [c for c in cs if c['user_delay_improved']]
    seedrows = [r for r in records if r['scenario']==scene]
    o=dict(scenario=scene,associated=len(us),eligible=len(cs),faster=len(improved),
           conditional_pct=100*len(improved)/len(cs),all_associated_pct=100*len(improved)/len(us),
           positive_delay_reduction_s_mean=float(np.mean([c['user_delay_reduction_s'] for c in improved])),
           positive_delay_reduction_s_median=float(np.median([c['user_delay_reduction_s'] for c in improved])),
           positive_delay_reduction_percent_mean=float(np.mean([c['user_delay_reduction_percent'] for c in improved])),
           best_single_transfer_gain_mean_across_seeds=float(np.mean([r['mean_best_single_transfer_total_slrr_gain'] for r in seedrows])),
           states_where_best_single_transfer_beats_recorded_swap=sum(s['best_single_transfer_gain']>s['swap_gain']+1e-8 for s in ss),
           threshold_sensitivity=[dict(relative_delay_reduction_percent=t,count=sum(c['user_delay_reduction_percent']>max(t,1e-6) for c in cs),
                                      percent_of_eligible=100*sum(c['user_delay_reduction_percent']>max(t,1e-6) for c in cs)/len(cs)) for t in thresholds])
    summary['scenarios'].append(o)
for filename,rows in [('per_seed.csv',records),('best_single_transfer_states.csv',all_states)]:
    with (OUT/filename).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
assert errors['delay_seconds'] < 1e-8
assert errors['normalized_rho'] < 1e-8
assert errors['cpu_relative'] < 1e-8
assert errors['bandwidth_relative'] < 1e-8
print(json.dumps(summary,ensure_ascii=False,indent=2))
