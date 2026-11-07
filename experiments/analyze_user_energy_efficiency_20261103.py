"""Post-hoc descriptive ratios and exact energy accounting; no new rollouts."""
from pathlib import Path
import csv
import json
import hashlib
from datetime import datetime
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'results/reviewer_users_m20_m25_m30_seed42_5000_20261102'
OUT=ROOT/'energy_mechanism_20261103'
ALGOS=['per_sac_cvx','esac_per_cvx','sac_cvx','td3_cvx','maddpg_cvx']
NAMES=['DARS','ESAC-PER','SAC','TD3','MADDPG']

def main():
    OUT.mkdir(exist_ok=False)
    data={};table=[];sources=[];boot={};contrasts=[];cohorts=[]
    for m in (20,25,30):
        idx=np.random.default_rng(20261103+m).integers(0,100,(10000,100))
        hashes=None
        for algo,name in zip(ALGOS,NAMES):
            p=ROOT/f'evaluation/m{m}/{algo}/episodes.jsonl';b=p.read_bytes();rows=[json.loads(l) for l in b.splitlines()];assert len(rows)==100
            assert [r['seed'] for r in rows]==list(range(931000,931100))
            hs=[r['external_state_hashes'] for r in rows]
            if hashes is None:hashes=hs
            else:assert hashes==hs
            keys=['cumulative_total_slrr','mean_total_slrr_fixed_horizon','energy_out_j','energy_in_j','energy_charge_accepted_j','energy_charge_overflow_j','energy_hover_j','energy_fly_j','energy_comp_j','executed_steps','survived']
            x={key:np.asarray([r[key] for r in rows],float) for key in keys}
            x['initial_j']=np.asarray([sum(r['initial_battery_j']) for r in rows]);x['final_j']=np.asarray([sum(r['final_battery_j']) for r in rows])
            assert np.allclose(x['initial_j']+x['energy_charge_accepted_j']-x['energy_out_j'],x['final_j'],atol=1e-6)
            assert np.allclose(x['energy_out_j'],x['energy_hover_j']+x['energy_fly_j']+x['energy_comp_j'])
            assert np.allclose(x['energy_hover_j'],x['executed_steps']*2358.86)
            assert np.allclose(x['cumulative_total_slrr']/50,x['mean_total_slrr_fixed_horizon'])
            record=dict(M=m,algorithm=name,training_seed=42,training_episodes=15000 if m==30 else 5000,
                completion_percent=float(x['survived'].mean()*100),mean_executed_slots=float(x['executed_steps'].mean()),
                charging_kj=float(x['energy_charge_accepted_j'].mean()/1000),consumed_kj=float(x['energy_out_j'].mean()/1000),
                final_battery_kj=float(x['final_j'].mean()/1000),mean_total_slrr_fixed50=float(x['mean_total_slrr_fixed_horizon'].mean()),
                mean_cumulative_total_slrr=float(x['cumulative_total_slrr'].mean()),
                consumed_kj_per_executed_slot=float(x['energy_out_j'].sum()/x['executed_steps'].sum()/1000))
            # Ratio of aggregate sums, NOT average of individual episode ratios.
            C=x['cumulative_total_slrr']
            for label,den in [('per_consumed_kj','energy_out_j'),('per_charged_kj','energy_charge_accepted_j')]:
                ratio=float(C.sum()/(x[den].sum()/1000));bs=C[idx].sum(1)/(x[den][idx].sum(1)/1000)
                ci=np.percentile(bs,[2.5,97.5]);record[label]=ratio;record[label+'_ci_low']=float(ci[0]);record[label+'_ci_high']=float(ci[1]);boot[m,algo,label]=bs
            for survived in (True,False):
                selected=[r for r in rows if bool(r['survived'])==survived]
                if not selected:continue
                cohorts.append(dict(M=m,algorithm=name,survived=survived,n=len(selected),
                    mean_slots=float(np.mean([r['executed_steps'] for r in selected])),
                    mean_consumed_kj=float(np.mean([r['energy_out_j'] for r in selected])/1000),
                    mean_final_min_uav_battery_kj=float(np.mean([min(r['final_battery_j']) for r in selected])/1000),
                    mean_final_max_uav_battery_kj=float(np.mean([max(r['final_battery_j']) for r in selected])/1000),
                    termination_reasons={key:sum(r['termination_reason']==key for r in selected) for key in sorted(set(r['termination_reason'] for r in selected))}))
            table.append(record);data[m,algo]=x
            sources.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest()))
        for algo,name in zip(ALGOS[1:],NAMES[1:]):
            for label in ['per_consumed_kj','per_charged_kj']:
                diff=boot[m,'per_sac_cvx',label]-boot[m,algo,label]
                d=next(r for r in table if r['M']==m and r['algorithm']=='DARS');other=next(r for r in table if r['M']==m and r['algorithm']==name)
                contrasts.append(dict(M=m,comparison='DARS minus '+name,metric=label,point_difference=d[label]-other[label],paired_bootstrap_ci95=np.percentile(diff,[2.5,97.5]).tolist()))
    d=data[30,'per_sac_cvx'];b=data[30,'maddpg_cvx']
    components={k:float((d[k]-b[k]).mean()/1000) for k in ['energy_out_j','energy_hover_j','energy_fly_j','energy_comp_j']}
    delta_slots=float((d['executed_steps']-b['executed_steps']).mean())
    assert np.isclose(components['energy_hover_j'],delta_slots*2.35886)
    assert np.isclose(components['energy_out_j'],sum(components[k] for k in ['energy_hover_j','energy_fly_j','energy_comp_j']))
    accounting=dict(comparison='DARS minus MADDPG at M30',components_kj=components,mean_executed_slot_difference=delta_slots,
        fleet_hover_kj_per_slot=2.35886,hover_share_of_total_gap=components['energy_hover_j']/components['energy_out_j'],
        interpretation='Observed accounting identity: nearly all fleet energy gap is hovering accumulated over different service durations. Not a counterfactual prediction of a forced-completion MADDPG episode. Flight and task-computation components still differ.')
    for filename,records in [('all_metrics.csv',table),('termination_cohorts.csv',cohorts)]:
        with (OUT/filename).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    caveats=['All100 episodes per model retained, including early termination; fixed50 totalSLRR denominator.',
        'Ratios are post-hoc descriptive metrics; no optimization/retraining for energy efficiency occurred.',
        'Charging-only denominator omits initial battery depletion and is not overall energy efficiency; no claim that it measures visits, overcharging, or charging optimality.',
        'Cumulative totalSLRR numerator equals50times fixed-horizon per-slot totalSLRR. Do not mix these definitions.',
        'Bootstrap intervals resample paired external test seeds within M and are conditional on one trained model; not training-seed robustness.',
        'Pairwise intervals are exploratory pointwise intervals across24 contrasts, not multiplicity-adjusted confirmatory evidence.',
        'Completed-only/failed-only groups are descriptive diagnostics, not fair standalone policy rankings.',
        'M20/25 trained5000 andM30 trained15000; cross-M trends do not isolate workload changes.',
        'Final battery imbalance can coexist with infeasible next action; rejected actions do not mean negative battery or literal battery exhaustion.']
    scan={'coverage':'11/11 checked','Simpson':'Report each M separately; no pooling across workloads.',
        'ecological':'Do not infer individual charging visits from algorithm-average totals.',
        'Berkson':'All scheduled episodes retained.',
        'collider':'Survival-conditioned cohorts diagnostic only.',
        'base_rate':'Completion counts and full100 denominator retained.',
        'regression_to_mean':'No selection of extreme episodes/models; final checkpoints retained.',
        'survivorship':'Full cohort primary, post-termination service zero.',
        'look_elsewhere':'Both proposed denominators and all15 configurations reported; no significance-only selection.',
        'forking_paths':'Exploratory post-hoc; formulas and bootstrap fixed before computation.',
        'causal':'Energy decomposition is an accounting identity; does not identify overcharging/route causation.',
        'reverse_causality':'Charging and service duration jointly evolve; direction not inferred from totals.'}
    report=dict(checked_at=datetime.now().astimezone().isoformat(),verification_status='VERIFIED from saved episode records',table=table,accounting_M30=accounting,cohorts=cohorts,paired_contrasts=contrasts,caveats=caveats,statistical_fallacy_scan=scan,sources=sources)
    (OUT/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tex=[r'% Exploratory descriptive metrics, one trained seed42. All100 episodes retained.',r'\begin{tabular}{rlrrrrrr}',r'\hline',r'$M$ & Method & Completion (\%) & $E^{\rm in}$ (kJ) & $E^{\rm out}$ (kJ) & $\bar S$ & $\eta_{\rm out}$ & $\eta_{\rm in}$ \\',r'\hline']
    for r in table:tex.append(f"{r['M']} & {r['algorithm']} & {r['completion_percent']:.0f} & {r['charging_kj']:.2f} & {r['consumed_kj']:.2f} & {r['mean_total_slrr_fixed50']:.4f} & {r['per_consumed_kj']:.4f} & {r['per_charged_kj']:.4f} "+r'\\')
    tex.extend([r'\hline',r'\end{tabular}',r'% Sbar is fixed50 mean totalSLRR. eta=sum(50*Sbar)/sum(energy in kJ).',r'% Charging-normalized benefit excludes initial-battery depletion; not total energy efficiency.',r'% M20/25:5000 training episodes; M30:15000. No causal charging-policy conclusion.'])
    (OUT/'table.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    print(json.dumps({'table':table,'accounting':accounting,'contrast30':[x for x in contrasts if x['M']==30 and x['comparison']=='DARS minus MADDPG'],'cohorts30':[x for x in cohorts if x['M']==30]},ensure_ascii=True))

if __name__=='__main__':main()
