"""Four-dimension paired audit; preserves all feasible alternatives and seed units."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/reviewer_r31_single_user_20261101/formal'
OUT = ROOT/'results/reviewer_r31_single_user_20261101/figures_four_metrics_v1'
SCENARIOS = ['base', 'high_load', 'low_battery']
LABELS = ['Nominal', 'High load', 'Low battery']
RED, BLUE, PURPLE = '#D62728', '#7196B3', '#A496B4'
METRICS = {
    'delay': ('Completion delay', 'greedy_delay_s', 'alternative_delay_s', 1., -1., 's'),
    'cpu': ('Allocated CPU', 'greedy_cpu_hz', 'alternative_cpu_hz', 1e9, 1., 'GHz'),
    'bandwidth': ('Allocated bandwidth', 'greedy_bandwidth_hz', 'alternative_bandwidth_hz', 1e6, 1., 'MHz'),
    'battery': ('Battery before flight', 'source_battery_before_j', 'target_battery_before_j', 1e3, 1., 'kJ'),
    'budget': ('Computation-energy budget', 'source_computation_budget_j', 'target_computation_budget_j', 1e3, 1., 'kJ'),
}

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def rows(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def csv_write(path, data):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman', 'font.size':9.5,
        'axes.titlesize':10, 'axes.labelsize':9.5, 'xtick.labelsize':9,
        'ytick.labelsize':9.5, 'legend.fontsize':8.5, 'pdf.fonttype':42,
        'ps.fonttype':42, 'svg.fonttype':'path', 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.linewidth':.7, 'savefig.dpi':600})
    seed_rows=[]; population=[]; all_rows={}; hashes={}; safety=[]
    for sc in SCENARIOS:
        all_rows[sc]=[]
        for seed in (42,43,44):
            folder=SOURCE/f'seed{seed}_{sc}'
            assert read(folder/'manifest.json')['status']=='completed'
            p=folder/'candidates.jsonl'; hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
            cs=[r for r in rows(p) if r['feasible']]
            assert len({(r['test_seed'],r['slot'],r['user']) for r in cs})==len(cs)
            us=rows(folder/'users.jsonl')
            assert len(cs)==sum(u['feasible_alternatives']>0 for u in us)
            population.append(dict(scenario=sc, training_seed=seed, associated_pairs=len(us),
                feasible_pairs=len(cs), eligible_percent=100*len(cs)/len(us)))
            all_rows[sc]+=cs
            safety.append(dict(scenario=sc,training_seed=seed,feasible_pairs=len(cs),
                greedy_any_scaling=sum(min(c['greedy_safety_scales'])<1-1e-8 for c in cs),
                alternative_any_scaling=sum(min(c['alternative_safety_scales'])<1-1e-8 for c in cs),
                own_offload_changed=sum(abs(c['greedy_offload']-c['alternative_offload'])>1e-8 for c in cs),
                system_improving=sum(c['system_slrr_improved'] for c in cs),
                faster_but_system_worse=sum(c['user_delay_improved'] and c['system_slrr_worsened'] for c in cs)))
            for key,(_,g,a,unit,sign,_) in METRICS.items():
                old=np.array([c[g] for c in cs]); new=np.array([c[a] for c in cs])
                delta=sign*(new-old)/unit
                tolerance=(np.full(len(cs),1e-6) if key in ('battery','budget') else 1e-8*np.maximum(1,old))/unit
                better=delta>tolerance
                if key=='delay':
                    assert better.tolist()==[c['user_delay_improved'] for c in cs]
                seed_rows.append(dict(scenario=sc,training_seed=seed,metric=key,feasible_pairs=len(cs),
                    favorable_pairs=int(better.sum()),favorable_percent=100*better.mean(),
                    detected_favorable_percent_all_associated=100*better.sum()/len(us),
                    greedy_mean=float(old.mean()/unit),alternative_mean=float(new.mean()/unit),
                    signed_difference_mean=float(delta.mean()),signed_difference_median=float(np.median(delta)),
                    signed_difference_p05=float(np.percentile(delta,5)),signed_difference_p95=float(np.percentile(delta,95)),
                    favorable_only_difference_mean=float(delta[better].mean()) if better.any() else None))
    summary=[]
    for sc in SCENARIOS:
        for key,(_,g,a,unit,sign,units) in METRICS.items():
            rr=[r for r in seed_rows if r['scenario']==sc and r['metric']==key]
            n=sum(r['feasible_pairs'] for r in rr); better=sum(r['favorable_pairs'] for r in rr)
            entry=dict(scenario=sc,metric=key,units=units,feasible_pairs=n,favorable_pairs=better,
                pooled_favorable_percent=100*better/n)
            for m in ('favorable_percent','signed_difference_mean','greedy_mean','alternative_mean'):
                v=[r[m] for r in rr]; entry[m+'_seed_mean']=float(np.mean(v));entry[m+'_seed_sd']=float(np.std(v,ddof=1))
            summary.append(entry)
    assert sum(p['feasible_pairs'] for p in population)==3502
    assert sum(p['associated_pairs'] for p in population)==24879
    assert sum(s['faster_but_system_worse'] for s in safety)==69
    csv_write(OUT/'per_training_seed.csv',seed_rows)
    csv_write(OUT/'four_metric_summary.csv',summary)
    csv_write(OUT/'population.csv',population)
    csv_write(OUT/'safety_and_system_slrr.csv',safety)
    protocol=dict(training_seeds=[42,43,44],input_sha256=hashes,
        numerator='Feasible alternative has strictly lower completion delay, or strictly larger CPU/bandwidth/battery/budget.',
        denominator='All feasible single-user relocation pairs; exactly one candidate per eligible user-state with K=2. Not all associated users.',
        signed_difference='Delay: Greedy minus alternative. Other metrics: alternative minus Greedy. Positive means alternative favorable on that metric.',
        aggregation='Within each training seed, include all feasible pairs, including negative and zero differences. Plot equal-weight mean across three training seeds; error bars sample SD; markers individual training seeds.',
        battery='Battery before flight; computation budget is after fixed flight, hover and charging accounting. Both are conditions, not direct measures of service quality.',
        caveats='No isolated causal attribution to resource or battery variables; conditional pre-termination snapshots; no full rollout replacement; no retraining. Pooled percentages in CSV differ from equal-seed plot means.',
        palette={'Greedy_zero_reference':RED,'alternative':BLUE,'computation_budget':PURPLE})
    (OUT/'plot_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')

    def panel(ax, keys, rate):
        for ki,key in enumerate(keys):
            color=BLUE if ki==0 else PURPLE
            offset=0 if len(keys)==1 else (-.17 if ki==0 else .17)
            y=np.arange(3)+offset
            field='favorable_percent' if rate else 'signed_difference_mean'
            vals=np.array([[r[field] for r in seed_rows if r['scenario']==sc and r['metric']==key] for sc in SCENARIOS])
            means=vals.mean(axis=1);sd=vals.std(axis=1,ddof=1)
            ax.barh(y,means,height=.30 if len(keys)>1 else .48,color=color,alpha=.90,zorder=2,
                edgecolor='white',hatch='///' if ki else None)
            ax.errorbar(means,y,xerr=sd,fmt='none',ecolor='#353535',capsize=2,lw=.9,zorder=4)
            for si,m in enumerate(['o','s','^']):
                ax.scatter(vals[:,si],y+(si-1)*.065,s=14,marker=m,facecolor='white',edgecolor='#353535',lw=.6,zorder=5)
            for j,mean in enumerate(means):
                label=f'{mean:.1f}' if rate else f'{mean:+.3f}'
                edge=max(mean+sd[j],max(vals[j])) if mean>=0 else min(mean-sd[j],min(vals[j]))
                ax.annotate(label,(edge,y[j]),
                    xytext=(5 if mean>=0 else -5,0),textcoords='offset points',va='center',
                    ha='left' if mean>=0 else 'right',fontsize=8)
        ax.set_yticks(range(3),LABELS);ax.set_ylim(2.55,-.55)
        ax.grid(axis='x',color='#DEE1E5',lw=.6,zorder=0);ax.set_axisbelow(True)
        if rate:
            ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Favorable alternative among eligible pairs (%)')
        else:
            ax.axvline(0,color=RED,lw=1.4,zorder=3)
            ax.margins(x=.36)
            unit=METRICS[keys[0]][-1]
            desc='Greedy minus alternative' if keys[0]=='delay' else 'Alternative minus Greedy'
            ax.set_xlabel(f'{desc} ({unit})')

    def legend(fig,battery=False,y=.97):
        handles=[Line2D([0],[0],color=RED,lw=1.5,label='Greedy reference (zero difference)'),
                 Patch(facecolor=BLUE,label='Battery before flight' if battery else 'Feasible alternative')]
        if battery:handles.append(Patch(facecolor=PURPLE,hatch='///',edgecolor='white',label='Computation-energy budget'))
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,y),ncol=len(handles),frameon=False,
            handlelength=1.5,columnspacing=1.)

    def save(fig,name):
        fig.savefig(OUT/f'{name}.pdf')
        fig.savefig(OUT/f'{name}.png',dpi=600)
        fig.savefig(OUT/f'{name}.svg')
        plt.close(fig)

    for key in ('delay','cpu','bandwidth','battery'):
        keys=['battery','budget'] if key=='battery' else [key]
        fig,axes=plt.subplots(1,2,figsize=(7.16,2.65))
        fig.subplots_adjust(left=.12,right=.97,bottom=.23,top=.73,wspace=.49)
        panel(axes[0],keys,True);panel(axes[1],keys,False)
        axes[0].set_title('(a) Frequency',loc='left',pad=7)
        axes[1].set_title('(b) Mean paired difference',loc='left',pad=7)
        legend(fig,key=='battery')
        save(fig,f'r31_{key}')
    fig,axes=plt.subplots(4,2,figsize=(7.16,9.0))
    fig.subplots_adjust(left=.12,right=.97,bottom=.07,top=.89,wspace=.49,hspace=.85)
    for i,key in enumerate(('delay','cpu','bandwidth','battery')):
        keys=['battery','budget'] if key=='battery' else [key]
        panel(axes[i,0],keys,True);panel(axes[i,1],keys,False)
        axes[i,0].set_title(f'({chr(97+i)}) {METRICS[key][0]}: frequency',loc='left',pad=7)
        axes[i,1].set_title('Mean paired difference',loc='left',pad=7)
    fig.legend(handles=[Line2D([0],[0],color=RED,lw=1.5,label='Greedy reference (zero difference)'),
        Patch(facecolor=BLUE,label='Feasible alternative'),
        Patch(facecolor=PURPLE,hatch='///',edgecolor='white',label='Computation-energy budget (d)')],
        loc='upper center',bbox_to_anchor=(.5,.98),frameon=False,ncol=2)
    save(fig,'r31_four_metrics_overview')
    caption=r'''% Figure caption only; main manuscript has not been modified.
\caption{Single-user reassociation diagnostics along frozen DARS trajectories.
Each comparison changes one user's association and recomputes resource allocation
and energy safety corrections. Left panels show the fraction of feasible alternatives
with lower completion delay or larger allocated CPU, bandwidth, battery, or
computation-energy budget. Right panels show signed paired differences over all
feasible alternatives, including unfavorable cases. Positive differences favor the
alternative on the displayed metric; red lines denote zero difference from Greedy.
Bars and error bars are the mean and sample standard deviation across three training
seeds; markers denote individual training seeds. The eligible sample counts are
1520, 1121, and 861 for nominal, high-load, and low-battery conditions, respectively.
Battery and energy-budget comparisons describe service conditions, not causal
effects on delay. Low-battery results cover states before episode termination.}
'''
    (OUT/'figure_caption.tex').write_text(caption,encoding='utf-8')
    shutil.copy2(__file__,OUT/Path(__file__).name)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
