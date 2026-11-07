"""Nominal-only R3.1 presentation; existing estimates and raw inputs unchanged."""
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
from plot_r31_four_metrics import ROOT, SOURCE, METRICS, RED, BLUE

PREVIOUS=ROOT/'results/reviewer_r31_single_user_20261101/figures_four_metrics_v1'
OUT=ROOT/'results/reviewer_r31_single_user_20261101/figures_nominal_only_v2'

def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def writecsv(p,data):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((PREVIOUS/'plot_protocol.json').read_text(encoding='utf-8'))
    hashes={k:v for k,v in protocol['input_sha256'].items() if Path(k).parent.name.endswith('_base')}
    assert len(hashes)==3
    for p,h in hashes.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    seeds=[r for r in readcsv(PREVIOUS/'per_training_seed.csv') if r['scenario']=='base']
    summary=[r for r in readcsv(PREVIOUS/'four_metric_summary.csv') if r['scenario']=='base']
    population=[r for r in readcsv(PREVIOUS/'population.csv') if r['scenario']=='base']
    assert sum(int(r['feasible_pairs']) for r in population)==1520
    assert sum(int(r['associated_pairs']) for r in population)==14160
    for filename,data in [('per_training_seed.csv',seeds),('nominal_summary.csv',summary),('population.csv',population)]:writecsv(OUT/filename,data)
    writecsv(OUT/'safety_and_system_slrr.csv',[r for r in readcsv(PREVIOUS/'safety_and_system_slrr.csv') if r['scenario']=='base'])
    protocol.update(input_sha256=hashes,display_scope='Nominal (base) only, per author request. Stress scenarios retained unchanged outside this presentation.',
        plot='Absolute paired means for Greedy and feasible alternative; mean and sample SD across three training seeds. Text shows equal-seed mean favorable-alternative frequency.',
        eligible_pairs=1520,associated_pairs=14160,eligible_percent_all_associated=100*1520/14160)
    (OUT/'plot_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9.5,'axes.titlesize':10,
        'axes.labelsize':9.5,'xtick.labelsize':9,'ytick.labelsize':9.5,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7})
    def panel(ax,key,letter):
        title,_,_,_,_,unit=METRICS[key]
        rr=[r for r in seeds if r['metric']==key]
        vals=np.array([[float(r[field]) for r in rr] for field in ('greedy_mean','alternative_mean')])
        means=vals.mean(axis=1);sd=vals.std(axis=1,ddof=1)
        ax.barh([0,1],means,color=[RED,BLUE],height=.48,zorder=2)
        ax.errorbar(means,[0,1],xerr=sd,fmt='none',ecolor='#333333',lw=.9,capsize=2,zorder=4)
        for si,marker in enumerate(['o','s','^']):
            ax.scatter(vals[:,si],np.array([0,1])+(si-1)*.07,marker=marker,s=16,facecolor='white',edgecolor='#333333',lw=.6,zorder=5)
        limit=max(np.max(vals),max(means+sd))*1.27
        ax.set_xlim(0,limit);ax.set_ylim(1.55,-.55)
        for i in range(2):
            ax.annotate(f'{means[i]:.3f}',(max(max(vals[i]),means[i]+sd[i]),i),xytext=(5,0),textcoords='offset points',va='center',fontsize=8.5)
        ax.set_yticks([0,1],['Greedy','Alternative'])
        ax.set_xlabel(f'{title} ({unit})')
        direction='Lower is better' if key=='delay' else 'Higher allocation' if key in ('cpu','bandwidth') else 'Higher stored energy'
        ax.set_title(f'({letter}) {direction}',loc='left',pad=20)
        rate=float(next(r for r in summary if r['metric']==key)['favorable_percent_seed_mean'])
        ax.text(0,1.04,f'Favorable alternative: {rate:.1f}% of eligible pairs',transform=ax.transAxes,fontsize=8.5,va='bottom')
        ax.grid(axis='x',color='#DFE2E6',lw=.6);ax.set_axisbelow(True)
    def save(fig,name):
        fig.savefig(OUT/f'{name}.pdf');fig.savefig(OUT/f'{name}.png',dpi=600);fig.savefig(OUT/f'{name}.svg');plt.close(fig)
    keys=['delay','cpu','bandwidth','battery']
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.5))
    fig.subplots_adjust(left=.12,right=.98,top=.87,bottom=.12,wspace=.46,hspace=.94)
    for i,key in enumerate(keys):panel(axes.flat[i],key,chr(97+i))
    save(fig,'r31_nominal_four_metrics')
    for i,key in enumerate(keys):
        fig,ax=plt.subplots(figsize=(3.5,2.25));fig.subplots_adjust(left=.25,right=.97,top=.73,bottom=.23)
        panel(ax,key,chr(97+i));save(fig,'r31_nominal_'+key)
    (OUT/'figure_caption.tex').write_text(r'''% Caption proposal only; manuscript unchanged.
\caption{Single-user reassociation diagnostics under the nominal setting.
Greedy and the feasible alternative are evaluated on the same eligible
user--state pairs after resource reallocation and energy-safety correction.
Bars and error bars show equal-weight means and sample standard deviations
across three training seeds; markers show individual seed means.
Annotations report the mean fraction of eligible pairs with lower delay or
larger CPU allocation, bandwidth allocation, or pre-flight battery at the
alternative UAV. There are 1520 eligible pairs among 14160 associated
user--state pairs (10.73\%). Battery describes an energy condition rather
than a guarantee of better service.}
''',encoding='utf-8')
    (OUT/'RESULTS.txt').write_text('R3.1 nominal-only visualization. No retraining or manuscript changes.\n'
        'Three training seeds; 1520 feasible alternatives among 14160 associated pairs (10.73%).\n'
        'All feasible pairs retained, including unfavorable alternatives. Plot values are equal-seed means, not pooled rates.\n'
        'Computation-energy budget retained in nominal_summary.csv: Greedy 31.981672 kJ, alternative 31.710175 kJ; difference -0.271498 kJ.\n'
        'No energy-safety scaling occurred in these nominal candidate comparisons.\n'
        '46 nominal candidates improve user delay but worsen system total SLRR.\n'
        'Previously produced high-load and low-battery results remain available as supplemental diagnostics; excluded from this R3.1 figure per author scope.\n',encoding='utf-8')
    shutil.copy2(__file__,OUT/Path(__file__).name)
    print(OUT)

if __name__=='__main__':main()
