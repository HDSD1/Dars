"""Visualize observed partial Swap training, without extrapolation or test claims."""
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

ROOT=Path(__file__).resolve().parents[1]
SWAP=ROOT/'results/reviewer_swap_retrain_seed42_20261101/source_snapshot/results/formal_swap_seed42/per_sac_cvx.train.jsonl'
GREEDY=ROOT/'results/original_main_seed42_gpu_retry_20261029_234000/per_sac_cvx.train.jsonl'
OUT=ROOT/'results/reviewer_swap_retrain_seed42_20261101/figures_partial_12668'
N=12668
WINDOW=300
RED='#D62728';PURPLE='#A496B4'

def records(p):
    with p.open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    swap=records(SWAP);greedy=records(GREEDY)[:N]
    assert len(swap)==len(greedy)==N
    assert [x['seed'] for x in swap]==[x['seed'] for x in greedy]
    for x in swap:
        assert x['horizon']==50
        assert abs(x['mean_total_slrr_fixed_horizon']-x['cumulative_total_slrr']/50)<1e-10
    fields=['reward','mean_total_slrr_fixed_horizon','slrr_fixed_horizon','delay_executed','survived','min_battery_j','energy_out_j']
    data={'Swap':{},'Greedy':{}}
    for label,rows in [('Swap',swap),('Greedy',greedy)]:
        for k in fields:
            if k not in rows[0]:continue
            a=np.array([r[k] for r in rows],dtype=float);assert np.isfinite(a).all()
            data[label][k]=a
    summary={label:{'last300':{k:float(v[-300:].mean()) for k,v in d.items()},'last1000':{k:float(v[-1000:].mean()) for k,v in d.items()}} for label,d in data.items()}
    summary['Swap']['last300']['survived_episodes']=sum(x['survived'] for x in swap[-300:])
    summary['Swap']['max_energy_violation_j']=max(x['max_energy_violation_j'] for x in swap)
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9.5,'axes.titlesize':10,
        'axes.labelsize':9.5,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    x=np.arange(1,N+1);sx=np.arange(WINDOW,N+1)
    def smooth(a):return np.convolve(a,np.ones(WINDOW)/WINDOW,mode='valid')
    def format_axis(ax,title,ylabel,percent=False):
        ax.set_xlim(0,N);ax.set_xticks([0,4000,8000,12668],['0','4,000','8,000','12,668'])
        ax.set_xlabel('Training episode');ax.set_ylabel(ylabel);ax.set_title(title,loc='left')
        ax.grid(alpha=.24,lw=.5)
        if percent:ax.set_ylim(0,105)
    def save(fig,name):
        fig.savefig(OUT/f'{name}.pdf');fig.savefig(OUT/f'{name}.png',dpi=600);fig.savefig(OUT/f'{name}.svg');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.9));fig.subplots_adjust(left=.10,right=.94,bottom=.11,top=.84,wspace=.34,hspace=.62)
    specs=[('mean_total_slrr_fixed_horizon',1,'(a) Total SLRR','Total SLRR per slot'),
        ('reward',1,'(b) Training return','Episode return'),
        ('survived',100,'(c) Full-horizon completion','Completion in trailing window (%)'),
        ('min_battery_j',.001,'(d) Minimum residual battery','Minimum battery (kJ)')]
    for ax,(key,scale,title,ylabel) in zip(axes.flat,specs):
        vals=data['Swap'][key]*scale
        if key!='survived':ax.plot(x,vals,color='#D4D1DA',alpha=.24,lw=.3,rasterized=True)
        ax.plot(sx,smooth(vals),color=PURPLE,lw=1.5)
        format_axis(ax,title,ylabel,key=='survived')
        if key=='min_battery_j':ax.set_ylim(bottom=0)
    fig.text(.5,.965,'Swap-DARS: observed training through episode 12,668',ha='center',fontsize=10)
    fig.text(.5,.922,'Seed 42 only; 300-episode trailing mean; training interrupted before 15,000.',ha='center',fontsize=8.5)
    save(fig,'swap_partial_training')
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.9));fig.subplots_adjust(left=.10,right=.94,bottom=.11,top=.84,wspace=.34,hspace=.62)
    specs=[('reward',1,'(a) Training return','Episode return'),
        ('slrr_fixed_horizon',1,'(b) Common logged SLRR','Mean-active SLRR (fixed 50 slots)'),
        ('delay_executed',1,'(c) Delay on executed slots','Completion delay (s)'),
        ('survived',100,'(d) Full-horizon completion','Completion in trailing window (%)')]
    for ax,(key,scale,title,ylabel) in zip(axes.flat,specs):
        for label,color,ls in [('Greedy',RED,'-'),('Swap',PURPLE,'--')]:
            ax.plot(sx,smooth(data[label][key]*scale),color=color,lw=1.3,ls=ls,label=label+'-DARS')
        format_axis(ax,title,ylabel,key=='survived')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,frameon=False)
    fig.text(.5,.922,'Matched training interval and seed 42; exploratory training, not held-out evaluation.',ha='center',fontsize=8.5)
    save(fig,'swap_vs_greedy_partial_training')
    with (OUT/'training_data.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['algorithm','episode','episode_seed',*fields])
        for label,rows in [('Swap',swap),('Greedy',greedy)]:
            for i,r in enumerate(rows):w.writerow([label,i+1,r['seed'],*[r.get(k,'') for k in fields]])
    protocol=dict(status='completed_visualization_of_interrupted_training',observed_episodes=N,target_episodes=15000,
        training_seed=42,window=WINDOW,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SWAP,GREEDY)},
        scope='No missing episodes filled, no extrapolation, no retraining or policy evaluation. Both curves use first12668 episodes; smoothing starts at300 with no padding.',
        total_slrr='Swap directly logged cumulative_total_slrr/50, verified per episode. Original Greedy training did not log total SLRR, so no fabricated Greedy total-SLRR curve.',
        common_slrr='The comparison uses slrr_fixed_horizon: mean-active SLRR accumulated over executed slots divided by50. This is distinct from total SLRR.',
        uncertainty='Single training seed. No confidence intervals or between-seed SD. Pale traces show raw episodes, not uncertainty bands.',
        checkpoint='Last observed metrics at12668; latest saved weights are episode12500, so no12668 checkpoint or final test exists.',summary=summary)
    (OUT/'summary.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(__file__,OUT/Path(__file__).name)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
