"""Audit completed K3/K4 runs and visualize actual independent-test data."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.ticker import StrMethodFormatter

ROOT=Path(__file__).resolve().parents[1]/'results/reviewer_uav_scaling_k3_k4_seed42_20261102'
OUT=ROOT/'figures_final_20261103'
COLORS={3:'#D62728',4:'#1F77B4'}

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(values):
    x=np.asarray(values,float)
    rng=np.random.default_rng(20261103)
    ci=np.percentile(rng.choice(x,(5000,len(x)),replace=True).mean(1),[2.5,97.5])
    return dict(mean=float(x.mean()),sd=float(x.std(ddof=1)),ci95=ci.tolist(),n=len(x))
def wilson(success,n):
    p=success/n;z=1.959963984540054;den=1+z*z/n
    c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [100*(c-h),100*(c+h)]
def save(fig,name):
    for ext in ['pdf','png','svg']: fig.savefig(OUT/f'{name}.{ext}',dpi=600)
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9.5,'axes.labelsize':10,
        'axes.titlesize':10,'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':9,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    summary={'created_at':datetime.now().astimezone().isoformat(),'training_seed':42,
        'scope':'One separately trained model per K; 100 test seeds 921000--921099 per WCT subset. Cross-K tests are distribution matched, not identical user realizations. Per-UAV resource budgets fixed; aggregate resources grow with K.',
        'uncertainty':'Bootstrap95 mean CI over 100 test seeds conditional on saved model; Wilson95 for nominal survival. Not across-training-seed uncertainty.',
        'timing':'Concurrent diagnostics excluded from formal scalability figures; no clean runtime-scaling claim.',
        'metric':'Mean per-slot total SLRR = sum over all served slots and users / 50, with zero contribution after early termination. Delay is mean active-user delay over executed slots. Energy is actual simulated fleet energy, including early-terminated episodes.',
        'models':{},'sources':[]}
    train={}; cases={}; csvrows=[]
    for k in (3,4):
        root=ROOT/f'k{k}';d=root/f'source_snapshot/results/formal_dars_k{k}_seed42'
        assert read(root/'run_status.json')['status']=='completed'
        verification=read(root/'completion_verification.json')
        assert sha(d/'per_sac_cvx.pth')==verification['checkpoint_sha256']
        assert read(root/'test/progress.json')['episodes']==1100
        train[k]=[json.loads(l) for l in (d/'per_sac_cvx.train.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(train[k])==15000 and len(read(d/'per_sac_cvx.validation.json'))==50
        cases[k]={};records=0
        for f in sorted((root/'test').glob('wct_*.episodes.json')):
            rows=read(f);assert len(rows)==100
            assert [r['seed'] for r in rows]==list(range(921000,921100))
            for r in rows:
                assert r['energy_violation_steps']==0
                assert np.isclose(r['cumulative_total_slrr']/50,r['mean_total_slrr_fixed_horizon'])
                assert np.isclose(r['energy_fly_j']+r['energy_hover_j']+r['energy_comp_j'],r['energy_out_j'])
                executed=[t for t in r['trace'] if t['action_executed']]
                assert len(executed)==r['executed_steps']
                for t in executed:
                    assert min(t['battery_after_flight'])>=-1e-6 and min(t['battery_after_service'])>=-1e-6
                for key,tk in [('energy_fly_j','energy_fly'),('energy_hover_j','energy_hover'),('energy_comp_j','energy_comp')]:
                    assert np.isclose(r[key],sum(np.sum(t[tk]) for t in executed))
            cases[k][f.name.split('.')[0]]=rows;records+=len(rows)
            summary['sources'].append({'path':str(f),'sha256':sha(f),'episodes':len(rows)})
        assert records==1100 and len(cases[k])==11
        nominal=cases[k]['wct_0123'];metrics={}
        for key in ['reward','mean_total_slrr_fixed_horizon','delay_executed','energy_out_j','energy_fly_j','energy_hover_j','energy_comp_j','executed_steps']:
            metrics[key]=stats([r[key] for r in nominal])
        success=sum(r['survived'] for r in nominal)
        metrics['completion_percent']={'mean':success,'ci95':wilson(success,100),'success':success,'n':100}
        metrics['last300_training_reward']=float(np.mean([r['reward'] for r in train[k][-300:]]))
        metrics['energy_violation_steps_all1100']=0
        metrics['scarcity']={}
        for count in (1,2,4):
            subsets=[rows for name,rows in cases[k].items() if len(name[4:])==count]
            means=np.mean([[r['mean_total_slrr_fixed_horizon'] for r in rows] for rows in subsets],axis=0)
            survival=np.mean([[float(r['survived']) for r in rows] for rows in subsets],axis=0)
            metrics['scarcity'][count]={'slrr':stats(means),'completion_percent':stats(survival*100),'subsets':len(subsets)}
        summary['models'][k]=metrics
        for name,rows in cases[k].items():
            for r in rows: csvrows.append([k,42,name,r['seed'],r['mean_total_slrr_fixed_horizon'],r['delay_executed'],r['survived'],r['executed_steps'],r['energy_fly_j'],r['energy_hover_j'],r['energy_comp_j'],r['energy_out_j']])
    with (OUT/'test_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['K','training_seed','case','test_seed','total_slrr_per_horizon','delay_s','survived','executed_slots','flight_j','hover_j','task_compute_j','total_j']);w.writerows(csvrows)
    fig,ax=plt.subplots(figsize=(7.16,3.1));fig.subplots_adjust(left=.10,right=.95,bottom=.19,top=.87)
    for k in (3,4):
        y=np.array([r['reward'] for r in train[k]]);x=np.arange(1,len(y)+1)
        ax.plot(x,y,c=COLORS[k],alpha=.10,lw=.3,rasterized=True)
        ax.plot(x[299:],np.convolve(y,np.ones(300)/300,'valid'),c=COLORS[k],ls='-' if k==3 else '--',lw=1.4,label=f'{k} UAVs')
    ax.set(xlabel='Training episode',ylabel='Episode return',xlim=(0,15000));ax.set_xticks(np.arange(0,15001,3000));ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'));ax.grid(alpha=.22,lw=.5)
    ax.legend(loc='lower right',frameon=True,edgecolor='none',facecolor='white')
    fig.text(.54,.95,'Training seed 42; 300-episode trailing mean (pale: raw return)',ha='center',va='top',fontsize=9)
    save(fig,'k3_k4_training')
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.65));fig.subplots_adjust(left=.10,right=.98,bottom=.13,top=.91,hspace=.60,wspace=.32)
    specifications=[('mean_total_slrr_fixed_horizon',1,'Mean per-slot total SLRR','(a) Service benefit',2,11),('delay_executed',1,'Mean completion delay (s)','(b) Task delay',2,8),('energy_out_j',.001,'Fleet energy per episode (kJ)','(c) Energy cost',1,320),('completion_percent',1,'Full-horizon completion (%)','(d) Service continuity',0,119)]
    for ax,(key,scale,label,title,dec,upper) in zip(axes.flat,specifications):
        for j,k in enumerate((3,4)):
            st=summary['models'][k][key];mean=st['mean']*scale;lo,hi=np.array(st['ci95'])*scale
            ax.bar(j,mean,width=.52,color=COLORS[k],alpha=.90,zorder=2)
            ax.errorbar(j,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='none',ecolor='#303030',capsize=4,lw=1,zorder=3)
            ax.text(j,max(mean,hi)+upper*.035,f'{mean:.{dec}f}'+('%' if key=='completion_percent' else ''),ha='center',va='bottom',fontsize=9.5)
        ax.set_xticks([0,1],['3 UAVs','4 UAVs']);ax.set_ylim(0,upper);ax.set_ylabel(label);ax.set_title(title,pad=9);ax.grid(axis='y',alpha=.23,lw=.5,zorder=0)
        if key=='completion_percent':ax.set_yticks([0,25,50,75,100])
    fig.text(.54,.025,'4 active WCTs; 100 test episodes / model; bars: mean; whiskers: 95% CI',ha='center',fontsize=9)
    save(fig,'k3_k4_test_performance')
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.9));fig.subplots_adjust(left=.10,right=.98,bottom=.25,top=.81,wspace=.33)
    for k in (3,4):
        x=np.array([1,2,4]);sc=summary['models'][k]['scarcity'];values=[sc[c]['slrr']['mean'] for c in (1,2,4)]
        ci=np.array([sc[c]['slrr']['ci95'] for c in (1,2,4)])
        axes[0].errorbar(x,values,yerr=[np.array(values)-ci[:,0],ci[:,1]-values],c=COLORS[k],marker='o' if k==3 else 's',ls='-' if k==3 else '--',capsize=3,lw=1.3,label=f'{k} UAVs')
        survival=[sc[c]['completion_percent']['mean'] for c in (1,2,4)]
        axes[1].plot(x,survival,c=COLORS[k],marker='o' if k==3 else 's',ls='-' if k==3 else '--',lw=1.3)
    for ax in axes:
        ax.set_xticks([1,2,4],['1','2','4']);ax.set_xlabel('Number of active WCTs');ax.grid(alpha=.23,lw=.5)
    axes[0].set_ylabel('Mean per-slot total SLRR');axes[0].set_ylim(0,10)
    axes[1].set_ylabel('Full-horizon completion (%)');axes[1].set_ylim(-5,106);axes[1].set_yticks([0,25,50,75,100])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
    fig.text(.54,.035,'Frozen policies trained with 4 WCTs; all station subsets retained',ha='center',fontsize=9)
    save(fig,'k3_k4_wct_transfer')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    captions=r'''% Proposed captions only. Main manuscript is unchanged.
% Training: k3_k4_training.pdf
Training returns of DARS with three and four UAVs (training seed 42).
Solid and dashed curves show trailing means over 300 episodes; pale traces
show unsmoothed returns. The return retains the original reward and battery
penalty; it is not the physical total SLRR.

% Test: k3_k4_test_performance.pdf
Independent evaluation of separately trained three- and four-UAV policies
with 30 users and four active WCTs. Each model is evaluated on 100 test
episodes. The total SLRR is summed over users and averaged over a fixed
50-slot horizon, with zero subsequent contribution after early termination.
Delay is measured over executed service slots, while energy includes actual
flight, hovering, and task-computation consumption. Error bars show 95\%
bootstrap confidence intervals for means and Wilson intervals for completion
rates, conditional on the single trained model. Per-UAV resources are fixed;
aggregate resources increase with the number of UAVs. The populations are
distribution matched across UAV counts, not identical paired realizations.

% Scarcity: k3_k4_wct_transfer.pdf
Frozen-policy transfer from four active WCTs to reduced charging availability.
All six two-station and all four one-station subsets are evaluated. Subsets
are averaged within each test seed before summary statistics are computed.
The three- and four-UAV policies both fail to complete the full horizon under
one or two active WCTs. These tests do not represent retraining under scarcity.
'''
    (OUT/'captions.tex').write_text(captions,encoding='utf-8')
    print(json.dumps({'output':str(OUT),'models':summary['models']},ensure_ascii=True))

if __name__=='__main__':main()
