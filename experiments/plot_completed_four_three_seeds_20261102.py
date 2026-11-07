"""Three-seed completed algorithms: training and existing common natural-ID tests."""
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
from plot_five_algorithms_seeds42_43 import directory

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/completed_four_three_seeds_20261102'
TEST=ROOT/'results/reviewer_hotspot_generalization_v2_iofix_20261031'
STYLES=[('per_sac_cvx','DARS','#D62728','-', '*'),('esac_per_cvx','ESAC-PER','#1F77B4','--','p'),
        ('sac_cvx','SAC','#2CA02C','-.','o'),('maddpg_cvx','MADDPG','#FF7F0E',':','^')]
WINDOW=300

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(p,rr):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9.5,'axes.labelsize':9.5,
        'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    smooth={};ends=[];testrows=[];sources=[];seedbank=None;env=None
    fields=['slrr_sum_per_horizon','delay_executed','survived','offloaded_bits_fraction',
        'energy_fly_j','energy_hover_j','energy_comp_j','energy_out_j','local_delay_active_mean_s',
        'edge_delay_offloading_mean_s','executed_steps']
    with (OUT/'training_curves.csv').open('w',newline='',encoding='utf-8') as cf:
        writer=csv.writer(cf);writer.writerow(['algorithm','training_seed','episode','return_raw','mean_active_slrr_fixed50_raw','return_ma300','mean_active_slrr_fixed50_ma300'])
        for algo,label,color,ls,marker in STYLES:
            smooth[algo]=[]
            for seed in (42,43,44):
                d=directory(algo,seed) if seed!=44 else ROOT/'results/formal_all_five_seeds43_44_restart_20261031/source_snapshot/results/formal_all_five_seeds43_44_session2_20261031_seed44'
                m=read(d/f'{algo}.manifest.json');assert m['status']=='completed' and m['training_seed']==seed
                signature=(m['environment'],m['training'],{k:m['source_sha256'][k] for k in ('UAV_env.py','evaluation.py','train_unified.py','config.py','DARS.py')})
                if env is None:env=signature
                assert signature==env,'Incompatible original training definitions'
                tp=d/f'{algo}.train.jsonl';raw=[]
                with tp.open(encoding='utf-8') as f:
                    for line in f:
                        r=json.loads(line);assert r['horizon']==50
                        raw.append([r['reward'],r['slrr_fixed_horizon']])
                raw=np.asarray(raw);assert raw.shape==(15000,2) and np.isfinite(raw).all()
                sm=np.column_stack([np.convolve(raw[:,j],np.ones(WINDOW)/WINDOW,mode='valid') for j in (0,1)])
                smooth[algo].append(sm)
                for i,row in enumerate(raw):writer.writerow([label,seed,i+1,*row,*(sm[i-WINDOW+1] if i>=WINDOW-1 else ('',''))])
                ends.append(dict(algorithm=label,training_seed=seed,last1000_return=float(raw[-1000:,0].mean()),last1000_mean_active_slrr_fixed50=float(raw[-1000:,1].mean())))
                ep=TEST/f'{algo}_seed{seed}'/'natural_id.episodes.jsonl';erows=[];bank=[]
                with ep.open(encoding='utf-8') as f:
                    for line in f:
                        r=json.loads(line);assert r['horizon']==50
                        erows.append({k:r[k] for k in fields});bank.append((r['seed'],r['bank_entry_sha256']))
                if seedbank is None:seedbank=bank
                assert bank==seedbank and len(bank)==201,'Unequal test tasks/seeds'
                testrows.append(dict(algorithm=label,training_seed=seed,episodes=len(erows),**{k:float(np.mean([r[k] for r in erows if r[k] is not None])) for k in fields}))
                sources.append(dict(algorithm=label,seed=seed,training_file=str(tp),training_sha256=sha(tp),manifest_sha256=sha(d/f'{algo}.manifest.json'),evaluation_file=str(ep),evaluation_sha256=sha(ep)))
    writecsv(OUT/'last1000_training_summary.csv',ends);writecsv(OUT/'test_per_training_seed.csv',testrows)
    aggregate=[]
    for _,label,_,_,_ in STYLES:
        rr=[r for r in testrows if r['algorithm']==label];row=dict(algorithm=label,training_seeds=3,test_episodes_per_seed=201)
        for f in fields:
            v=[r[f] for r in rr];row[f+'_mean']=float(np.mean(v));row[f+'_sd']=float(np.std(v,ddof=1))
        aggregate.append(row)
    writecsv(OUT/'test_summary.csv',aggregate)
    def save(fig,name):
        fig.savefig(OUT/f'{name}.pdf');fig.savefig(OUT/f'{name}.png',dpi=600);fig.savefig(OUT/f'{name}.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.9));fig.subplots_adjust(left=.09,right=.95,bottom=.22,top=.77,wspace=.32)
    x=np.arange(WINDOW,15001)
    for algo,label,color,ls,marker in STYLES:
        v=np.array(smooth[algo]);mean=v.mean(axis=0);sd=v.std(axis=0,ddof=1)
        for j,ax in enumerate(axes):
            ax.fill_between(x,mean[:,j]-sd[:,j],mean[:,j]+sd[:,j],color=color,alpha=.12,lw=0)
            ax.plot(x,mean[:,j],color=color,ls=ls,lw=1.1,label=label)
    for ax in axes:
        ax.set_xlim(0,15000);ax.set_xticks([0,5000,10000,15000],['0','5,000','10,000','15,000']);ax.set_xlabel('Training episode');ax.grid(alpha=.25,lw=.5)
    axes[0].set_ylabel('Episode return');axes[0].set_title('(a) Training return',loc='left')
    axes[1].set_ylabel('Mean-active SLRR (fixed 50 slots)');axes[1].set_title('(b) Logged training SLRR',loc='left')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.99),ncol=4,frameon=False)
    save(fig,'training_convergence')
    def bars(ax,field,scale,ylabel,title,percent=False):
        upper=0.
        for i,(_,label,color,_,marker) in enumerate(STYLES):
            vals=np.array([r[field] for r in testrows if r['algorithm']==label])*scale
            mean=vals.mean();sd=vals.std(ddof=1)
            upper=max(upper,float(max(vals)),float(mean+sd))
            ax.bar(i,mean,color=color,width=.58,alpha=.9,zorder=2)
            ax.errorbar(i,mean,yerr=sd,fmt='none',ecolor='#333',capsize=2,lw=.8,zorder=3)
            for j,mk in enumerate(['o','s','^']):ax.scatter(i+(j-1)*.10,vals[j],marker=mk,s=14,facecolor='white',edgecolor='#333',lw=.6,zorder=4)
            labeltext=f'{mean:.1f}' if percent or mean>=20 else f'{mean:.3f}'
            ax.annotate(labeltext,(i,max(max(vals),mean+sd)),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(range(4),[x[1] for x in STYLES]);ax.tick_params(axis='x',labelsize=8.5)
        ax.set_ylabel(ylabel);ax.set_title(title,loc='left');ax.set_ylim(0,upper*1.20)
        if percent:ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100])
        ax.grid(axis='y',alpha=.25,lw=.5);ax.set_axisbelow(True)
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.7));fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.93,wspace=.30,hspace=.48)
    specs=[('slrr_sum_per_horizon',1,'Total SLRR per slot','(a) Fixed-horizon total SLRR',False),
        ('delay_executed',1,'Completion delay (s)','(b) Delay on executed slots',False),
        ('survived',100,'Full-horizon completion (%)','(c) Episode completion',True),
        ('offloaded_bits_fraction',100,'Offloaded task bits (%)','(d) Offloading fraction',True)]
    for ax,spec in zip(axes.flat,specs):bars(ax,*spec)
    save(fig,'test_performance')
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.7));fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.93,wspace=.30,hspace=.48)
    for ax,(field,title) in zip(axes.flat,[('energy_fly_j','(a) Flight'),('energy_hover_j','(b) Hovering'),('energy_comp_j','(c) Computation'),('energy_out_j','(d) Total consumption')]):bars(ax,field,.001,'Energy per episode (kJ)',title)
    save(fig,'test_energy_breakdown')
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.8));fig.subplots_adjust(left=.09,right=.98,bottom=.19,top=.84,wspace=.30)
    bars(axes[0],'local_delay_active_mean_s',1,'Local-branch delay (s)','(a) Active-user local branch')
    bars(axes[1],'edge_delay_offloading_mean_s',1,'Edge-branch delay (s)','(b) Offloading-user edge branch')
    save(fig,'test_delay_branches')
    protocol=dict(status='completed',included_algorithms=[x[1] for x in STYLES],training_seeds=[42,43,44],sources=sources,
        exclusion='TD3 seed44 pending authorized retry; Swap seed42 training. Neither mixed with completed original three-seed algorithms.',
        training='Original four-cluster diverse training. 300-episode trailing mean per seed, then equal-seed mean and sample SD. Logged training SLRR is mean-active SLRR over fixed50 slots, NOT total SLRR.',
        testing='Existing protocol-locked natural_id case, all201 common exogenous bank entries per checkpoint, identical across12 checkpoints; includes representative seed per original bank, no outcome selection.',
        seedbank=seedbank,statistics='Mean within each trained checkpoint, then equal mean and sample SD across3 training seeds. Markers are individual training seeds, not test episodes.',
        total_slrr='slrr_sum_per_horizon = sum of total active-user SLRR on executed slots divided by50; early termination has zero future contribution.',
        conditional_metrics='Completion delay and local delay condition on executed active-user slots; edge delay conditions on offloading users. Different populations: do not add local and edge means or interpret their ordering as per-user max ordering.',
        energy='Actual cumulative consumed energy until episode end/termination, without filling unexecuted slots. Lower consumption may reflect early termination; inspect completion rate.',
        timing='No formal uncontended or airborne hardware timing claim.',main_paper_changed=False)
    (OUT/'manifest.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'figure_notes.txt').write_text('\n'.join(str(protocol[k]) for k in ('training','testing','statistics','total_slrr','conditional_metrics','energy','exclusion'))+'\n',encoding='utf-8')
    shutil.copy2(__file__,OUT/Path(__file__).name)
    print(json.dumps({'output':str(OUT),'test_summary':aggregate},ensure_ascii=True))

if __name__=='__main__':main()
