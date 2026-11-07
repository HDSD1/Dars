"""Describe all valid existing R3.9 paired windows without new solver runs."""
from pathlib import Path
from datetime import datetime
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.lines import Line2D

BASE=Path(__file__).resolve().parents[1]/'results/reviewer_r39_grid_terminal_20261104'
OUT=BASE/'slrr_comparison_20261105'

def save(fig,name):
    for ext in ['pdf','png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=600,facecolor='white')
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=BASE/'formal_v1/completed_results.json'
    original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    jobs=json.loads(source.read_text(encoding='utf-8'))
    assert len(jobs)==300
    valid=[];excluded=[];hashes={str(source):original_hash}
    for j in jobs:
        p=Path(j['output'])/'result.json';raw=json.loads(p.read_text(encoding='utf-8'));hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        assert raw['valid_reference']==j['valid_reference']
        assert raw['certified_within_requested_gap']==j['certified']
        if not raw['valid_reference']:
            excluded.append(j['name']);continue
        assert raw['independent_audit']['feasible']
        assert abs(raw['oracle_total_slrr']-j['reference_slrr'])<1e-8
        assert abs(raw['dars_total_slrr']-j['dars_slrr'])<1e-8
        valid.append({k:j[k] for k in ['name','horizon','network_users','training_seed','test_seed','start_slot','low_battery','sampling_weight','certified','solver_status','dars_slrr','reference_slrr','reference_upper','verified_J_gap']})
    assert len(valid)==299 and len(excluded)==1
    by_seed=[];aggregates=[];strata=[]
    for h in [3,5]:
        for m in [20,25,30]:
            rows=[r for r in valid if r['horizon']==h and r['network_users']==m]
            ss=[]
            for seed in sorted(set(r['training_seed'] for r in rows)):
                a=[r for r in rows if r['training_seed']==seed];w=np.array([r['sampling_weight'] for r in a]);entry={'horizon':h,'users':m,'training_seed':seed,'valid_windows':len(a),'weight_sum':float(w.sum())}
                for method,field in [('dars','dars_slrr'),('oracle','reference_slrr')]:entry[method+'_mean']=float(np.average([r[field] for r in a],weights=w))
                by_seed.append(entry);ss.append(entry)
                for low in [False,True]:
                    sub=[r for r in a if r['low_battery']==low]
                    if sub:
                        weights=[r['sampling_weight'] for r in sub]
                        strata.append({'horizon':h,'users':m,'training_seed':seed,'low_battery':low,'valid_windows':len(sub),'weight_sum':sum(weights),'dars_mean':float(np.average([r['dars_slrr'] for r in sub],weights=weights)),'oracle_mean':float(np.average([r['reference_slrr'] for r in sub],weights=weights))})
            row={'horizon':h,'users':m,'training_seeds':len(ss),'valid_windows':len(rows),'certified_windows':sum(r['certified'] for r in rows)}
            for method in ['dars','oracle']:
                vals=[s[method+'_mean'] for s in ss];row[method+'_mean']=float(np.mean(vals));row[method+'_seed_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
            row['relative_gain_of_means_percent']=100*(row['oracle_mean']/row['dars_mean']-1)
            aggregates.append(row)
    for name,rows in [('paired_windows',valid),('per_training_seed',by_seed),('slrr_summary',aggregates),('low_battery_strata',strata)]:
        with (OUT/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9.5,'axes.labelsize':9.5,'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})
    fig,axs=plt.subplots(1,2,figsize=(7.16,3.95));fig.subplots_adjust(left=.085,right=.98,top=.79,bottom=.34,wspace=.25)
    fig.suptitle('Window total SLRR: DARS vs. oracle feasible reference',fontsize=11,y=.975)
    fig.text(.5,.916,'Paired starts and future demand; per-UAV terminal battery no lower than paired DARS',ha='center',fontsize=8.6)
    handles=[]
    for ax,h in zip(axs,[3,5]):
        a=[r for r in aggregates if r['horizon']==h];x=np.arange(3)
        for method,off,color,label in [('dars',-.18,'#D62728','DARS'),('oracle',.18,'#1F77B4','Oracle feasible reference')]:
            y=[r[method+'_mean'] for r in a];bars=ax.bar(x+off,y,.34,color=color,label=label,zorder=3)
            if h==3:handles.append(bars)
            for xx,yy,r in zip(x+off,y,a):
                sd=r[method+'_seed_sd']
                if sd is not None:ax.errorbar(xx,yy,yerr=sd,fmt='none',ecolor='#273746',elinewidth=.8,capsize=2.5,zorder=4)
                ax.text(xx,yy+(sd or 0)+.55,f'{yy:.2f}',ha='center',va='bottom',fontsize=8)
        ax.set_xticks(x,['20','25','30']);ax.set_xlabel('Number of users, M');ax.set_ylabel('Cumulative total SLRR per window')
        ax.set_ylim(0,38);ax.set_yticks([0,10,20,30]);ax.grid(axis='y',ls=':',lw=.55,color='#D8DDE3',zorder=0)
        ax.set_title(f"({'a' if h==3 else 'b'}) H = {h}: {'249 valid; 90 certified' if h==3 else '50 valid; 0 certified'}",pad=7)
    fig.legend(handles,['DARS','Oracle feasible reference'],loc='lower center',bbox_to_anchor=(.5,.135),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.105,'M20/M25: one training seed (no seed error bar); M30: mean and sample SD across three training seeds.',ha='center',fontsize=7.8)
    fig.text(.5,.073,'H5 uses the prespecified sampling weights within each seed; seeds are then equally weighted.',ha='center',fontsize=8)
    fig.text(.5,.040,'All H5 references equal the DARS incumbent after the time limit: equality does not establish optimality.',ha='center',fontsize=8)
    fig.text(.5,.012,'Finite candidate graph with future information; one invalid H3 result excluded. H3 and H5 are distinct samples.',ha='center',fontsize=7.6)
    save(fig,'oracle_dars_slrr_summary')
    fig,axs=plt.subplots(1,2,figsize=(7.16,4.05));fig.subplots_adjust(left=.085,right=.98,top=.81,bottom=.34,wspace=.29)
    fig.suptitle('Paired window SLRR: every valid reference',fontsize=11,y=.98)
    fig.text(.5,.92,'Filled: gap-certified reference | Hollow: feasible reference without certification',ha='center',fontsize=8.6)
    palette={20:'#FF7F0E',25:'#2CA02C',30:'#1F77B4'}
    for ax,h in zip(axs,[3,5]):
        rr=[r for r in valid if r['horizon']==h]
        for m,color in palette.items():
            for cert in [False,True]:
                a=[r for r in rr if r['network_users']==m and r['certified']==cert]
                ax.scatter([r['dars_slrr'] for r in a],[r['reference_slrr'] for r in a],s=17,facecolors=color if cert else 'none',edgecolors=color,linewidths=.7,alpha=.75,zorder=3)
        limit=30 if h==3 else 50
        ax.plot([0,limit],[0,limit],color='#697586',ls='--',lw=.9,zorder=1)
        ax.set_xlim(0,limit);ax.set_ylim(0,limit);ax.set_aspect('equal');ax.grid(ls=':',lw=.5,color='#D8DDE3')
        ax.set_xlabel('DARS cumulative total SLRR');ax.set_ylabel('Oracle-reference cumulative total SLRR')
        improved=sum(r['reference_slrr']>r['dars_slrr']+1e-6 for r in rr)
        ax.set_title(f"({'a' if h==3 else 'b'}) H = {h}: {improved}/{len(rr)} improved",pad=7)
    hh=[Line2D([],[],marker='o',color=color,ls='none',ms=5,label=f'M = {m}') for m,color in palette.items()]
    hh.append(Line2D([],[],color='#697586',ls='--',label='Equality'))
    fig.legend(handles=hh,loc='lower center',bbox_to_anchor=(.5,.12),ncol=4,frameon=False,fontsize=8.5)
    fig.text(.5,.086,'One point per valid window; overlapping points are retained. Windows from the same episode are correlated.',ha='center',fontsize=8)
    fig.text(.5,.049,'Certification: verified relative J gap <= 0.1% within the specified candidate graph, not continuous P1.',ha='center',fontsize=8)
    fig.text(.5,.015,'H5 points on the diagonal retain the warm-start value; none is certified optimal.',ha='center',fontsize=8)
    save(fig,'oracle_dars_slrr_paired')
    metadata={'created_at':datetime.now().astimezone().isoformat(),'status':'ANALYZED','total_windows':300,'valid_windows':299,'excluded':excluded,'certified_references':90,'metric':'Sum of physical total SLRR over H slots; neither training reward nor fixed-50-slot mean.','aggregation':'Within each training seed use protocol sampling weights, then equal-weight training seeds. H3 weights1. H5 includes oversampled low-battery windows; strata supplied separately.','errorbars':'Sample SD across training-seed means only for M30 (3 seeds). M20/25 only seed42, no error bar. No inferential p values or window-independent confidence intervals.','budgets':'M20/25 models trained5000, M30 models15000. Oracle comparisons are paired within model; not matched-budget comparison across M.','limitations':'Only90 references certified in transformed J within candidate graph. Feasible incumbent is not exact optimum. One invalid H3 excluded from paired means, not hidden. H3 and H5 different horizons and samples.','no_training_or_solving':True,'source_sha256':hashes,'summary':aggregates}
    (OUT/'provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'visual_review_status.json').write_text(json.dumps({'status':'rendered_pending_visual_inspection','figures':2},indent=2)+'\n',encoding='utf-8')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    print(json.dumps({'output':str(OUT),'summary':aggregates},ensure_ascii=True))

if __name__=='__main__':main()
