"""Full paired results and figures for the prespecified WCT offset layout."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.patches import Circle,Patch
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
ALGOS=['per_sac_cvx','esac_per_cvx','sac_cvx','td3_cvx','maddpg_cvx']
LABELS=['DARS','ESAC-PER','SAC','TD3','MADDPG']


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def estimate(x,boot):
    return {'mean':float(x),'ci95':np.percentile(boot,[2.5,97.5]).tolist()}


def wilson(p,n):
    z=1.959963984540054;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def run(root):
    status=read(root/'batch_status.json');assert status['status']=='completed'
    protocol=read(root/'protocol.json')
    out=root/'figures';out.mkdir(exist_ok=False)
    idx=np.random.default_rng(20261105).integers(0,100,size=(10000,100))
    summaries=[];paired=[];samples={};source_files=[];external=None
    for algo,name in zip(ALGOS,LABELS):
        manifest=read(root/'evaluation'/algo/'manifest.json')
        assert manifest['status']=='completed' and manifest['model_unchanged']
        b={};v={}
        for layout in ('colocated','offset'):
            path=root/'evaluation'/algo/layout/'episodes.jsonl'
            raw=path.read_bytes();rows=sorted([json.loads(l) for l in raw.decode('utf-8').splitlines()],key=lambda r:r['seed'])
            assert len(rows)==100 and [r['seed'] for r in rows]==list(range(931000,931100))
            exo=[r['external_state_hashes'] for r in rows]
            if external is None:external=exo
            else:assert external==exo
            assert all(r['energy_violation_steps']==0 for r in rows)
            fields=['cumulative_total_slrr','min_battery_j','energy_out_j','energy_in_j',
                    'energy_charge_accepted_j','energy_fly_j','energy_hover_j','energy_comp_j','executed_steps','survived']
            a={f:np.array([r[f] for r in rows],dtype=float) for f in fields}
            np.testing.assert_allclose(a['energy_out_j'],a['energy_fly_j']+a['energy_hover_j']+a['energy_comp_j'],atol=1e-6)
            bs={f:values[idx].mean(axis=1) for f,values in a.items()}
            metrics={
                'slrr_per_planned_slot':estimate(a['cumulative_total_slrr'].mean()/50,bs['cumulative_total_slrr']/50),
                'cumulative_total_slrr':estimate(a['cumulative_total_slrr'].mean(),bs['cumulative_total_slrr']),
                'min_battery_kj':estimate(a['min_battery_j'].mean()/1000,bs['min_battery_j']/1000),
                'energy_kj':estimate(a['energy_out_j'].mean()/1000,bs['energy_out_j']/1000),
                'harvested_energy_kj':estimate(a['energy_in_j'].mean()/1000,bs['energy_in_j']/1000),
                'accepted_harvest_kj':estimate(a['energy_charge_accepted_j'].mean()/1000,bs['energy_charge_accepted_j']/1000),
                'slrr_per_kj':estimate(a['cumulative_total_slrr'].sum()/(a['energy_out_j'].sum()/1000),bs['cumulative_total_slrr']/(bs['energy_out_j']/1000)),
                'executed_slots':estimate(a['executed_steps'].mean(),bs['executed_steps']),
                'survival_probability':{'mean':float(a['survived'].mean()),'ci95':wilson(a['survived'].mean(),len(rows))},
            }
            summary={'algorithm':algo,'label':name,'layout':layout,'episodes':100,
                'complete_episodes':int(a['survived'].sum()),'energy_violation_steps':0,**metrics,
                'minimum_battery_worst_kj':float(a['min_battery_j'].min()/1000),
                'energy_components_mean_kj':{f:float(a[f].mean()/1000) for f in ('energy_fly_j','energy_hover_j','energy_comp_j')},
                'termination_reasons':{r:sum(x['termination_reason']==r for x in rows) for r in sorted({x['termination_reason'] for x in rows})}}
            summaries.append(summary);samples[algo,layout]=rows;v[layout]=a;b[layout]=bs
            source_files.append({'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest()})
        paired.append({'algorithm':algo,'label':name,
            'slrr_change_percent':estimate(100*(v['offset']['cumulative_total_slrr'].mean()/v['colocated']['cumulative_total_slrr'].mean()-1),100*(b['offset']['cumulative_total_slrr']/b['colocated']['cumulative_total_slrr']-1)),
            'survival_change_percentage_points':estimate(100*(v['offset']['survived'].mean()-v['colocated']['survived'].mean()),100*(b['offset']['survived']-b['colocated']['survived'])),
            'min_battery_change_kj':estimate((v['offset']['min_battery_j'].mean()-v['colocated']['min_battery_j'].mean())/1000,(b['offset']['min_battery_j']-b['colocated']['min_battery_j'])/1000),
            'slrr_per_kj_change_percent':estimate(100*((v['offset']['cumulative_total_slrr'].sum()/v['offset']['energy_out_j'].sum())/(v['colocated']['cumulative_total_slrr'].sum()/v['colocated']['energy_out_j'].sum())-1),100*((b['offset']['cumulative_total_slrr']/b['offset']['energy_out_j'])/(b['colocated']['cumulative_total_slrr']/b['colocated']['energy_out_j'])-1))})
    report={'status':'completed','protocol':str(root/'protocol.json'),'summaries':summaries,'paired_layout_effects':paired,
        'source_files':source_files,'verified_matching_exogenous_sequences_all_algorithms_layouts':True,
        'statistics':'All100 paired episodes/model/layout. Mean and ratio percentile95% intervals from10000 paired episode resamples; survival Wilson95%. Single training seed42. No simultaneous inference or training-seed uncertainty.',
        'scope':'Frozen transfer to the single prespecified offset layout; no retraining. All five methods trained15000episodes in original co-located geometry. Original and offset evaluated afresh on CPU. Energy stops at actual termination; totalSLRR divided by50planned slots.',
        'example_selection':'First prespecified test seed931000, not chosen by performance.'}
    (root/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':10,
        'axes.titlesize':10.5,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
        'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.75})
    def save(fig,name):
        fig.savefig(out/f'{name}.pdf');fig.savefig(out/f'{name}.svg')
        fig.savefig(out/f'{name}.png',dpi=400);fig.savefig(out/f'{name}_preview.png',dpi=190);plt.close(fig)
    radius=math.sqrt(100**2-30**2);centers=np.asarray(protocol['cluster_centers'])
    def geography(ax,layout,labels=True):
        towers=np.asarray(protocol['layouts'][layout])
        for x,y in towers:ax.add_patch(Circle((x,y),radius,facecolor='#dbeaf5',edgecolor='#7397b5',alpha=.55,lw=.8))
        ax.scatter(towers[:,0],towers[:,1],marker='^',s=60,facecolor='white',edgecolor='#285f87',linewidth=1.5,zorder=5)
        ax.scatter(centers[:,0],centers[:,1],marker='x',s=38,color='#222222',linewidth=1.4,zorder=6)
        if labels:
            for i,(x,y) in enumerate(centers):ax.text(x+13,y+13,f'C{i}',fontsize=9,zorder=7)
        ax.set(xlim=(0,500),ylim=(0,500),xlabel='x (m)',ylabel='y (m)',aspect='equal')
        ax.set_xticks([0,125,250,375,500]);ax.set_yticks([0,125,250,375,500]);ax.grid(alpha=.18,lw=.5)
    handles=[Line2D([],[],marker='^',mfc='white',mec='#285f87',ls='none',label='WCT'),
             Line2D([],[],marker='x',color='#222222',ls='none',label='User-cluster center'),
             Patch(facecolor='#dbeaf5',edgecolor='#7397b5',label='Charging footprint')]
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.8));fig.subplots_adjust(left=.08,right=.98,bottom=.19,top=.80,wspace=.28)
    for ax,layout,title in zip(axes,('colocated','offset'),('(a) Co-located layout','(b) Offset layout')):
        geography(ax,layout);ax.set_title(title,loc='left')
    fig.legend(handles=handles,ncol=3,loc='upper center',frameon=False,bbox_to_anchor=(.52,.98))
    fig.text(.52,.055,'Horizontal charging radius: 95.4 m. Nearest WCT from each cluster: 0 m / 127.5 m.',ha='center',fontsize=8.5)
    save(fig,'layout_geometry')

    fig,axes=plt.subplots(2,2,figsize=(7.3,5.6));fig.subplots_adjust(left=.10,right=.98,bottom=.16,top=.90,wspace=.30,hspace=.51)
    spec=[('slrr_per_planned_slot','Total SLRR per planned slot','(a) Service performance',1),
          ('survival_probability','Complete episodes (%)','(b) 50-slot completion',100),
          ('min_battery_kj','Mean minimum battery (kJ)','(c) Battery margin',1),
          ('slrr_per_kj','Total SLRR per energy (kJ$^{-1}$)','(d) Energy efficiency',1)]
    for ax,(metric,ylabel,title,scale) in zip(axes.flat,spec):
        x=np.arange(5);upper=[]
        for shift,layout,color,label in [(-.18,'colocated','#739bb9','Co-located'),(.18,'offset','#e89551','Offset')]:
            rr=[next(r for r in summaries if r['algorithm']==a and r['layout']==layout)[metric] for a in ALGOS]
            y=np.array([r['mean'] for r in rr])*scale;ci=np.array([r['ci95'] for r in rr])*scale;upper.extend(ci[:,1])
            ax.bar(x+shift,y,width=.34,color=color,label=label,zorder=3,
                yerr=np.maximum(0,np.array([y-ci[:,0],ci[:,1]-y])),error_kw={'elinewidth':.7,'capsize':2,'capthick':.7})
        ax.set_xticks(x,LABELS,rotation=15);ax.set_ylabel(ylabel);ax.set_title(title,loc='left')
        ax.set_ylim(0,110 if metric=='survival_probability' else max(upper)*1.15)
        ax.grid(axis='y',alpha=.2,lw=.5,zorder=0)
    fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=2,loc='upper center',frameon=False)
    fig.text(.54,.055,'Frozen models; K = 2, M = 30; 100 paired test episodes; training seed 42.',ha='center',fontsize=8.5)
    fig.text(.54,.019,'95% intervals; all early terminations included; only WCT positions change.',ha='center',fontsize=8.5)
    save(fig,'paired_layout_metrics')

    fig,axes=plt.subplots(2,2,figsize=(7.3,6.6));fig.subplots_adjust(left=.10,right=.98,bottom=.13,top=.90,wspace=.30,hspace=.43)
    for col,layout in enumerate(('colocated','offset')):
        r=samples['per_sac_cvx',layout][0];traj=np.asarray(r['uav_trajectory'])
        geography(axes[0,col],layout)
        axes[0,col].set_title('(a) Co-located' if col==0 else '(b) Offset',loc='left')
        for k,color in enumerate(('#0067b1','#db581a')):
            axes[0,col].plot(traj[:,k,0],traj[:,k,1],color=color,lw=1.4,zorder=8,label=f'UAV {k+1}')
            axes[0,col].scatter(*traj[0,k],color=color,s=26,marker='o',zorder=9)
            axes[0,col].scatter(*traj[-1,k],color=color,s=45,marker='s',zorder=9)
        battery=np.vstack([r['initial_battery_j']]+[s['battery_after_service_j'] for s in r['slots'] if s['executed']])/1000
        for k,color in enumerate(('#0067b1','#db581a')):
            axes[1,col].plot(np.arange(len(battery)),battery[:,k],color=color,lw=1.35)
        axes[1,col].set(xlim=(0,50),ylim=(0,42),xlabel='Executed time slot',ylabel='Battery energy (kJ)')
        axes[1,col].grid(alpha=.2,lw=.5)
        axes[1,col].set_title(f'Completed slots: {r["executed_steps"]}/50',loc='left')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
    fig.text(.54,.046,'DARS, training seed 42; first prespecified test seed 931000; no outcome selection.',ha='center',fontsize=8.3)
    save(fig,'dars_first_paired_episode')
    print(json.dumps({'summary':str(root/'summary.json'),'paired_layout_effects':paired},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'results/reviewer_wct_offset_seed42_20261105')
    args=p.parse_args();run(args.root.resolve())
