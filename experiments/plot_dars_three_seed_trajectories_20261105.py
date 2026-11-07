"""Plot stored, paired complete trajectories; no training or policy rollout."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'results/reviewer_hotspot_generalization_v2_iofix_20261031'
OUT=BASE/'results/dars_three_seed_trajectories_20261105'
TEST_SEED=700000
COLORS=['#1F77B4','#FF7F0E']
PHASE_COLORS=['#E9D6AC','#C9E2D1','#D2DEF0']
PHASES=[[0,1],[1,2],[2,3]]
CENTERS=np.array([[125,375],[375,375],[375,125],[125,125]])
WCT_LABELS=[3,4,2,1]

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def load():
    cases=[]
    for seed in [42,43,44]:
        path=SOURCE/f'per_sac_cvx_seed{seed}/natural_id.episodes.jsonl'
        with path.open(encoding='utf-8') as f:
            row=next(json.loads(line) for line in f if json.loads(line)['seed']==TEST_SEED)
        prov=read(path.parent/'provenance.json');meta=prov['manifest']
        assert meta['training_seed']==seed and meta['training']['episodes']==15000
        assert meta['system_parameters']['M']==30 and meta['system_parameters']['K']==2
        trace=row['trace'];executed=[t for t in trace if t['executed']]
        assert len(executed)==row['executed_steps']==50 and row['survived']
        assert [t['slot'] for t in executed]==list(range(50))
        q=np.asarray([executed[0]['uav_before']]+[t['uav_after'] for t in executed])
        assert np.allclose(q[:-1],np.asarray([t['uav_before'] for t in executed]),atol=1e-9,rtol=0)
        assert np.max(np.linalg.norm(np.diff(q,axis=0),axis=-1))<=25.000001
        assert np.all((q>=0)&(q<=500))
        assert np.isclose(sum(t['rho_sum'] for t in executed)/50,row['slrr_sum_per_horizon'])
        phase=np.array([t['phase'] for t in executed])
        cases.append(dict(seed=seed,row=row,path=q,phase=phase,source=path,source_sha256=sha(path),provenance_sha256=sha(path.parent/'provenance.json')))
    for c in cases[1:]:
        assert c['row']['bank_entry_sha256']==cases[0]['row']['bank_entry_sha256']
        assert np.array_equal(c['phase'],cases[0]['phase'])
        assert np.allclose(c['path'][0],cases[0]['path'][0],atol=1e-12,rtol=0)
    return cases

def terrain(ax,phase=None,small=False):
    for i,xy in enumerate(CENTERS):
        hot=phase is not None and i in PHASES[phase]
        ax.add_patch(Circle(xy,60,fc=PHASE_COLORS[phase] if hot else 'none',ec='#B8BEC6',lw=.65,ls=':',alpha=.65,zorder=1))
        ax.scatter(*xy,s=25,marker='s',c='#38845D',edgecolors='white',lw=.4,zorder=5)
        ax.annotate(f'C{i+1}/W{WCT_LABELS[i]}',xy,xytext=(0,12),textcoords='offset points',ha='center',fontsize=8 if small else 8.5,color='#3D4952',zorder=7)
    ax.set(xlim=(0,500),ylim=(0,500),aspect='equal',xticks=[0,250,500],yticks=[0,250,500])
    ax.grid(color='#E1E5E9',ls=':',lw=.5,zorder=0)
    ax.tick_params(labelsize=9,pad=2)
    for spine in ax.spines.values():spine.set_color('#73808A');spine.set_linewidth(.6)

def paths(ax,q,start,end,mark_switch=False):
    for k,color in enumerate(COLORS):
        v=q[start:end+1,k]
        ax.plot(*v.T,color=color,lw=1.35 if end-start>20 else 1.55,zorder=4)
        ax.scatter(*v[0],s=27,facecolor='white',edgecolor=color,lw=1,zorder=8)
        ax.scatter(*v[-1],s=48,marker='*',color=color,edgecolor='white',lw=.4,zorder=9)
        for j in range(2,len(v)-1,8):
            a=v[j-1];b=v[j]
            if np.linalg.norm(b-a)>1:
                ax.annotate('',xy=a+.8*(b-a),xytext=a+.2*(b-a),arrowprops=dict(arrowstyle='->',color=color,lw=.85,mutation_scale=8),zorder=6)
        if mark_switch:
            for t in [16,32,46]:ax.scatter(*q[t,k],s=16,marker='D',facecolor=color,edgecolor='white',lw=.4,zorder=7)

def handles():
    return [Line2D([],[],color=COLORS[k],lw=1.6,label=f'UAV {k+1}') for k in range(2)]+[
        Line2D([],[],marker='s',color='#38845D',ls='none',ms=5,label='WCT / cluster center'),
        Line2D([],[],marker='o',mfc='white',mec='#4A5663',ls='none',ms=5,label='Start'),
        Line2D([],[],marker='*',color='#4A5663',ls='none',ms=7,label='End')]

def save(fig,name):
    for ext in ['pdf','png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=600,facecolor='white')
    fig.savefig(OUT/f'{name}_preview.png',dpi=160,facecolor='white')
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path'})
    cases=load();starts=[0,16,32,46];ends=[16,32,46,50];phases=[2,0,1,2]
    fig,axs=plt.subplots(1,3,figsize=(7.16,4.1))
    fig.subplots_adjust(left=.075,right=.98,bottom=.38,top=.84,wspace=.24)
    for ax,c in zip(axs,cases):
        terrain(ax);paths(ax,c['path'],0,50,True)
        ax.set_title(f"Training seed {c['seed']}",fontsize=10,pad=8)
        ax.set_xlabel('x (m)',labelpad=2)
    axs[0].set_ylabel('y (m)',labelpad=2)
    fig.text(.5,.966,'DARS trajectories across three training seeds',ha='center',fontsize=12)
    fig.text(.5,.907,'M = 30 | 4 WCTs | 15,000 training episodes | Common test seed 700000',ha='center',fontsize=9)
    axbar=fig.add_axes([.12,.19,.80,.064])
    for a,b,p in zip(starts,ends,phases):
        axbar.add_patch(Rectangle((a,0),b-a,1,fc=PHASE_COLORS[p],ec='white',lw=1))
        label='/'.join(f'C{i+1}' for i in PHASES[p])
        axbar.text((a+b)/2,.5,label if b-a>5 else 'C',ha='center',va='center',fontsize=8.5)
    axbar.set(xlim=(0,50),ylim=(0,1),yticks=[],xticks=[0,16,32,46,50])
    axbar.tick_params(axis='x',labelsize=8.5,length=2,pad=2)
    for sp in axbar.spines.values():sp.set_visible(False)
    fig.text(.5,.108,'Elapsed slot boundaries; switches before slots 17, 33 and 47. Final C = {C3, C4}.',ha='center',fontsize=8)
    h=handles()+[Line2D([],[],marker='D',color='#4A5663',ls='none',ms=4,label='Hotspot switch')]
    fig.legend(handles=h,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.005),frameon=False,fontsize=8.5,columnspacing=1.4)
    save(fig,'dars_three_seed_trajectory_overview')
    fig,axs=plt.subplots(3,4,figsize=(7.16,7.3))
    fig.subplots_adjust(left=.115,right=.98,bottom=.115,top=.86,wspace=.32,hspace=.29)
    for row,c in enumerate(cases):
        for col,(a,b,p) in enumerate(zip(starts,ends,phases)):
            ax=axs[row,col];terrain(ax,p,True);paths(ax,c['path'],a,b)
            if row==0:ax.set_title(f"Slots {a+1}-{b}\nHot: "+'/'.join(f'C{i+1}' for i in PHASES[p]),fontsize=9.5,pad=8)
            if col==0:ax.set_ylabel(f"Seed {c['seed']}\ny (m)",fontsize=9.5)
            if row==2:ax.set_xlabel('x (m)',fontsize=9.5)
    fig.text(.5,.974,'DARS trajectories by hotspot stage',ha='center',fontsize=12)
    fig.text(.5,.938,'Same test episode for all seeds | M = 30 | 50 slots | 15,000 training episodes',ha='center',fontsize=9)
    fig.legend(handles=handles(),ncol=5,loc='lower center',bbox_to_anchor=(.5,.044),frameon=False,fontsize=8.2,columnspacing=1.0)
    fig.text(.5,.028,'Shaded circles identify hot clusters; dashed circles have radius 60 m (one coordinate standard deviation).',ha='center',fontsize=8)
    fig.text(.5,.009,'Circles are not coverage boundaries. Start/end markers refer to each panel; paths use exact stored coordinates.',ha='center',fontsize=8)
    save(fig,'dars_three_seed_trajectory_stages')
    points=[]
    for c in cases:
        for t,positions in enumerate(c['path']):
            for k,xy in enumerate(positions):points.append(dict(training_seed=c['seed'],test_seed=TEST_SEED,boundary=t,uav=k+1,x_m=xy[0],y_m=xy[1]))
    with (OUT/'trajectory_points.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    dump(OUT/'selected_traces.json',dict(test_seed=TEST_SEED,models=[dict(training_seed=c['seed'],trace=c['row']['trace']) for c in cases]))
    dump(OUT/'verification.json',dict(created_at=datetime.now().astimezone().isoformat(),selection='First natural_id test episode700000, selected before examining outcomes; all3trainingseeds retained.',training_episodes=15000,M=30,K=2,test_seed=TEST_SEED,paired_bank_sha256=cases[0]['row']['bank_entry_sha256'],same_initial_positions=True,same_hotspot_schedule=True,continuity_and_25m_step_bound=True,all_completed_50_slots=True,no_new_training_or_rollouts=True,no_smoothing_or_coordinate_perturbations=True,visual_review='pending',switches_before_one_based_slots=[17,33,47],sigma_c_m=60,example_not_aggregate_evidence=True,cases=[dict(seed=c['seed'],source=str(c['source']),source_sha256=c['source_sha256'],provenance_sha256=c['provenance_sha256'],mean_total_slrr_fixed_horizon=c['row']['slrr_sum_per_horizon']) for c in cases]))
    print(json.dumps(dict(output=str(OUT),paired=True,plots=2),ensure_ascii=True))

if __name__=='__main__':main()
