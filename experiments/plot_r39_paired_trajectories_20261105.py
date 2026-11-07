"""Plot existing paired H3 trajectories; no solver/training or source mutation."""
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
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parents[1] / 'results/reviewer_r39_grid_terminal_20261104'
OUT = BASE / 'trajectory_comparison_20261105'
COLORS = ['#1F77B4', '#FF7F0E']

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load_case(m):
    name=f'm{m}_train42_test970000_t0_h3'
    job=next(j for j in read(BASE/'formal_v1/batch_status.json')['jobs'] if j['name']==name)
    case_path=Path(job['cases'])
    case=next(json.loads(s) for s in case_path.read_text(encoding='utf-8').splitlines() if json.loads(s)['case_id']==job['case_id'])
    result_path=Path(job['output'])/'result.json'
    result=read(result_path)
    assert result['valid_reference'] and result['independent_audit']['feasible']
    assert result['baseline_audit']['feasible']
    h=3
    initial=np.asarray(case['initial_positions'])
    paths={
      'DARS':np.array([initial]+[s['positions_after'] for s in case['dars'][:h]]),
      'Oracle':np.array([initial]+[s['positions_after'] for s in result['solution']]),
    }
    batteries={
      'DARS':np.array([case['initial_battery']]+[s['battery_after'] for s in case['dars'][:h]]),
      'Oracle':np.array([case['initial_battery']]+[s['battery_after'] for s in result['independent_audit']['stages']]),
    }
    for method,rows in [('DARS',case['dars'][:h]),('Oracle',result['solution'])]:
        assert len(rows)==h
        for t,row in enumerate(rows):
            assert np.allclose(row['positions_before'],paths[method][t],atol=1e-7,rtol=0)
            assert np.allclose(row['user_positions'],case['stages'][t]['user_positions'],atol=1e-10,rtol=0)
            assert np.allclose(row['tasks'],case['stages'][t]['tasks'],atol=1e-10,rtol=0)
        assert np.max(np.linalg.norm(np.diff(paths[method],axis=0),axis=-1))<=25.000001
    assert abs(sum(s['total_slrr'] for s in case['dars'][:h])-result['dars_total_slrr'])<1e-7
    assert abs(sum(s['total_slrr'] for s in result['independent_audit']['stages'])-result['oracle_total_slrr'])<1e-7
    assert np.allclose(batteries['DARS'][-1],result['terminal_battery_dars'],atol=1e-4,rtol=0)
    assert np.allclose(batteries['Oracle'][-1],result['terminal_battery_oracle'],atol=1e-4,rtol=0)
    assert np.all(batteries['Oracle'][-1]>=batteries['DARS'][-1]-1e-4)
    return dict(m=m,name=name,case=case,result=result,paths=paths,batteries=batteries,
                source_hashes={str(case_path):sha(case_path),str(result_path):sha(result_path)})

def terrain(ax,c,full=True,show_users=True):
    users=np.asarray(c['case']['stages'][0]['user_positions'])
    active=np.asarray(c['case']['stages'][0]['active_mask'],dtype=bool)
    if show_users:
        ax.scatter(*users[~active].T,s=13,color='#C3C7CC',edgecolors='white',linewidths=.25,zorder=2)
        ax.scatter(*users[active].T,s=18,color='#D96A80',edgecolors='white',linewidths=.3,zorder=3)
    towers=np.asarray(c['case']['wct_positions'])
    ax.scatter(*towers.T,s=40,marker='s',color='#38845D',edgecolors='white',linewidths=.6,zorder=4)
    centres=np.array([[125,375],[375,375],[375,125],[125,125]])
    phases=[[0,1],[1,2],[2,3]]
    hot=phases[int(c['case']['stages'][0]['hotspot_phase'])]
    ax.scatter(*centres[hot].T,s=160,facecolors='none',edgecolors='#C89D28',linewidths=1.3,zorder=3)
    if full:
        for i in hot:ax.annotate(f'C{i}',centres[i],xytext=(9,-13),textcoords='offset points',fontsize=8,color='#886709')
    ax.set_aspect('equal');ax.grid(ls=':',lw=.5,color='#D8DDE3',zorder=0)
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    if full:
        ax.set_xlim(0,500);ax.set_ylim(0,500);ax.set_xticks([0,125,250,375,500]);ax.set_yticks([0,125,250,375,500])
    else:
        ax.set_xlim(170,315);ax.set_ylim(175,285);ax.set_xticks([180,220,260,300]);ax.set_yticks([180,210,240,270])

def trajectory(ax,path,method,labels=False):
    ls='-' if method=='DARS' else '--'
    for k,color in enumerate(COLORS):
        q=path[:,k,:]
        ax.plot(q[:,0],q[:,1],color=color,lw=1.6,ls=ls,marker='o',ms=3.2,zorder=6)
        ax.scatter(*q[0],s=42,facecolors='white',edgecolors='#202B38',linewidths=1,zorder=8)
        ax.scatter(*q[-1],s=85,marker='*',color=color,edgecolors='white',linewidths=.35,zorder=9)
        for t in range(len(q)-1):
            a=q[t]+.34*(q[t+1]-q[t]);b=q[t]+.67*(q[t+1]-q[t])
            if np.linalg.norm(b-a)>1e-7:ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=color,lw=.9,mutation_scale=7),zorder=7)
        if labels:
            for t,xy in enumerate(q):
                offset=(-11,6) if k==0 else (5,5)
                ax.annotate(str(t),xy,xytext=offset,textcoords='offset points',fontsize=8,color=color,
                            bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.25),zorder=10)

def save(fig,stem):
    for ext in ['pdf','png','svg']:
        fig.savefig(OUT/f'{stem}.{ext}',dpi=600,facecolor='white')
    plt.close(fig)

def paired(c):
    fig,axs=plt.subplots(2,2,figsize=(7.16,7.0),gridspec_kw={'height_ratios':[1.35,1]})
    fig.subplots_adjust(left=.085,right=.975,top=.865,bottom=.23,wspace=.22,hspace=.37)
    result=c['result'];m=c['m']
    fig.suptitle(f'Paired UAV trajectories: DARS and oracle reference (M = {m})',y=.981,fontsize=12)
    cert='gap-certified' if result['certified_within_requested_gap'] else 'time-limit feasible; not certified'
    fig.text(.5,.946,f'Train seed 42 | Test seed 970000 | Slots 0-2 | Oracle: {cert}',ha='center',fontsize=9)
    fig.text(.5,.922,f"Window total SLRR: DARS {result['dars_total_slrr']:.3f} | Oracle {result['oracle_total_slrr']:.3f}"
             +f" | Verified J gap: {100*result['verified_J_gap']:.3f}%",ha='center',fontsize=9)
    for col,method in enumerate(['DARS','Oracle']):
        for row in [0,1]:
            ax=axs[row,col];terrain(ax,c,full=row==0);trajectory(ax,c['paths'][method],method,labels=row==1)
            ax.set_title(f"({'abcd'[row*2+col]}) {method}"+(' - full area' if row==0 else ' - trajectory detail'),fontsize=10,pad=6)
            if row==0:ax.add_patch(Rectangle((170,175),145,110,fill=False,ec='#7E8996',lw=.7,ls='--'))
    handles=[Line2D([],[],color=COLORS[0],lw=1.6,label='UAV 1'),Line2D([],[],color=COLORS[1],lw=1.6,label='UAV 2'),
      Line2D([],[],marker='o',mfc='white',mec='#202B38',ls='none',label='Start'),Line2D([],[],marker='*',color='#49576A',ls='none',ms=8,label='End'),
      Line2D([],[],marker='o',color='#D96A80',ls='none',ms=4,label='Active UE at slot 0'),Line2D([],[],marker='o',color='#C3C7CC',ls='none',ms=4,label='Inactive UE at slot 0'),
      Line2D([],[],marker='s',color='#38845D',ls='none',ms=5,label='WCT'),Line2D([],[],marker='o',mfc='none',mec='#C89D28',ls='none',ms=8,label='Hot-cluster center')]
    fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.5,.085),frameon=False,columnspacing=1.3,handletextpad=.5,fontsize=8)
    b=c['batteries']
    fig.text(.5,.071,f"Terminal battery (UAV 1 / UAV 2, kJ): DARS {b['DARS'][-1,0]/1000:.3f} / {b['DARS'][-1,1]/1000:.3f}"
             +f"; Oracle {b['Oracle'][-1,0]/1000:.3f} / {b['Oracle'][-1,1]/1000:.3f}",ha='center',fontsize=8.5)
    fig.text(.5,.047,'Numbers 0-3 mark position boundaries; user markers show slot 0 only. Hot clusters: C2/C3.',ha='center',fontsize=8)
    fig.text(.5,.026,'Oracle: clairvoyant 25 m candidate graph + DARS path; paired terminal-energy floors. Not continuous-P1 optimality.',ha='center',fontsize=7.8)
    save(fig,f'oracle_dars_trajectory_m{m}_h3')

def overview(cases):
    fig,axs=plt.subplots(1,3,figsize=(7.16,3.3))
    fig.subplots_adjust(left=.075,right=.985,top=.81,bottom=.27,wspace=.29)
    fig.suptitle('DARS vs. oracle: paired three-slot trajectory details',fontsize=11,y=.98)
    fig.text(.5,.91,'Fixed example selection: train seed 42, test seed 970000, start slot 0',ha='center',fontsize=8.5)
    for ax,c in zip(axs,cases):
        terrain(ax,c,full=False,show_users=False)
        for method in ['DARS','Oracle']:trajectory(ax,c['paths'][method],method)
        r=c['result'];cert='Certified' if r['certified_within_requested_gap'] else 'Not certified'
        ax.set_title(f"M = {c['m']} | {cert}\nSLRR {r['dars_total_slrr']:.2f} / {r['oracle_total_slrr']:.2f}",fontsize=9,pad=5)
        ax.tick_params(labelsize=8);ax.set_xlabel('x (m)',fontsize=9);ax.set_ylabel('y (m)',fontsize=9)
    handles=[Line2D([],[],color=COLORS[k],lw=1.5,label=f'UAV {k+1}') for k in range(2)]
    handles += [Line2D([],[],color='#434D5A',lw=1.5,ls=ls,label=label) for ls,label in [('-','DARS'),('--','Oracle')]]
    handles += [Line2D([],[],color='#434D5A',marker=marker,mfc='white' if marker=='o' else '#434D5A',ls='none',ms=6,label=label) for marker,label in [('o','Start'),('*','End')]]
    fig.legend(handles=handles,ncol=6,loc='lower center',bbox_to_anchor=(.5,.12),frameon=False,fontsize=8,columnspacing=1.2)
    fig.text(.5,.075,'SLRR values: DARS / oracle, accumulated over three slots. Coincident trajectories are retained.',ha='center',fontsize=8)
    fig.text(.5,.026,'Candidate-graph reference with future demand and per-UAV terminal-energy floors; not continuous-P1 global optimum.',ha='center',fontsize=7.6)
    save(fig,'oracle_dars_trajectory_overview')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9.5,'axes.labelsize':9.5,'axes.titlesize':10,
                        'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','axes.linewidth':.65})
    cases=[load_case(m) for m in [20,25,30]]
    metadata={'created_at':datetime.now().astimezone().isoformat(),'selection':'Predetermined first H3 window for each M20/25/30: train42,test970000,start0; no ranking by gain/certification.','verification_status':'ANALYZED','no_new_training_or_solving':True,'curves':'Exact stored boundary coordinates, straight segments, no smoothing or perturbed positions.','user_markers':'Slot-0 user positions and activity only; users change over subsequent slots.','hotspot_geometry_source':'Frozen UAV_env.py: centers at quarter/three-quarter field positions; phase2=C2/C3.','scope':'Illustrative paired windows, not aggregate performance or error-bar estimates. M20/25 models trained5000, M30 trained15000; no cross-M equal-budget claim.','cases':[]}
    rows=[]
    for c in cases:
        paired(c)
        r=c['result'];metadata['cases'].append({'case':c['name'],'valid':r['valid_reference'],'certified':r['certified_within_requested_gap'],'verified_J_gap':r['verified_J_gap'],'dars_total_slrr':r['dars_total_slrr'],'oracle_total_slrr':r['oracle_total_slrr'],'terminal_battery_dars':r['terminal_battery_dars'],'terminal_battery_oracle':r['terminal_battery_oracle'],'source_sha256':c['source_hashes']})
        for method in ['DARS','Oracle']:
            for t in range(4):
                for k in range(2):rows.append({'case':c['name'],'method':method,'boundary':t,'uav':k+1,'x_m':c['paths'][method][t,k,0],'y_m':c['paths'][method][t,k,1],'battery_j':c['batteries'][method][t,k]})
    overview(cases)
    with (OUT/'trajectory_points.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'visual_review_status.json').write_text(json.dumps({'status':'rendered_pending_visual_inspection','pdfs':4,'dpi_png':600},indent=2)+'\n',encoding='utf-8')
    assert all(sha(Path(p))==v for c in cases for p,v in c['source_hashes'].items())
    print(json.dumps({'output':str(OUT),'figures':4,'data_checks':'passed','cases':[(c['name'],c['result']['certified_within_requested_gap']) for c in cases]},ensure_ascii=True))

if __name__=='__main__':main()
