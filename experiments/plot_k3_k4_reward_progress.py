"""Read-only snapshot of ongoing K3/K4 training returns."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.ticker import StrMethodFormatter

ROOT=Path(__file__).resolve().parents[1]/'results/reviewer_uav_scaling_k3_k4_seed42_20261102'
WINDOW=300

def main():
    stamp=datetime.now().astimezone();out=ROOT/('figures_reward_progress_'+stamp.strftime('%Y%m%d_%H%M%S'));out.mkdir()
    data={};provenance=[]
    for k in (3,4):
        p=ROOT/f'k{k}/source_snapshot/results/formal_dars_k{k}_seed42/per_sac_cvx.train.jsonl'
        b=p.read_bytes();end=b.rfind(b'\n')+1;b=b[:end]
        rows=[json.loads(x) for x in b.splitlines()];y=np.array([r['reward'] for r in rows]);assert np.isfinite(y).all()
        assert all(r['horizon']==50 for r in rows)
        data[k]=y
        provenance.append(dict(K=k,source=str(p),complete_records=len(rows),snapshot_sha256=hashlib.sha256(b).hexdigest(),last300_reward=float(y[-300:].mean()),last300_survival=sum(r['survived'] for r in rows[-300:])/min(300,len(rows))))
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9.5,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
        'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    fig,ax=plt.subplots(figsize=(7.16,3.45));fig.subplots_adjust(left=.10,right=.97,bottom=.18,top=.78)
    with (out/'reward_snapshot.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['K','training_seed','episode','episode_return','trailing_mean_300'])
        for k,color,ls in [(3,'#D62728','-'),(4,'#1F77B4','--')]:
            y=data[k];x=np.arange(1,len(y)+1);smooth=np.convolve(y,np.ones(WINDOW)/WINDOW,mode='valid');sx=np.arange(WINDOW,len(y)+1)
            ax.plot(x,y,color=color,alpha=.10,lw=.35,rasterized=True)
            ax.plot(sx,smooth,color=color,ls=ls,lw=1.6,label=f'{k} UAVs (episode {len(y):,})')
            ax.scatter(sx[-1],smooth[-1],color=color,s=20,zorder=4)
            for i,val in enumerate(y):writer.writerow([k,42,i+1,val,smooth[i-WINDOW+1] if i>=WINDOW-1 else ''])
    ax.set_xlabel('Training episode');ax.set_ylabel('Episode return');ax.set_xlim(0,max(map(len,data.values()))+100)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'));ax.set_xticks(np.arange(0,max(map(len,data.values()))+1,1000));ax.grid(alpha=.22,lw=.5)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.98),ncol=2,frameon=False)
    fig.text(.52,.855,'Seed 42; pale lines: raw rewards; bold lines: trailing mean over 300 episodes',ha='center',fontsize=9)
    for ext in ('pdf','png','svg'):fig.savefig(out/f'k3_k4_reward.{ext}',dpi=600)
    plt.close(fig)
    summary=dict(created_at=stamp.isoformat(),metric='Logged episode return, including original mean-active-rho reward and battery penalty; not physical totalSLRR.',smoothing='Trailing mean300, valid windows only, no padding or extrapolation.',scope='Ongoing training, one seed42 per K; not final held-out performance. No between-seed uncertainty.',sources=provenance)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');shutil.copy2(__file__,out/Path(__file__).name)
    print(json.dumps(dict(output=str(out),summary=summary),ensure_ascii=True))

if __name__=='__main__':main()
