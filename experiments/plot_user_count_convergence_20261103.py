"""Actual training returns, one panel per algorithm, all authorized user counts."""
from pathlib import Path
import json
import hashlib
import csv
import subprocess
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from run_user_sweep_5000_20261102 import OUT, ALGOS, result

DEST=OUT/'convergence_by_algorithm_20261103'
STYLES={'per_sac_cvx':('DARS','#D62728'),'esac_per_cvx':('ESAC-PER','#1F77B4'),
        'sac_cvx':('SAC','#2CA02C'),'td3_cvx':('TD3','#9467BD'),'maddpg_cvx':('MADDPG','#FF7F0E')}
LINES={20:(':','^'),25:('--','s'),30:('-','o')}
WINDOW=300

def main():
    DEST.mkdir(exist_ok=False)
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':9.5,'axes.titlesize':10,
        'legend.fontsize':8.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    data={};sources=[]
    with (DEST/'training_curves.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['algorithm','M','training_seed','episode','raw_return','trailing_mean_300'])
        for algo in ALGOS:
            for m in (20,25,30):
                folder=result(m,algo);p=folder/f'{algo}.train.jsonl';b=p.read_bytes();rows=[json.loads(l) for l in b.splitlines() if l.strip()]
                meta=json.loads((folder/f'{algo}.manifest.json').read_text(encoding='utf-8'));assert meta['status']=='completed' and meta['training_seed']==42
                assert len(rows)==(15000 if m==30 else 5000)
                y=np.asarray([r['reward'] for r in rows]);assert np.isfinite(y).all()
                smooth=np.convolve(y,np.ones(WINDOW)/WINDOW,'valid');x=np.arange(WINDOW,len(y)+1)
                data[algo,m]=(y,x,smooth)
                sources.append(dict(algorithm=algo,M=m,episodes=len(y),sha256=hashlib.sha256(b).hexdigest(),path=str(p),last300_mean=float(y[-300:].mean())))
                for ep,val in enumerate(y,1):w.writerow([algo,m,42,ep,val,smooth[ep-WINDOW] if ep>=WINDOW else ''])
    def draw(ax,algo):
        name,color=STYLES[algo]
        for m in (20,25,30):
            y,x,sm=data[algo,m];ls,marker=LINES[m]
            ax.plot(np.arange(1,len(y)+1),y,color=color,alpha=.055,lw=.25,rasterized=True)
            ax.plot(x,sm,color=color,ls=ls,lw=1.35,marker=marker,markevery=1500,markersize=3.4,
                    markerfacecolor='white' if m!=30 else color,markeredgewidth=.8,label=f'{m} users')
            ax.plot(x[-1],sm[-1],marker=marker,color=color,ms=4,markerfacecolor='white' if m!=30 else color)
        ax.set_xlim(0,15300);ax.set_xticks([0,5000,10000,15000]);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p: '0' if x==0 else f'{x/1000:g}k'))
        ax.set(xlabel='Training episode',ylabel='Episode return',title=name);ax.grid(alpha=.22,lw=.5)
    def save(fig,name):
        for ext in ('pdf','png','svg'):fig.savefig(DEST/f'{name}.{ext}',dpi=600)
        plt.close(fig)
    for algo in ALGOS:
        fig,ax=plt.subplots(figsize=(3.5,2.8));fig.subplots_adjust(left=.18,right=.97,bottom=.29,top=.86)
        draw(ax,algo);ax.legend(loc='lower right',frameon=True,facecolor='white',edgecolor='none',fontsize=8)
        fig.text(.55,.045,'Seed 42; 300-episode trailing mean',ha='center',fontsize=8)
        save(fig,f'convergence_{algo}')
    fig,axes=plt.subplots(3,2,figsize=(7.16,7.15));fig.subplots_adjust(left=.095,right=.98,bottom=.10,top=.91,hspace=.66,wspace=.29)
    for ax,algo in zip(axes.flat,ALGOS):draw(ax,algo)
    axes[2,1].axis('off')
    handles=[Line2D([0],[0],color='#333333',ls=LINES[m][0],marker=LINES[m][1],markerfacecolor='white' if m!=30 else '#333333',lw=1.35,ms=4,label=f'{m} users') for m in (20,25,30)]
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.54,.985),fontsize=9.5)
    axes[2,1].text(.06,.80,'One training seed (42) per setting\n\n20 / 25 users: 5,000 episodes\n30 users: 15,000 episodes\n\nBold lines: trailing mean (300)\nPale traces: raw training return\n\nNo extrapolation beyond training.',transform=axes[2,1].transAxes,fontsize=9,va='top',linespacing=1.25)
    fig.text(.535,.025,'Training return includes the original reward and battery penalty; it is not total SLRR.',ha='center',fontsize=9)
    save(fig,'user_count_convergence_all_algorithms')
    (DEST/'provenance.json').write_text(json.dumps(dict(created_at=datetime.now().astimezone().isoformat(),training_seed=42,window=300,sources=sources,
        scope='Actual completed training logs; one seed per algorithm/M; no confidence band across training seeds. Raw traces pale, same trailing300 valid window for all runs. No padding or extrapolation. M30 has15000 versus5000 forM20/25. Original reward depends on user count, so absolute returns acrossM do not isolate scalability.'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (DEST/'caption.tex').write_text(r'''Training returns for five algorithms under 20, 25, and 30 users.
Each setting uses training seed 42. Curves show a trailing average over 300
episodes, while pale traces show the raw returns. The 20- and 25-user policies
are trained for 5,000 episodes; the 30-user policies reuse completed
15,000-episode runs. Curves stop at the actual training budget without
extrapolation. Returns use the original user-count-dependent reward and
battery penalty and are not physical total SLRR. This is single-seed
training evidence rather than an independent-test performance comparison.
''',encoding='utf-8')
    renderer='C:/Users/lclroad/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe'
    for p in DEST.glob('*.pdf'):
        subprocess.run([renderer,'-scale-to','1500' if 'all_algorithms' in p.name else '1000','-singlefile','-png',str(p),str(p.with_name(p.stem+'_preview'))],check=True)
    print(json.dumps({'output':str(DEST),'curves':len(sources),'sources':[{k:r[k] for k in ['algorithm','M','episodes','last300_mean']} for r in sources]},ensure_ascii=True))

if __name__=='__main__':main()
