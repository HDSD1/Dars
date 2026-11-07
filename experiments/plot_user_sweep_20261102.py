"""Final user-sensitivity (a,b), energy and decision latency; no panel(c)."""
import csv
import json
from pathlib import Path
import subprocess
from run_user_sweep_20261102 import OUT, ALGOS, read

def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    rows=[read(OUT/f'evaluation/m{m}/{a}/summary.json') for a in ALGOS for m in (20,25,30)]
    for m in (20,25,30):
        for a in ALGOS:assert read(OUT/f'evaluation/m{m}/{a}/manifest.json')['status']=='completed'
    dest=OUT/'figures';dest.mkdir(exist_ok=False)
    marker=Path('C:/Users/lclroad/.codex/plugins/cache/openai-primary-runtime/pdf/26.1025.11240/skills/pdf/container_tools/mark_artifact_operation_started.mjs')
    subprocess.run(['C:/Program Files/nodejs/node.exe',str(marker),'--operation-kind','create','--expected-output-count','3','--output-format','pdf'],check=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9.5,'axes.labelsize':9.5,'axes.titlesize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    styles=[('DARS','#D62728','*'),('ESAC-PER','#1F77B4','p'),('SAC','#2CA02C','o'),('TD3','#9467BD','s'),('MADDPG','#FF7F0E','^')]
    flat=[]
    for r in rows:
        f={k:v for k,v in r.items() if not isinstance(v,dict)}
        for k,v in r.items():
            if isinstance(v,dict):
                f[k+'_mean']=v['mean'];f[k+'_ci_low']=v['ci95'][0] if v['ci95'] else None;f[k+'_ci_high']=v['ci95'][1] if v['ci95'] else None
        flat.append(f)
    with (dest/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    def draw(ax,field,ylabel,scale=1.,ci=True):
        for a,(name,color,marker) in zip(ALGOS,styles):
            rr=[r for r in rows if r['algorithm']==a];x=[r['M'] for r in rr]
            if ci:
                y=np.array([r[field]['mean'] for r in rr])*scale;bounds=np.array([r[field]['ci95'] for r in rr])*scale
                ax.errorbar(x,y,yerr=np.maximum(0,np.stack([y-bounds[:,0],bounds[:,1]-y])),color=color,marker=marker,ms=5,lw=1.1,capsize=2,label=name)
            else:ax.plot(x,[r[field]*scale for r in rr],color=color,marker=marker,ms=5,lw=1.1,label=name)
        ax.set_xticks([20,25,30]);ax.set_xlabel('Number of users');ax.set_ylabel(ylabel);ax.grid(alpha=.25,lw=.5)
    def save(fig,name):
        fig.savefig(dest/f'{name}.pdf');fig.savefig(dest/f'{name}.png',dpi=600);fig.savefig(dest/f'{name}.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.9));fig.subplots_adjust(left=.09,right=.98,bottom=.2,top=.78,wspace=.34)
    draw(axes[0],'reward','Evaluation episode return');draw(axes[1],'delay_executed','Mean completion delay (s)')
    axes[0].set_title('(a) Episode return',loc='left');axes[1].set_title('(b) Average delay',loc='left')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False);save(fig,'user_count_ab')
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.65));fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.87,wspace=.34,hspace=.5)
    for ax,(field,title) in zip(axes.flat,[('energy_fly_j','Flight'),('energy_hover_j','Hovering'),('energy_comp_j','Task computation'),('energy_out_j','Total')]):
        draw(ax,field,'Energy per episode (kJ)',.001);ax.set_title(title,loc='left');ax.set_ylim(bottom=0)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False);save(fig,'user_count_energy')
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.9));fig.subplots_adjust(left=.09,right=.98,bottom=.2,top=.78,wspace=.34)
    for ax,field,title in zip(axes,['decision_mean_ms','decision_p95_ms'],['Mean','95th percentile']):
        draw(ax,field,'Decision time per slot (ms)',ci=False);ax.set_title(title,loc='left');ax.set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False);save(fig,'user_count_decision_time')
    (dest/'figure_notes.txt').write_text('One training seed42 per algorithm/M;100 new evaluation episodes per point. Return and delay are distinct metrics; no redundant panel(c). Error bars are95% bootstrap intervals across test episodes conditional on the trained model, not multi-training-seed uncertainty. Actual energy until termination is retained; lower energy can reflect early failure. See survived/executed_steps and physical totalSLRR columns. Decision mean and P95 pool all attempted slots, including rejected actions. Local workstation measurements, not onboard hardware measurements. Simulated task-computation energy excludes neural-inference power.\n',encoding='utf-8')
    # Check font embedding and render all PDFs for subsequent visual inspection.
    py='C:/Users/lclroad/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    check="""import json,sys
from pathlib import Path
from pypdf import PdfReader
d=Path(sys.argv[1]);result={}
for p in d.glob('*.pdf'):
 fonts=[]
 for page in PdfReader(p).pages:
  for ref in page['/Resources']['/Font'].values():
   f=ref.get_object();df=f.get('/DescendantFonts');ff=df[0].get_object() if df else f
   desc=ff.get('/FontDescriptor');desc=desc.get_object() if desc else {}
   fonts.append(dict(name=str(f.get('/BaseFont')),embedded=any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3'])))
 assert fonts and all(f['embedded'] for f in fonts),fonts
 result[p.name]=fonts
(d/'pdf_font_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
"""
    subprocess.run([py,'-c',check,str(dest)],check=True)
    renderer='C:/Users/lclroad/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe'
    for p in dest.glob('*.pdf'):subprocess.run([renderer,'-scale-to','1400','-singlefile','-png',str(p),str(dest/(p.stem+'_preview'))],check=True)
    (dest/'visual_review_status.json').write_text(json.dumps(dict(status='rendered_pending_visual_inspection',pdf_font_embedding='verified',figures=3),indent=2),encoding='utf-8')

if __name__=='__main__':main()
