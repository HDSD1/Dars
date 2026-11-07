"""Audit and visualize recorded user-permutation tests, without model execution."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]/'results/reviewer_r32_permutation_20261102'
OUT=ROOT/'figures_iotj'
RED='#D62728';BLUE='#1F77B4';GRAY='#555555'

def main():
    OUT.mkdir(exist_ok=True)
    errors={};provenance=[];summary=[];allrows=[];example=None
    for seed in (42,43,44):
        p=ROOT/f'seed{seed}/raw_test_arrays.npz';z=np.load(p)
        report=json.loads((p.parent/'report.json').read_text(encoding='utf-8'));a=z['baseline_normalized_actions'];ap=z['permuted_normalized_actions'];perm=z['permutations'];ix=z['source_rows']
        assert a.shape==(32,36) and ap.shape==(320,36) and perm.shape==(320,30)
        assert all(np.array_equal(np.sort(q),np.arange(30)) for q in perm)
        off=(a[:,6:]+1)/2;offp=(ap[:,6:]+1)/2
        aligned=np.take_along_axis(offp,np.argsort(perm,axis=1),axis=1)
        eo=np.max(np.abs(aligned-off[ix]),axis=1);em=np.max(np.abs(ap[:,:6]-a[ix,:6]),axis=1)
        for label,val in [('offload_max_abs_error',eo.max()),('movement_normalized_max_abs_error',em.max())]:assert float(val)==report['metrics'][label]
        expected_mean=z['baseline_mean'][ix].copy();expected_ls=z['baseline_log_std'][ix].copy()
        expected_mean[:,6:]=z['baseline_mean'][ix[:,None],6+perm];expected_ls[:,6:]=z['baseline_log_std'][ix[:,None],6+perm]
        assert np.allclose(z['permuted_mean'],expected_mean,atol=1e-5,rtol=1e-5)
        assert np.allclose(np.exp(z['permuted_log_std']),np.exp(expected_ls),atol=1e-5,rtol=1e-5)
        assert np.max(eo)<=1e-5 and np.max(em)<=1e-5 and np.min(off.std(axis=1))>1e-4
        errors[seed]=(eo,em)
        summary.append(dict(training_seed=seed,states=32,permutations=320,offload_max=float(eo.max()),movement_max=float(em.max()),minimum_userwise_output_sd=float(off.std(axis=1).min())))
        provenance.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),checkpoint_sha256=report['checkpoint_sha256']))
        for j in range(320):allrows.append(dict(training_seed=seed,state_index=int(ix[j]),permutation_index=j%10,offload_max_abs_error=float(eo[j]),movement_max_abs_error=float(em[j])))
        if seed==42:
            ids=json.loads((p.parent/'state_ids.json').read_text(encoding='utf-8'))
            example=dict(seed=42,state_index=0,permutation_index=0,**ids[0],original=off[0],permuted_slot_order=offp[0],realigned=aligned[0],permutation=perm[0],max_error=float(eo[0]))
    def csvwrite(name,rows):
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csvwrite('all_960_permutation_errors.csv',allrows);csvwrite('per_seed_maxima.csv',summary)
    csvwrite('example_user_alignment.csv',[dict(original_user_id=i+1,original_offload=float(example['original'][i]),realigned_offload=float(example['realigned'][i]),new_slot_for_this_user=int(np.argsort(example['permutation'])[i])+1) for i in range(30)])
    csvwrite('example_permuted_slots.csv',[dict(new_slot=i+1,original_user_id=int(example['permutation'][i])+1,permuted_offload=float(example['permuted_slot_order'][i])) for i in range(30)])
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9.5,'axes.labelsize':9.5,'axes.titlesize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'mathtext.fontset':'stix'})
    def alignment(ax):
        x=np.arange(1,31)
        ax.plot(x,example['original'],color=RED,lw=1.,marker='o',ms=2.7,label='Original')
        ax.plot(x,example['realigned'],color=BLUE,ls='--',lw=1.,dashes=(3,2),marker='s',mfc='none',mec=BLUE,mew=.85,ms=4.8,label='Permuted, then realigned',zorder=3)
        ax.set(xlim=(.3,30.7),ylim=(0,1.03),xlabel='Original user ID',ylabel='Offloading proposal')
        ax.set_xticks([1,5,10,15,20,25,30]);ax.set_yticks([0,.25,.5,.75,1]);ax.grid(axis='y',alpha=.2,lw=.5)
        ax.legend(loc='lower left',bbox_to_anchor=(-.02,1.01),frameon=False,ncol=1,handletextpad=.5,labelspacing=.35)
        ax.set_title('(a) Actions follow user identity',loc='left',pad=50)
    def discrepancy(ax):
        vals=[];pos=[]
        for i,seed in enumerate((42,43,44)):
            vals.extend(errors[seed]);pos.extend([i-.16,i+.16])
        boxes=ax.boxplot(vals,positions=pos,widths=.24,whis=(0,100),showfliers=False,patch_artist=True,manage_ticks=False,
            medianprops=dict(color='#222222',lw=1),whiskerprops=dict(color=GRAY,lw=.8),capprops=dict(color=GRAY,lw=.8),boxprops=dict(lw=.8))
        for i,b in enumerate(boxes['boxes']):b.set_facecolor(RED if i%2==0 else BLUE);b.set_alpha(.50)
        for i,v in enumerate(vals):ax.scatter(pos[i],max(v),marker='D',s=13,color=RED if i%2==0 else BLUE,zorder=4)
        ax.axhline(1e-5,color=GRAY,ls='--',lw=1)
        ax.text(1,1.12e-5,r'Tolerance $10^{-5}$',ha='center',va='bottom',fontsize=8.5,color=GRAY)
        ax.set_yscale('log');ax.set_ylim(3e-8,2e-5);ax.set_xlim(-.55,2.55);ax.set_xticks([0,1,2],['42','43','44'])
        ax.set_yticks([1e-7,1e-6,1e-5]);ax.set_xlabel('Training seed');ax.set_ylabel('Max. absolute discrepancy')
        ax.grid(axis='y',which='major',alpha=.2,lw=.5)
        ax.legend(handles=[Patch(facecolor=RED,alpha=.5,label='Offloading equivariance'),Patch(facecolor=BLUE,alpha=.5,label='Movement invariance')],loc='lower left',bbox_to_anchor=(-.02,1.01),frameon=False,ncol=1,labelspacing=.35)
        ax.set_title('(b) All 960 permutation tests',loc='left',pad=50)
    def save(fig,name):
        for ext in ('pdf','png','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=600)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.16,3.65));fig.subplots_adjust(left=.075,right=.985,bottom=.22,top=.69,wspace=.37)
    alignment(axes[0]);discrepancy(axes[1])
    fig.text(.5,.035,'Boxes: interquartile range; whiskers: minimum–maximum; diamonds: maximum.',ha='center',fontsize=8.5)
    save(fig,'r32_permutation_test')
    for fun,name in [(alignment,'r32_user_alignment'),(discrepancy,'r32_permutation_error')]:
        fig,ax=plt.subplots(figsize=(3.5,3.2));fig.subplots_adjust(left=.19,right=.97,bottom=.17,top=.65);fun(ax);save(fig,name)
    caption=r'''% Suggested caption only; main manuscript is not modified.
\caption{User-order permutation consistency of the trained DARS Actor.
(a) Deterministic offloading proposals for all 30 users before permutation and
after inverse-permutation alignment. The example is fixed to training seed 42,
the first sampled test state (test seed 951000, slot 0), and its first random
permutation, without outcome-based selection. The distinct user-specific
proposals overlap after realignment.
(b) Distributions of the maximum absolute discrepancy per permutation for
three independently trained models. Each model uses 32 test states and 10
random permutations per state (960 tests in total). User positions, task sizes,
and activity flags are permuted jointly; UAV-side features and time are held
fixed. Boxes show the interquartile range and median, whiskers the full range,
and diamonds the maximum. All offloading and normalized movement discrepancies
are below the prespecified absolute tolerance of $10^{-5}$; the maximum in
each category is $1.61\times10^{-6}$. These are descriptive distributions,
not confidence intervals. The test concerns Actor proposals before association
and energy-safety processing.}
'''
    (OUT/'caption.tex').write_text(caption,encoding='utf-8')
    notes=dict(source_protocol=str(ROOT/'protocol.json'),sources=provenance,summary=summary,example_selection='Seed42, first test state, first permutation, all30users; fixed by record order, not error magnitude.',example=dict(test_seed=example['test_seed'],slot=example['slot'],max_aligned_error=example['max_error']),
        errors='offload: max_i |inverse_permutation(offload(s_permuted))_i - offload(s)_i|; movement: max_j |move(s_permuted)_j - move(s)_j|.',
        support='Retained per-user attention representations plus a parameter-shared head explain correspondence; these plots numerically test sampled Actor outputs, not the full downstream scheduler or a proof for all possible states.',
        uncertainty='Boxes show empirical320tests/model; states and within-state permutations are dependent. No inferential CI or training-seed errorbar.',
        scope='CPUfloat32 frozen deterministicActor with all user tuples permuted; Gaussian mean/std parameter consistency rechecked. No new model inference or training. Main manuscript, Bib and referenced figures unchanged.')
    (OUT/'plot_audit.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');shutil.copy2(__file__,OUT/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),summary=summary),ensure_ascii=True))

if __name__=='__main__':main()
