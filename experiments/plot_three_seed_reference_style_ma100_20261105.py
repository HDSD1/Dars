"""Read-only visualization of 15 completed original training runs."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import json
import hashlib
import csv
import subprocess
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/five_algorithms_three_seeds_reference_style_20261105_ma100'
WINDOW = 100
STYLES = [
    ('maddpg_cvx', 'MADDPG', '#FF7F0E', '-.', '^'),
    ('td3_cvx', 'TD3', '#9467BD', '-.', 's'),
    ('sac_cvx', 'SAC', '#2CA02C', '--', 'o'),
    ('esac_per_cvx', 'ESAC-PER', '#1F77B4', '--', 'p'),
    ('per_sac_cvx', 'DARS (Ours)', '#D62728', '-', '*'),
]

def source(algo, seed):
    if seed == 44 and algo == 'td3_cvx':
        return ROOT / 'results/td3_seed44_retry_20261102/source_snapshot/results/formal_td3_seed44_retry_20261102'
    if seed in (43, 44):
        return ROOT / f'results/formal_all_five_seeds43_44_restart_20261031/source_snapshot/results/formal_all_five_seeds43_44_session2_20261031_seed{seed}'
    if algo in ('td3_cvx', 'maddpg_cvx'):
        return ROOT / 'results/remaining_td3_maddpg_seed42_20261030/source_snapshot/results/remaining_td3_maddpg_seed42_20261030'
    return ROOT / 'results/original_main_seed42_gpu_retry_20261029_234000'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10,
        'axes.labelsize': 10, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'legend.fontsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'path', 'axes.linewidth': .65, 'axes.edgecolor': '#333333'})
    raw = {}; smooth = {}; sources = []; signature = None; ends = []
    source_keys = ('UAV_env.py', 'evaluation.py', 'train_unified.py', 'config.py', 'DARS.py')
    for algo, label, *_ in STYLES:
        raw[algo] = []; smooth[algo] = []
        for seed in (42, 43, 44):
            folder = source(algo, seed)
            mp = folder / f'{algo}.manifest.json'; m = read(mp)
            assert m['status'] == 'completed' and m['training_seed'] == seed
            sig = (m['environment'], m['training'], {k: m['source_sha256'][k] for k in source_keys})
            if signature is None: signature = sig
            assert sig == signature, f'Incompatible source or training definition: {algo}/{seed}'
            tp = folder / f'{algo}.train.jsonl'
            with tp.open(encoding='utf-8') as f:
                records = [json.loads(line) for line in f if line.strip()]
            assert len(records) == 15000 and all(r['horizon'] == 50 for r in records)
            assert [r['seed'] for r in records] == m['training_episode_seeds']
            y = np.asarray([r['reward'] for r in records], dtype=float)
            assert np.isfinite(y).all()
            sm = np.convolve(y, np.ones(WINDOW) / WINDOW, mode='valid')
            raw[algo].append(y); smooth[algo].append(sm)
            sources.append(dict(algorithm=algo, label=label, seed=seed, episodes=len(y),
                path=str(tp), training_sha256=sha(tp), manifest_path=str(mp), manifest_sha256=sha(mp),
                reward_field='reward', original_failed_td3_seed44_excluded=(algo=='td3_cvx' and seed==44)))
            ends.append(dict(algorithm=algo, seed=seed, last1000_mean=float(y[-1000:].mean())))
    x = np.arange(WINDOW, 15001)
    aggregate = {a: (np.mean(smooth[a], axis=0), np.std(smooth[a], axis=0, ddof=1)) for a, *_ in STYLES}
    # Every curve uses the same valid trailing window, with no padding or fabricated initial values.
    fig, ax = plt.subplots(figsize=(7.16, 5.25))
    fig.subplots_adjust(left=.105, right=.98, bottom=.12, top=.97)
    for algo, label, color, ls, marker in STYLES:
        mean, sd = aggregate[algo]
        ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.12, linewidth=0)
        ax.plot(x, mean, color=color, linestyle=ls, linewidth=1.8 if algo=='per_sac_cvx' else 1.25,
            label=label, marker=marker, markersize=4.5 if algo=='per_sac_cvx' else 3.6,
            markeredgewidth=.35, markevery=[1000, 3000, 5000, 7000, 9000, 11000, 13000, len(x)-1])
    ax.set(xlabel='Episode', ylabel='Episode Reward', xlim=(0, 15500))
    ax.set_xticks(np.arange(0, 15001, 2000))
    ax.grid(True, color='#b0b0b0', alpha=.32, linestyle='--', linewidth=.45)
    ax.set_axisbelow(True)
    legend = ax.legend(loc='upper left', frameon=True, fancybox=False, framealpha=1,
        borderpad=.35, labelspacing=.25, handlelength=2.4, handletextpad=.6)
    legend.get_frame().set_linewidth(.5); legend.get_frame().set_edgecolor('#777777')
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(OUT/f'three_seed_training_convergence.{ext}', dpi=600)
    plt.close(fig)
    np.savez_compressed(OUT/'verified_training_data.npz', episodes=x,
        **{f'{a}_raw': np.asarray(raw[a]) for a, *_ in STYLES},
        **{f'{a}_ma100': np.asarray(smooth[a]) for a, *_ in STYLES})
    with (OUT/'plotted_mean_sd.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['algorithm', 'episode', 'mean_reward', 'sample_sd_reward'])
        for a, *_ in STYLES:
            mean, sd = aggregate[a]
            w.writerows(zip([a]*len(x), x, mean, sd))
    caption = ('Training convergence of five algorithms over three training seeds (42, 43, and 44). '
        'Each run contains 15,000 episodes. Curves show the equal-weight mean of per-seed trailing '
        '100-episode average rewards; shaded bands indicate one sample standard deviation across '
        'the three smoothed seed curves. TD3 seed 44 uses its completed fresh retry; the original '
        'failed 557-episode run is excluded. Reward is the originally trained M-times-mean-active-rho '
        'objective plus the battery penalty, not physical total SLRR.\n')
    (OUT/'caption.txt').write_text(caption, encoding='utf-8')
    (OUT/'plot_three_seed_reference_style_ma100_20261105.py').write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
    poppler = Path('C:/Users/lclroad/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')
    pdf = OUT/'three_seed_training_convergence.pdf'
    subprocess.run([str(poppler/'pdftoppm.exe'), '-scale-to', '1500', '-singlefile', '-png', str(pdf), str(OUT/'preview')], check=True)
    inspect_font = """import json,sys
from pypdf import PdfReader
r=PdfReader(sys.argv[1]);out=[]
for ref in r.pages[0]['/Resources']['/Font'].values():
 f=ref.get_object(); child=f['/DescendantFonts'][0].get_object();d=child['/FontDescriptor'].get_object()
 blob=d['/FontFile2'].get_object().get_data()
 out.append({'name':str(f['/BaseFont']),'embedded_bytes':len(blob)})
print(json.dumps(out))
"""
    runtime='C:/Users/lclroad/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    fonts = json.loads(subprocess.run([runtime, '-c', inspect_font, str(pdf)], check=True,
        capture_output=True, encoding='utf-8').stdout)
    (OUT/'pdf_fonts.json').write_text(json.dumps(fonts,indent=2)+'\n',encoding='utf-8')
    assert fonts and all('TimesNewRoman' in f['name'] and f['embedded_bytes']>0 for f in fonts)
    manifest = dict(created_at=datetime.now().astimezone().isoformat(), status='rendered_pending_visual_review',
        training_seeds=[42,43,44], algorithms=5, runs=15, episodes_per_run=15000,
        smoothing='Trailing 100-episode arithmetic mean per seed, valid windows only (episode100-15000)',
        band='Equal-seed mean +/- one sample standard deviation (ddof=1), not confidence interval',
        sources=sources, compatibility_verified=True, source_keys=source_keys,
        td3_retry='Fresh seed44; CPU threads reduced to2, same learning definition. Original557 failed records excluded.',
        reward_version='Original four-cluster diverse, Greedy association; M*mean-active rho plus battery penalty.',
        dimensions_inches=[7.16,5.25], png_dpi=600, font='Embedded Times New Roman',
        source_changes=False, manuscript_changes=False, last1000=ends)
    (OUT/'provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output':str(OUT), 'runs':15,'last1000':ends,'fonts':fonts}, ensure_ascii=True))

if __name__ == '__main__':
    main()
