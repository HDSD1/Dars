"""Plot the ten completed, compatible training runs without touching training."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/five_algorithms_seeds42_43_20261101'
WINDOW = 300
STYLES = [
    ('per_sac_cvx', 'DARS', '#D62728'),
    ('esac_per_cvx', 'ESAC-PER', '#1976B9'),
    ('sac_cvx', 'SAC', '#238B45'),
    ('td3_cvx', 'TD3', '#8064A2'),
    ('maddpg_cvx', 'MADDPG', '#DE8B16'),
]
SOURCE_KEYS = ('UAV_env.py', 'evaluation.py', 'train_unified.py', 'config.py', 'DARS.py')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def directory(algo, seed):
    if seed == 43:
        return ROOT / ('results/formal_all_five_seeds43_44_restart_20261031/'
                       'source_snapshot/results/formal_all_five_seeds43_44_session2_20261031_seed43')
    if algo in ('td3_cvx', 'maddpg_cvx'):
        return ROOT / ('results/remaining_td3_maddpg_seed42_20261030/source_snapshot/'
                       'results/remaining_td3_maddpg_seed42_20261030')
    return ROOT / 'results/original_main_seed42_gpu_retry_20261029_234000'


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    series, provenance, endings = [], [], []
    comparison = None
    for algo, label, color in STYLES:
        for seed in (42, 43):
            base = directory(algo, seed)
            path = base / f'{algo}.train.jsonl'
            mp = base / f'{algo}.manifest.json'
            m = json.loads(mp.read_text(encoding='utf-8'))
            assert m['status'] == 'completed' and m['training_seed'] == seed
            signature = (m['environment'], m['training'],
                         {k: m['source_sha256'][k] for k in SOURCE_KEYS})
            if comparison is None:
                comparison = signature
            assert signature == comparison, 'Do not pool incompatible training versions'
            values = []
            with path.open(encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)
                    assert r['horizon'] == 50
                    values.append((r['reward'], r['slrr_fixed_horizon']))
            raw = np.asarray(values, dtype=float)
            assert raw.shape == (15000, 2) and np.isfinite(raw).all()
            smooth = np.column_stack([np.convolve(raw[:, j], np.ones(WINDOW)/WINDOW,
                                                  mode='valid') for j in (0, 1)])
            series.append(dict(algo=algo, label=label, color=color, seed=seed, raw=raw, smooth=smooth))
            provenance.append(dict(algorithm=algo, training_seed=seed, episodes=len(raw),
                path=str(path), sha256=sha(path), manifest=str(mp), manifest_sha256=sha(mp),
                status=m['status'], hotspot_mode=m['environment']['hotspot_mode'],
                clusters=m['system_parameters']['N_CLUSTERS']))
            endings.append(dict(algorithm=label, training_seed=seed,
                last_1000_training_return_mean=float(raw[-1000:, 0].mean()),
                last_1000_training_slrr_fixed_horizon_mean=float(raw[-1000:, 1].mean())))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.labelsize': 13, 'axes.titlesize': 13, 'xtick.labelsize': 11,
        'ytick.labelsize': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.subplots_adjust(left=.075, right=.98, bottom=.17, top=.75, wspace=.25)
    episodes = np.arange(WINDOW, 15001)
    for item in series:
        for j, ax in enumerate(axes):
            ax.plot(episodes, item['smooth'][:, j], color=item['color'],
                    ls='-' if item['seed'] == 42 else (0, (5, 3)),
                    lw=1.6, alpha=.93)
    for ax in axes:
        ax.set_xlim(0, 15000)
        ax.set_xticks(np.arange(0, 15001, 3000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.set_xlabel('Training episode')
        ax.grid(color='#D9DDE2', alpha=.7, linewidth=.6)
        ax.set_axisbelow(True)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(0, lo), hi)
    axes[0].set(title='(a) Training return', ylabel='Episode return')
    axes[1].set(title='(b) Service latency reduction', ylabel='Mean SLRR (fixed 50-slot horizon)')
    fig.suptitle('Five algorithms across two training seeds', fontsize=18, y=.98, fontweight='bold')
    fig.legend([Line2D([0], [0], color=c, lw=2.7) for _, _, c in STYLES],
               [name for _, name, _ in STYLES], loc='upper center',
               bbox_to_anchor=(.5, .926), ncol=5, frameon=False, columnspacing=2.1)
    fig.legend([Line2D([0], [0], color='#333333', lw=1.8, ls='-'),
                Line2D([0], [0], color='#333333', lw=1.8, ls=(0, (5, 3)))],
               ['Seed 42', 'Seed 43'], loc='upper center', bbox_to_anchor=(.5, .864),
               ncol=2, frameon=False, handlelength=3.5, columnspacing=2.5)
    fig.text(.5, .062, '300-episode trailing moving average; each line is one training run.',
             ha='center', color='#444444', fontsize=11)
    fig.text(.5, .027, 'Existing four-cluster diverse-hotspot training. These are training curves, not held-out evaluation results.',
             ha='center', color='#666666', fontsize=10)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'five_algorithms_seeds42_43.{ext}', dpi=220, facecolor='white')
    plt.close(fig)

    with (OUT/'plot_data.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['algorithm', 'training_seed', 'episode', 'episode_return_raw',
                    'slrr_fixed_horizon_raw', 'episode_return_ma300', 'slrr_fixed_horizon_ma300'])
        for item in series:
            for i, values in enumerate(item['raw']):
                sm = item['smooth'][i-WINDOW+1] if i >= WINDOW-1 else ('', '')
                w.writerow([item['algo'], item['seed'], i+1, *values, *sm])
    with (OUT/'last_1000_training_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(endings[0])); w.writeheader(); w.writerows(endings)
    manifest = dict(status='completed', training_seeds=[42,43], window=WINDOW,
        smoothing='Trailing mean of 300 full episodes, first point at episode 300; no padding',
        data_scope='Training trajectories, including early terminations; not held-out performance or a two-seed confidence interval',
        metrics={'reward':'Accumulated simulator reward, including battery penalty; not total SLRR',
                 'slrr_fixed_horizon':'Sum of executed-slot rho_mean divided by the fixed horizon of 50'},
        sources=provenance, common_environment=comparison[0], common_training=comparison[1],
        common_source_sha256=comparison[2], script_sha256=sha(Path(__file__)))
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(OUT), runs=len(series), episodes_per_run=15000,
                         status='completed', last_1000=endings), ensure_ascii=False))


if __name__ == '__main__':
    run()
