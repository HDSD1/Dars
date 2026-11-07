"""Recompute service/energy metrics from all existing paired test episodes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/reviewer_users_m20_m25_m30_seed42_5000_20261102'
ALGORITHMS = ['per_sac_cvx', 'esac_per_cvx', 'sac_cvx', 'td3_cvx', 'maddpg_cvx']
STYLES = dict(zip(ALGORITHMS, [
    ('DARS', '#D62728', '*'), ('ESAC-PER', '#1F77B4', 'p'),
    ('SAC', '#2CA02C', 'o'), ('TD3', '#9467BD', 's'),
    ('MADDPG', '#FF7F0E', '^')]))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def estimate(value, samples):
    return {'estimate': float(value), 'ci95': np.percentile(samples, [2.5, 97.5]).tolist()}


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    points, comparisons, sources, checks = {}, [], [], []
    for m in (20, 25, 30):
        seeds, external = None, None
        boot, arrays = {}, {}
        index = np.random.default_rng(20261105 + m).integers(0, 100, size=(10000, 100))
        for algo in ALGORITHMS:
            folder = DATA / f'evaluation/m{m}' / algo
            manifest, summary = read(folder / 'manifest.json'), read(folder / 'summary.json')
            assert manifest['status'] == 'completed' and manifest['training_seed'] == 42
            raw = (folder / 'episodes.jsonl').read_bytes()
            rows = sorted([json.loads(line) for line in raw.decode('utf-8').splitlines()], key=lambda r: r['seed'])
            assert len(rows) == 100 and len({r['seed'] for r in rows}) == 100
            actual_seeds = [r['seed'] for r in rows]
            actual_external = [r['external_state_hashes'] for r in rows]
            if seeds is None:
                seeds, external = actual_seeds, actual_external
            else:
                assert seeds == actual_seeds and external == actual_external
            fields = ['cumulative_total_slrr', 'energy_out_j', 'energy_fly_j',
                      'energy_hover_j', 'energy_comp_j', 'executed_steps', 'survived']
            a = {field: np.asarray([r[field] for r in rows], dtype=float) for field in fields}
            assert all(np.all(np.isfinite(v)) for v in a.values())
            assert np.all(a['energy_out_j'] > 0) and np.all(a['executed_steps'] > 0)
            for field in fields:
                np.testing.assert_allclose(a[field].mean(), summary[field]['mean'], rtol=1e-10, atol=1e-8)
            np.testing.assert_allclose(a['energy_out_j'], a['energy_fly_j'] + a['energy_hover_j'] + a['energy_comp_j'], rtol=1e-10, atol=1e-6)
            for r in rows:
                assert sum(s['executed'] for s in r['slots']) == r['executed_steps']
                np.testing.assert_allclose(sum(s['total_slrr'] for s in r['slots']), r['cumulative_total_slrr'], rtol=1e-10, atol=1e-8)
                for label, field in [('fly', 'energy_fly_j'), ('hover', 'energy_hover_j'), ('comp', 'energy_comp_j')]:
                    total = sum(sum(s['energy_' + label + '_j']) for s in r['slots'] if s['executed'])
                    np.testing.assert_allclose(total, r[field], rtol=1e-10, atol=1e-6)
            means = {field: values[index].mean(axis=1) for field, values in a.items()}
            efficiency = a['cumulative_total_slrr'].sum() / (a['energy_out_j'].sum() / 1000)
            efficiency_bs = means['cumulative_total_slrr'] / (means['energy_out_j'] / 1000)
            flight = a['energy_fly_j'].sum() / (2 * a['executed_steps'].sum())
            flight_bs = means['energy_fly_j'] / (2 * means['executed_steps'])
            point = {
                'M': m, 'algorithm': algo, 'label': STYLES[algo][0],
                'training_seed': 42, 'training_episodes': manifest['training_episodes'],
                'test_episodes': len(rows), 'complete_episodes': int(a['survived'].sum()),
                'slrr_per_kj': estimate(efficiency, efficiency_bs),
                'joules_per_unit_total_slrr': estimate(1000 / efficiency, 1000 / efficiency_bs),
                'flight_j_per_executed_uav_slot': estimate(flight, flight_bs),
                'energy_kj_per_episode': estimate(a['energy_out_j'].mean()/1000, means['energy_out_j']/1000),
                'cumulative_total_slrr': estimate(a['cumulative_total_slrr'].mean(), means['cumulative_total_slrr']),
                'total_j_per_executed_uav_slot': estimate(a['energy_out_j'].sum()/(2*a['executed_steps'].sum()), means['energy_out_j']/(2*means['executed_steps'])),
                'component_percent': {kind: float(100*a[f'energy_{kind}_j'].sum()/a['energy_out_j'].sum()) for kind in ('fly', 'hover', 'comp')},
                'mean_energy_j': {kind: float(a[f'energy_{kind}_j'].mean()) for kind in ('fly', 'hover', 'comp', 'out')},
            }
            points[m, algo] = point
            boot[algo] = {'efficiency': efficiency_bs, 'flight': flight_bs}
            arrays[algo] = a
            sources.append({'path': str((folder/'episodes.jsonl').resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                            'checkpoint_sha256': manifest['checkpoint_sha256']})
            checks.append({'M': m, 'algorithm': algo, 'all_100_episodes_included': True,
                           'summary_reproduced': True, 'component_energy_sum_verified': True,
                           'slot_slrr_and_energy_verified': True, 'paired_exogenous_sequences_verified': True})
        dars = points[m, ALGORITHMS[0]]
        for algo in ALGORITHMS[1:]:
            baseline = points[m, algo]
            comparisons.append({
                'M': m, 'baseline': STYLES[algo][0],
                'efficiency_gain_percent': estimate(100*(dars['slrr_per_kj']['estimate']/baseline['slrr_per_kj']['estimate']-1),
                    100*(boot[ALGORITHMS[0]]['efficiency']/boot[algo]['efficiency']-1)),
                'flight_energy_reduction_percent': estimate(100*(1-dars['flight_j_per_executed_uav_slot']['estimate']/baseline['flight_j_per_executed_uav_slot']['estimate']),
                    100*(1-boot[ALGORITHMS[0]]['flight']/boot[algo]['flight'])),
                'energy_per_slrr_reduction_percent': estimate(100*(1-baseline['slrr_per_kj']['estimate']/dars['slrr_per_kj']['estimate']),
                    100*(1-boot[algo]['efficiency']/boot[ALGORITHMS[0]]['efficiency'])),
            })
    document = {
        'status': 'completed', 'analysis_type': 'Exploratory secondary analysis of existing frozen-model test data.',
        'definitions': {
            'slrr_per_kj': 'sum of actual cumulative total SLRR across all 100 episodes / sum of UAV flight+hover+task-computation energy in kJ.',
            'flight_j_per_executed_uav_slot': 'sum of actual flight energy / (2 * total executed slots), in J/UAV-slot.',
            'energy_scope': 'Simulated UAV flight, hovering, and task-computation energy; no subtraction of harvested energy; excludes neural-inference/control-radio energy.',
            'termination': 'All episodes included. Service and energy stop at actual termination. No survivor filtering, clipping, or fabricated post-termination energy.',
            'uncertainty': '10000 paired episode bootstrap resamples, same test-seed indices across all algorithms at fixed M. Pointwise percentile95% intervals conditional on training seed42; not simultaneous or across-training-seed intervals.',
            'budget': 'M20/25:5000 training episodes;M30:15000. Compare algorithms within each M.',
            'primary_objective': 'Original total SLRR unchanged. Energy efficiency is an additional reporting metric, not a new training reward.',
        },
        'points': list(points.values()), 'paired_comparisons': comparisons,
        'source_files': sources, 'validation': checks,
        'statistical_review': {
            'coverage': '11/11 statistical fallacy types reviewed',
            'simpson': 'M groups reported separately; no pooled superiority claim.',
            'ecological': 'Aggregate ratios describe system output per energy, not each user.',
            'selection': 'All algorithms and all100 matched test episodes retained.',
            'collider': 'No conditioning on survival or low energy.',
            'base_rate': 'Complete-episode counts retained for all15 configurations.',
            'regression_to_mean': 'No extreme-window selection or before-after claim.',
            'survivorship': 'Early-terminated episodes included; duration-normalized flight metric and actual costs.',
            'look_elsewhere': 'Exploratory metrics; no hypothesis-test or confirmatory significance claim.',
            'forking_paths': 'Definitions, all selected-metric results, bootstrap seed, source hashes archived.',
            'causal': 'Descriptive system comparisons; no claim attention alone causes energy differences.',
            'reverse_causality': 'No causal battery/performance inference.',
        },
    }
    (out/'metrics.json').write_text(json.dumps(document, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10.5, 'legend.fontsize': 9,
        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'path', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .75})

    def save(fig, name):
        for suffix in ('pdf', 'svg'):
            fig.savefig(out/f'{name}.{suffix}', facecolor='white')
        fig.savefig(out/f'{name}.png', dpi=400, facecolor='white')
        fig.savefig(out/f'{name}_preview.png', dpi=195, facecolor='white')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.55))
    fig.subplots_adjust(left=.09, right=.985, top=.78, bottom=.26, wspace=.36)
    for ax, field, label, title in zip(axes,
            ['slrr_per_kj', 'flight_j_per_executed_uav_slot'],
            ['Total SLRR per energy (kJ$^{-1}$)', 'Flight energy per UAV-slot (J)'],
            ['(a) Energy efficiency', '(b) Flight energy per executed slot']):
        for algo in ALGORITHMS:
            name, color, marker = STYLES[algo]
            rows = [points[m, algo][field] for m in (20, 25, 30)]
            y = np.array([r['estimate'] for r in rows]); ci = np.array([r['ci95'] for r in rows])
            ax.errorbar([20, 25, 30], y, yerr=np.maximum(0, np.array([y-ci[:, 0], ci[:, 1]-y])),
                        color=color, marker=marker, ms=7 if algo==ALGORITHMS[0] else 5,
                        lw=1.4 if algo==ALGORITHMS[0] else 1.1, capsize=2.7, elinewidth=.8,
                        label=name, zorder=5 if algo==ALGORITHMS[0] else 3)
        ax.set_xticks([20, 25, 30]); ax.set_xlim(19.3, 30.7)
        ax.set_xlabel('Number of users'); ax.set_ylabel(label)
        ax.set_title(title, loc='left', pad=10); ax.grid(alpha=.22, lw=.5)
    axes[0].set_ylim(1.3, 2.55); axes[0].set_yticks([1.4,1.6,1.8,2.0,2.2,2.4])
    axes[1].set_ylim(140, 220); axes[1].set_yticks([140,160,180,200,220])
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.52, .985), ncol=5,
               frameon=False, columnspacing=1.35, handlelength=1.6)
    fig.text(.53,.105,'100 paired test episodes; 95% bootstrap intervals; training seed 42.',ha='center',fontsize=8.5)
    fig.text(.53,.052,'Training: 5,000 episodes for 20/25 users; 15,000 for 30 users.',ha='center',fontsize=8.5)
    save(fig,'energy_efficiency_and_flight')

    fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.45), sharey=True, sharex=True)
    fig.subplots_adjust(left=.14, right=.98, top=.85, bottom=.23, wspace=.17)
    for ax, m in zip(axes, (20,25,30)):
        for j, algo in enumerate(ALGORITHMS[1:]):
            name,color,marker = STYLES[algo]
            result = next(r for r in comparisons if r['M']==m and r['baseline']==name)['efficiency_gain_percent']
            v = result['estimate']; lo,hi = result['ci95']
            ax.errorbar(v,j,xerr=[[max(0,v-lo)],[max(0,hi-v)]],color=color,marker=marker,ls='none',ms=6,
                        capsize=3,elinewidth=1.25)
        ax.axvline(0,color='#555555',ls='--',lw=.9,zorder=0)
        ax.set_title(f'{m} users',pad=9);ax.set_xlabel('DARS efficiency gain (%)')
        ax.set_yticks(range(4),[STYLES[a][0] for a in ALGORITHMS[1:]])
        ax.set_xlim(-5,36);ax.set_xticks([0,10,20,30]);ax.grid(axis='x',alpha=.2,lw=.5)
    axes[0].invert_yaxis()
    fig.text(.55,.075,'Paired 95% bootstrap intervals across all 100 test episodes (one trained model).',ha='center',fontsize=8.3)
    fig.text(.55,.025,'Training: 5,000 episodes for 20/25 users; 15,000 for 30 users.',ha='center',fontsize=8.3)
    save(fig,'paired_energy_efficiency_gain')

    captions = r'''% Secondary energy analysis; definitions and all source hashes are in metrics.json.
\textcolor{change}{
We evaluate the energy efficiency as the cumulative total SLRR divided by
the aggregate UAV energy consumption, including flight, hovering, and task
computation. We also report flight energy per executed UAV-slot to account
for differences in service duration. Both metrics use all 100 test episodes
for each trained policy, including episodes that terminate early.
}

% Suggested figure caption; use your own figure labels and filenames.
\caption{Energy performance under different user populations:
(a) total SLRR per consumed energy and (b) flight energy per executed UAV-slot.
Error bars indicate 95\% bootstrap confidence intervals over 100 test episodes
for training seed 42. Models with 20/25 users were trained for 5,000 episodes;
those with 30 users were trained for 15,000 episodes.}
'''
    (out/'caption_proposal.tex').write_text(captions,encoding='utf-8')
    print(json.dumps({'output':str(out),'points':len(points),'paired_comparisons':comparisons},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DATA/'figures_energy_efficiency_20261105')
    args=parser.parse_args()
    run(args.output.resolve())
