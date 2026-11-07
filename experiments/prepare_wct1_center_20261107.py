"""Prepare the explicitly authorized single-seed central-WCT experiment."""
from pathlib import Path
import hashlib
import json
import re
import shutil
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/reviewer_users_m15_m35_three_seeds_10000_server_20261105/bundle'
OLD = BASE / 'sources/m30'
OUT = ROOT / 'results/reviewer_wct1_center_dars_seed42_10000_20261107'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def prepare():
    OUT.mkdir(exist_ok=False)
    src = OUT / 'source_snapshot'
    src.mkdir(); (src / 'experiments').mkdir()
    for p in OLD.glob('*.py'): shutil.copy2(p, src / p.name)
    text = (src / 'config.py').read_text(encoding='utf-8')
    text, n = re.subn(r'^N_WCT = 4[^\n]*', 'N_WCT = 1  # Author-authorized single central WCT', text, flags=re.M)
    assert n == 1
    text, n = re.subn(r'WCT_POSITIONS = np\.array\(\[.*?\], dtype=np\.float64\)',
        'WCT_POSITIONS = np.array([[GROUND_LENGTH * 0.5, GROUND_WIDTH * 0.5]], dtype=np.float64)', text, flags=re.S)
    assert n == 1
    (src / 'config.py').write_text(text, encoding='utf-8')
    cfg = json.loads((OLD / 'experiments/train_diverse.json').read_text(encoding='utf-8'))
    old_cfg = dict(cfg); cfg['wct_indices'] = [0]
    write(src / 'experiments/train_diverse.json', cfg)
    changes = {p.name: {'old': sha(OLD / p.name), 'new': sha(p)} for p in src.glob('*.py') if sha(p) != sha(OLD / p.name)}
    assert set(changes) == {'config.py'}
    assert {k for k in cfg if cfg[k] != old_cfg[k]} == {'wct_indices'}
    shutil.copy2(ROOT / 'experiments/run_wct1_center_20261107.py', OUT / 'run_wct1_center_20261107.py')
    # Preserve the independently checked evaluator; change only experiment routing and scope labels.
    ev = (BASE / 'evaluate_server_users.py').read_text(encoding='utf-8')
    ev = ev.replace('from server_users_runner_20261105 import', 'from run_wct1_center_20261107 import')
    ev = ev.replace('TRAIN_SEED = int(os.environ["IOTJ_TRAIN_SEED"])', 'TRAIN_SEED = 42')
    ev = ev.replace('Slurm server GPU; own training ended before this evaluation. Other-node processes not excluded. Diagnostic timings; not airborne or matched to workstation. Hardware recorded in task runtime.',
        'Local GPU; own training ended before evaluation. Other processes not excluded. Diagnostic workstation timing, not uncontended or airborne hardware measurements.')
    ev = ev.replace('One of the registered training seeds42/43/44; per-model episode bootstrap CI remains conditional. Aggregate3 training-seed means and sample SD separately.',
        'Only one freshly trained seed42 model. Test-episode bootstrap CI is conditional on this model, not training-seed robustness.')
    (OUT / 'evaluate_wct1_center_20261107.py').write_text(ev, encoding='utf-8')
    shutil.copy2(ROOT / 'experiments/probe_wct1_center_20261107.py', OUT / 'probe_wct1_center_20261107.py')
    write(OUT / 'protocol.json', dict(created_at=datetime.now().astimezone().isoformat(),
        authorization='User explicitly requested fresh K2 / one central WCT / unchanged hotspots; selected DARS only, seed42,10000 episodes.',
        algorithm='per_sac_cvx', label='DARS', training_seed=42, episodes=10000, warmup_steps=8000, batch_size=512, threads=2,
        K=2, M=30, wct_positions=[[250.,250.]], wct_indices=[0], state_dim=133, action_dim=36,
        original_source=str(OLD), source_changes=changes, environment_change={'wct_indices': {'old':[0,1,2,3], 'new':[0]}},
        reward='Original M * mean-active rho plus original battery penalty, NOT relabeled as sum-rho training reward.',
        objective='P1 total SLRR retained. Evaluation physical sum-rho averaged over fixed50 slots; zero future contribution after early termination.',
        unchanged='Original four hotspot centers and mobility/demand RNG,0.9/0.1 arrival probabilities,K2,M30,F_UAV3GHz,bandwidth1MHz,40kJ/UAV,network hidden sizes,Greedy association,safety,physics.',
        charging='One200W station shares its power across UAVs using unchanged charging rule. UAV initial positions remain in central225..275m region; central-WCT placement also changes initial accessibility.',
        smoke=dict(episodes=22,validation=2,warmup_steps=0,batch_size=512,test_episodes=2,formal_evidence=False),
        validation_episodes=50,test=dict(episodes=100,seeds=list(range(931000,931100)),horizon=50,layout='same single central WCT',paired_external_state_check=True),
        caveats=['One training seed only; no three-seed robustness claim.', 'No claim of benefit or survival before actual evaluation.', 'Weight-only checkpoints every500 episodes; no exact optimizer/replay/RNG continuation.', 'Server user-count batch remains explicitly paused; this run is independent.', 'Diagnostic workstation timing only.'],
        source_sha256={str(p.relative_to(src)).replace('\\','/'):sha(p) for p in src.rglob('*') if p.is_file()}))
    hashes={str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file()}
    write(OUT / 'package_hashes.json',hashes)
    write(OUT / 'run_status.json',dict(status='prepared',updated_at=datetime.now().astimezone().isoformat()))
    print(str(OUT))

if __name__ == '__main__': prepare()
