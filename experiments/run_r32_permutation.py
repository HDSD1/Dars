"""Frozen trained-Actor user permutation test, using archived training sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/reviewer_r32_permutation_20261102'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def worker(seed):
    import numpy as np
    import torch
    ref_path = ROOT / f'results/reviewer_phase2_three_seeds_20261101/formal/association/seed{seed}_base/manifest.json'
    ref = json.loads(ref_path.read_text(encoding='utf-8'))
    source, model = Path(ref['source_dir']), Path(ref['model'])
    assert sha(model) == ref['model_sha256']
    for name, digest in ref['source_sha256'].items():
        assert sha(source/name) == digest
    train = json.loads(Path(ref['model_manifest']).read_text(encoding='utf-8'))
    assert train['status'] == 'completed'
    sys.path.insert(0, str(source))
    from DARS import SACActor
    from config import K, M, STATE_DIM, ACTION_DIM
    from UAV_env import UAVEnv
    from experiment_config import EnvironmentConfig
    from state_normalization import StateNormalization
    torch.set_num_threads(1)
    torch.manual_seed(320260)
    actor = SACActor(STATE_DIM, ACTION_DIM).cpu().eval()
    checkpoint = torch.load(model, map_location='cpu', weights_only=True)
    actor.load_state_dict(checkpoint['actor'], strict=True)
    cfg = EnvironmentConfig(**ref['environment'])
    norm = StateNormalization(cfg.battery_capacity)
    folder = OUTPUT / f'seed{seed}'
    folder.mkdir()
    raw, ids = [], []
    episode_seed = 951000
    used = set(train.get('training_episode_seeds', [])) | set(train.get('validation_seeds', []))
    with torch.inference_mode():
        while len(raw) < 32:
            assert episode_seed not in used
            env = UAVEnv(use_cvx_alloc=True, experiment_config=cfg, seed=episode_seed)
            state = env.reset(seed=episode_seed)
            for slot in range(50):
                if slot in (0, 6, 12, 18, 24, 30, 36, 42):
                    raw.append(np.array(state, copy=True))
                    ids.append(dict(test_seed=episode_seed, slot=slot))
                    if len(raw) == 32:
                        break
                x = torch.tensor(norm.normalize(state), dtype=torch.float32).unsqueeze(0)
                action = actor.deterministic(x).numpy()[0]
                state, _, done, _ = env.step(action)
                if done:
                    break
            episode_seed += 1
    raw = np.asarray(raw)
    rng = np.random.default_rng(320261)
    perms = np.array([rng.permutation(M) for _ in range(320)])
    source_rows = np.repeat(np.arange(32), 10)
    perm_raw = raw[source_rows].copy()
    for j, p in enumerate(perms):
        s = raw[source_rows[j]]
        perm_raw[j, 3*K:3*K+2*M] = s[3*K:3*K+2*M].reshape(M, 2)[p].reshape(-1)
        perm_raw[j, 3*K+2*M:3*K+3*M] = s[3*K+2*M:3*K+3*M][p]
        perm_raw[j, 3*K+3*M:3*K+4*M] = s[3*K+3*M:3*K+4*M][p]
    x = torch.tensor(np.array([norm.normalize(s) for s in raw]), dtype=torch.float32)
    xp = torch.tensor(np.array([norm.normalize(s) for s in perm_raw]), dtype=torch.float32)
    with torch.inference_mode():
        a, ap = actor.deterministic(x).numpy(), actor.deterministic(xp).numpy()
        mu, ls = [v.numpy() for v in actor(x)]
        mup, lsp = [v.numpy() for v in actor(xp)]
    expected_mu, expected_ls = mu[source_rows].copy(), ls[source_rows].copy()
    expected_mu[:,3*K:] = mu[source_rows[:,None], 3*K+perms]
    expected_ls[:,3*K:] = ls[source_rows[:,None], 3*K+perms]
    off, offp = (a[:,3*K:]+1)/2, (ap[:,3*K:]+1)/2
    expected_off = off[source_rows[:,None], perms]
    offerr = np.abs(offp-expected_off)
    moverr = np.abs(ap[:,:3*K]-a[source_rows,:3*K])
    sigma, sigmap = np.exp(expected_ls), np.exp(lsp)
    tol = 1e-5
    rows=[]
    for j in range(320):
        rows.append(dict(training_seed=seed, state_index=int(source_rows[j]), **ids[source_rows[j]],
                         permutation_index=j%10, offload_max_abs_error=float(offerr[j].max()),
                         movement_normalized_max_abs_error=float(moverr[j].max()),
                         gaussian_mean_max_abs_error=float(np.abs(mup[j]-expected_mu[j]).max()),
                         gaussian_std_max_abs_error=float(np.abs(sigmap[j]-sigma[j]).max())))
    metrics = dict(training_seed=seed, states=32, permutations=320,
        offload_max_abs_error=float(offerr.max()),offload_mean_abs_error=float(offerr.mean()),
        movement_normalized_max_abs_error=float(moverr.max()),
        gaussian_mean_max_abs_error=float(np.abs(mup-expected_mu).max()),
        gaussian_std_max_abs_error=float(np.abs(sigmap-sigma).max()),
        mean_between_user_offload_std=float(off.std(axis=1).mean()),
        minimum_between_user_offload_std=float(off.std(axis=1).min()),
        offload_pass=bool(offerr.max()<=tol),movement_pass=bool(moverr.max()<=tol),
        distribution_parameters_pass=bool(np.allclose(mup,expected_mu,atol=tol,rtol=tol) and np.allclose(sigmap,sigma,atol=tol,rtol=tol)),
        nonconstant_outputs=bool(off.std(axis=1).min()>1e-4))
    metrics['passed'] = all(metrics[k] for k in ['offload_pass','movement_pass','distribution_parameters_pass','nonconstant_outputs'])
    np.savez_compressed(folder/'raw_test_arrays.npz',raw_states=raw,permutations=perms,source_rows=source_rows,
                        baseline_normalized_actions=a,permuted_normalized_actions=ap,
                        baseline_mean=mu,permuted_mean=mup,baseline_log_std=ls,permuted_log_std=lsp)
    write(folder/'state_ids.json', ids)
    with (folder/'permutation_errors.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert sha(model)==ref['model_sha256']
    write(folder/'report.json',dict(metrics=metrics,checkpoint=str(model),checkpoint_sha256=sha(model),
        source_dir=str(source),source_sha256=ref['source_sha256'],environment=ref['environment'],
        architecture=checkpoint.get('architecture'),torch_version=str(torch.__version__),device='cpu',dtype='float32',
        checkpoint_unchanged=True,absolute_action_tolerance=tol,parameter_atol=tol,parameter_rtol=tol,
        scope='Actor proposal before association/safety. Complete user tuples permuted, UAV/WCT/time fixed. Physical offload proposals in [0,1]; movement actions in normalized [-1,1] coordinates.'))
    print(json.dumps(metrics),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);args=p.parse_args()
    if args.seed:
        worker(args.seed)
        return
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT/'analysis_source.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    write(OUTPUT/'protocol.json',dict(training_seeds=[42,43,44],states_per_model=32,permutations_per_state=10,
        sampling='First 32 available states at zero-based slots 0,6,12,18,24,30,36,42 from frozen deterministic trajectories with consecutive independent episode seeds starting951000; no outcome-based exclusion.',
        permutation_rng_seed=320261,absolute_action_tolerance=1e-5,parameter_atol=1e-5,parameter_rtol=1e-5,
        offload_definition='(tanh(mean)+1)/2, before association and safety.',
        no_training=True,manuscript_unchanged=True))
    metrics=[]
    for seed in [42,43,44]:
        result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--seed',str(seed)],capture_output=True,text=True,encoding='utf-8')
        (OUTPUT/f'seed{seed}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Seed {seed} test execution failed; inspect log')
        row=json.loads((OUTPUT/f'seed{seed}/report.json').read_text(encoding='utf-8'))['metrics']
        metrics.append(row)
        print(json.dumps(row),flush=True)
    write(OUTPUT/'summary.json',dict(status='completed',all_passed=all(r['passed'] for r in metrics),
        models=3,states=96,permutations=960,per_model=metrics))
    with (OUTPUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)


if __name__=='__main__':
    main()
