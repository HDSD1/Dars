"""Disposable timing experiment; does not save or select training models."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from train_unified import ALGO_REGISTRY
from experiment_config import EnvironmentConfig
from UAV_env import UAVEnv
from state_normalization import StateNormalization
import config

p = argparse.ArgumentParser()
p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
p.add_argument('--threads', type=int, default=24)
p.add_argument('--algos', nargs='+', default=['per_sac_cvx'])
p.add_argument('--label', required=True)
args = p.parse_args()
torch.set_num_threads(args.threads)
if args.device == 'cuda':
    assert torch.cuda.is_available()
modules = ['DARS', 'esac_per_agent', 'sac_agent', 'td3_agent', 'maddpg_agent']
for name in modules:
    __import__(name).DEVICE = torch.device(args.device)

def sync():
    if args.device == 'cuda':
        torch.cuda.synchronize()

cfg = EnvironmentConfig.from_json(ROOT / 'experiments/train_diverse.json')
report = dict(label=args.label, device=args.device, threads=args.threads,
              python=sys.executable, torch=torch.__version__, numpy=np.__version__,
              gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              batch_size=config.BATCH_SIZE, warmup_updates=8, measured_updates=24,
              note='Timing only: 1100 real random transitions, then policy+environment+store+learn. No checkpoint saved.', results=[])
for algo in args.algos:
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    agent, _, kwargs = ALGO_REGISTRY[algo]()
    env = UAVEnv(**kwargs, experiment_config=cfg, seed=987654)
    norm = StateNormalization(env.battery_capacity)
    state = env.reset(seed=987654)
    for i in range(1100):
        action = np.random.uniform(-1, 1, env.action_dim)
        nxt, reward, done, info = env.step(action)
        agent.store(norm.normalize(state), action, reward, norm.normalize(nxt), done)
        state = env.reset(seed=987655+i) if done else nxt
    assert agent.buffer.ready(config.BATCH_SIZE)
    samples = []
    for i in range(32):
        sync()
        start = time.perf_counter()
        s = norm.normalize(state)
        action = agent.select_action(s, noise_std=0.3, use_ou=True)
        sync()
        inferred = time.perf_counter()
        nxt, reward, done, info = env.step(action)
        agent.store(s, action, reward, norm.normalize(nxt), done)
        stepped = time.perf_counter()
        metrics = agent.learn()
        sync()
        ended = time.perf_counter()
        if isinstance(metrics, dict):
            assert all(np.isfinite(float(v)) for v in metrics.values())
        if i >= 8:
            samples.append([(inferred-start)*1000, (stepped-inferred)*1000,
                            (ended-stepped)*1000, (ended-start)*1000])
        state = env.reset(seed=988800+i) if done else nxt
    values = np.array(samples)
    result = dict(algorithm=algo, inference_ms=values[:,0].mean(),
                  environment_store_ms=values[:,1].mean(), learn_ms=values[:,2].mean(),
                  total_ms=values[:,3].mean(), median_ms=np.median(values[:,3]),
                  samples_ms=samples)
    report['results'].append(result)
    print(json.dumps({k:v for k,v in result.items() if k!='samples_ms'}), flush=True)
    Path(__file__).with_name(args.label + '.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    del agent
    if args.device == 'cuda':
        torch.cuda.empty_cache()
