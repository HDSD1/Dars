"""Reproduce a small development run; this is not a full training experiment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='review_smoke')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / 'results' / args.tag
    if destination.exists():
        raise FileExistsError('Choose a new --tag to preserve previous results')
    destination.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    commands = [
        ('tests.log', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']),
        ('train.log', [sys.executable, '-u', 'train_unified.py', '--algo', 'per_sac_cvx',
            '--episodes', '20', '--seed', '42', '--tag', args.tag,
            '--env-config', 'experiments/train_diverse.json', '--warmup-steps', '128',
            '--batch-size', '32', '--eval-episodes', '3', '--threads', '1']),
        ('evaluation.log', [sys.executable, '-u', 'evaluate_suite.py', '--manifests',
            str(destination / 'per_sac_cvx.manifest.json'), '--suite', 'smoke',
            '--episodes', '3', '--output', str(destination / 'evaluation'), '--threads', '1']),
    ]
    for filename, command in commands:
        print('Running:', ' '.join(command), flush=True)
        with (destination / filename).open('w', encoding='utf-8') as stream:
            subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('Completed:', destination, flush=True)


if __name__ == '__main__':
    main()
