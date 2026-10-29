# DARS revision 3.2: user/action correspondence

The Actor retains row-wise TDA tokens. Its movement head receives global context;
its parameter-shared offloading head receives each user token and global context.
Both Critics share a Critic-side TDA (independent of the Actor's TDA), then use
separate nonlinear user/action pair encoders and value networks. Pairing precedes
mean pooling. The action layout remains `3K` movement components followed by `M`
normalized offloading proposals.

SAC log-probabilities sum all squashed-Gaussian proposal dimensions. Replay keeps
the original proposal, including when the environment corrects or rejects it.
Terminal n-step prefixes are flushed once with terminal flags, so they do not
bootstrap. Actor/temperature updates are delayed; target Critics are soft-updated
after every Critic update, as specified in the revised Algorithm 1.

## Verification

Run from the repository root:

```bash
python -m unittest discover -s tests -v
python tools/user_order_permutation_test.py --untrained --samples 32 --permutations 10 --seed 42 --device cpu --output validation/user_order_untrained_seed42.json
```

The 20 regression tests cover energy admission, proposal replay, terminal returns,
token/action equivariance, paired-noise sampling and joint log-probability,
Critic invariance and sensitivity to mismatched pairs, gradients, checkpoint
round-trip, and the training update schedule (including a short SAC update).

The three committed JSON reports use seeds 42, 43, and 44, each with 32 environment
reset states and 10 permutations per state, K=2, M=30, CPU float32, Python 3.12,
and PyTorch 2.15.0+cpu. All 960 cases pass. Across reports, the maximum offloading
ratio error is 5.96e-8, movement-action error is 2.98e-8, and joint-permutation
Critic error is 2.05e-8. The test checks non-uniform user outputs and maximum
errors, not only averages. Gaussian parameters and Q-values use absolute plus
relative tolerance 1e-5; action errors use absolute tolerance 1e-5.

**These reports use random initialization. They verify architecture consistency,
not trained-policy performance or convergence.** The regression suite also checks
Actor equivariance after short SAC updates. A full retraining experiment has not
been run as part of this code revision.

## Retraining and trained-policy verification

Old pooled-only Actor/Critic checkpoints are incompatible with this architecture.
Loading them raises an explicit error; do not use non-strict loading to silently
reuse the old heads. Train with a new experiment tag to preserve old results:

```bash
python train_unified.py --algo per_sac_cvx --seed 42 --tag userwise_revision_seed42 --hotspot
python tools/user_order_permutation_test.py --checkpoint results/userwise_revision_seed42/per_sac_cvx.pth --samples 32 --permutations 10 --seed 42 --device cpu --output validation/user_order_trained_seed42.json
```

The training command uses the script's default episode count; set `--episodes`
to the paper's training budget for the final experiment. Repeat for the other
experimental seeds. The checkpoint test strictly loads both Actor and Critics;
failure or nearly uniform user outputs return a nonzero exit status.

The permutation claim here concerns proposal generation and Critic evaluation.
It does not assert that the environment's index-based tie resolution for exactly
equal association distances is permutation-invariant.
