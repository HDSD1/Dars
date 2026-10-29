"""Permutation-equivariance test for DARS per-user offloading actions.

The test keeps the physical UAV state fixed, permutes each user's complete
feature tuple (location, task size, active flag), and checks whether the
policy's offloading outputs undergo the same permutation. Also check Gaussian
parameters and the Critics under joint state/action permutation. Use a revised
checkpoint for a learned-policy test, or --untrained for an explicitly labelled
architecture-only check (not a performance experiment).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import ACTION_DIM, K, M, STATE_DIM
from DARS import SACActor, TwinCritic, decompose_state
from UAV_env import UAVEnv
from state_normalization import StateNormalization


def permute_user_order(state: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """Reorder location, task, and activity blocks using new->old indices."""
    result = np.asarray(state, dtype=np.float64).copy()
    location_start = 3 * K
    location_end = location_start + 2 * M
    task_start = location_end
    task_end = task_start + M
    active_start = task_end
    active_end = active_start + M

    result[location_start:location_end] = (
        state[location_start:location_end].reshape(M, 2)[permutation].reshape(-1)
    )
    result[task_start:task_end] = state[task_start:task_end][permutation]
    result[active_start:active_end] = state[active_start:active_end][permutation]
    return result


def load_models(checkpoint_path, device):
    actor = SACActor(STATE_DIM, ACTION_DIM).to(device)
    critic = TwinCritic(STATE_DIM, ACTION_DIM).to(device)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not {'actor', 'critic'} <= checkpoint.keys():
            raise ValueError("Checkpoint must contain Actor and Critic state dictionaries")
        if 'offload_mean_head.weight' not in checkpoint['actor']:
            raise ValueError("Legacy pooled-only checkpoint; retrain the revised user-wise model")
        actor.load_state_dict(checkpoint['actor'], strict=True)
        critic.load_state_dict(checkpoint['critic'], strict=True)
    actor.eval()
    critic.eval()
    return actor, critic


def sample_states(sample_count: int, seed: int) -> np.ndarray:
    env = UAVEnv(use_cvx_alloc=True, use_hotspot=True)
    states = []
    for sample_index in range(sample_count):
        np.random.seed(seed + sample_index)
        state = env.reset()
        states.append(np.asarray(state, dtype=np.float64))
    return np.stack(states)


def run_test(args: argparse.Namespace) -> dict:
    if ACTION_DIM != 3 * K + M:
        raise ValueError(
            f"Expected 3K+M action dimensions for per-user offloading, got {ACTION_DIM}"
        )
    if args.samples < 1 or args.permutations < 1:
        raise ValueError("samples and permutations must both be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no GPU is visible to this process")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actor, critic = load_models(checkpoint_path, device)

    raw_states = sample_states(args.samples, args.seed)
    normalizer = StateNormalization()
    base_states = np.stack([normalizer.normalize(state) for state in raw_states])

    rng = np.random.default_rng(args.seed + 1)
    permuted_states = []
    sample_rows = []
    permutations = []
    for sample_index, state in enumerate(raw_states):
        for _ in range(args.permutations):
            permutation = rng.permutation(M)
            permuted_states.append(
                normalizer.normalize(permute_user_order(state, permutation))
            )
            sample_rows.append(sample_index)
            permutations.append(permutation)

    base_tensor = torch.as_tensor(base_states, dtype=torch.float32, device=device)
    perm_tensor = torch.as_tensor(
        np.asarray(permuted_states), dtype=torch.float32, device=device
    )
    sample_rows_array = np.asarray(sample_rows, dtype=np.int64)
    permutations_array = np.asarray(permutations, dtype=np.int64)

    with torch.inference_mode():
        base_actions = actor.deterministic(base_tensor).cpu().numpy()
        permuted_actions = actor.deterministic(perm_tensor).cpu().numpy()

        _, base_user_features = decompose_state(base_tensor)
        _, perm_user_features = decompose_state(perm_tensor)
        base_user_embeddings = actor.user_attn(base_user_features).cpu().numpy()
        perm_user_embeddings = actor.user_attn(perm_user_features).cpu().numpy()
        base_mean, base_logstd = [x.cpu().numpy() for x in actor(base_tensor)]
        perm_mean, perm_logstd = [x.cpu().numpy() for x in actor(perm_tensor)]
        # Random proposals avoid an uninformative near-uniform action vector
        # when checking the Critic's user/action pairing.
        base_q_actions = rng.uniform(-1, 1, (args.samples, ACTION_DIM)).astype(np.float32)
        perm_q_actions = base_q_actions[sample_rows_array].copy()
        perm_q_actions[:, 3*K:] = base_q_actions[sample_rows_array[:, None], 3*K+permutations_array]
        base_q = torch.cat(critic(base_tensor, torch.as_tensor(base_q_actions, device=device)), dim=1).cpu().numpy()
        perm_q = torch.cat(critic(perm_tensor, torch.as_tensor(perm_q_actions, device=device)), dim=1).cpu().numpy()

    base_rho = (base_actions[:, 3 * K:] + 1.0) / 2.0
    permuted_rho = (permuted_actions[:, 3 * K:] + 1.0) / 2.0
    expected_rho = base_rho[sample_rows_array[:, None], permutations_array]

    # Map each permuted output back to its original user identity before
    # comparing it with that user's unpermuted decision.
    inverse_permutations = np.argsort(permutations_array, axis=1)
    aligned_rho = np.take_along_axis(
        permuted_rho, inverse_permutations, axis=1
    )
    aligned_error = np.abs(aligned_rho - base_rho[sample_rows_array])
    equivariance_error = np.abs(permuted_rho - expected_rho)
    same_index_error = np.abs(permuted_rho - base_rho[sample_rows_array])

    movement_error = np.abs(
        permuted_actions[:, : 3 * K] - base_actions[sample_rows_array, : 3 * K]
    )
    embedding_error = np.abs(
        perm_user_embeddings - base_user_embeddings[sample_rows_array]
    )
    expected_mean = base_mean[sample_rows_array].copy()
    expected_logstd = base_logstd[sample_rows_array].copy()
    expected_mean[:, 3*K:] = base_mean[sample_rows_array[:, None], 3*K+permutations_array]
    expected_logstd[:, 3*K:] = base_logstd[sample_rows_array[:, None], 3*K+permutations_array]
    mean_error = np.abs(perm_mean - expected_mean)
    logstd_error = np.abs(perm_logstd - expected_logstd)
    q_error = np.abs(perm_q - base_q[sample_rows_array])
    per_user = []
    for user_index in range(M):
        per_user.append(
            {
                "user_index": user_index,
                "baseline_mean_rho": float(np.mean(base_rho[:, user_index])),
                "permuted_aligned_mean_rho": float(np.mean(aligned_rho[:, user_index])),
                "mean_absolute_alignment_error": float(
                    np.mean(aligned_error[:, user_index])
                ),
            }
        )

    action_spread = float(np.mean(np.std(base_rho, axis=1)))
    equivariance_mae = float(np.mean(equivariance_error))
    max_equivariance_error = float(np.max(equivariance_error))
    if action_spread <= args.uniform_tolerance:
        conclusion = "UNINFORMATIVE_UNIFORM_OUTPUT"
        passed = False
    else:
        # A mean error can hide individual misassigned users: check maxima.
        passed = (max_equivariance_error <= args.tolerance
                  and float(movement_error.max()) <= args.tolerance
                  and np.allclose(perm_mean, expected_mean, atol=args.tolerance, rtol=args.tolerance)
                  and np.allclose(perm_logstd, expected_logstd, atol=args.tolerance, rtol=args.tolerance)
                  and np.allclose(perm_q, base_q[sample_rows_array], atol=args.tolerance, rtol=args.tolerance))
        conclusion = "PASS" if passed else "FAIL"

    return {
        "experiment": "user_order_permutation_equivariance",
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() if checkpoint_path else None,
        "model_source": "checkpoint" if checkpoint_path else "random_initialization_architecture_only",
        "torch_version": str(torch.__version__),
        "K": K,
        "M": M,
        "seed": args.seed,
        "samples": args.samples,
        "permutations_per_sample": args.permutations,
        "total_permutations": len(permutations_array),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "action_dim": ACTION_DIM,
        "uav_movement_dim": 3 * K,
        "user_offload_dim": M,
        "metrics": {
            "pooled_user_embedding_max_abs_difference": float(np.max(embedding_error)),
            "movement_action_mean_abs_difference": float(np.mean(movement_error)),
            "movement_action_max_abs_difference": float(np.max(movement_error)),
            "gaussian_mean_max_abs_error": float(np.max(mean_error)),
            "gaussian_logstd_max_abs_error": float(np.max(logstd_error)),
            "critic_joint_permutation_max_abs_error": float(np.max(q_error)),
            "offload_equivariance_mae_rho": equivariance_mae,
            "offload_equivariance_max_abs_error_rho": max_equivariance_error,
            "offload_same_index_mae_rho": float(np.mean(same_index_error)),
            "mean_between_user_rho_std": action_spread,
            "fraction_user_actions_within_tolerance": float(
                np.mean(equivariance_error <= args.tolerance)
            ),
        },
        "tolerance_rho": args.tolerance,
        "uniform_output_tolerance": args.uniform_tolerance,
        "passed": passed,
        "conclusion": conclusion,
        "interpretation": (
            "A pass requires the offloading decision to follow the same user when user "
            "feature tuples are reordered. Nearly uniform outputs are reported as "
            "uninformative rather than as evidence of user-specific association."
        ),
        "per_user": per_user,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test permutation consistency of trained DARS offloading actions"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--untrained", action="store_true",
                        help="Check random initialization only, not a trained policy")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--uniform-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("user_order_permutation_report.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    report = run_test(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "checkpoint": report["checkpoint"],
        "model_source": report["model_source"],
        "device": report["device"],
        "gpu_name": report["gpu_name"],
        "metrics": report["metrics"],
        "conclusion": report["conclusion"],
        "report": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
