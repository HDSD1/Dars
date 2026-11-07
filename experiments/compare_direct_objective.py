#!/usr/bin/env python
"""Paired, frozen-policy comparison: closed-form resources vs direct SLRR objective."""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

# Keep this audit on CPU so it does not compete with the separate training job.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def summarize(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "p95": None}
    import numpy as np
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "p95": float(np.percentile(vals, 95)),
    }


def make_timer_wrappers(env):
    """Measure closed-form allocator calls made by the real CF environment step."""
    counters = {"cpu_seconds": 0.0, "bandwidth_seconds": 0.0, "cpu_calls": 0, "bandwidth_calls": 0}
    original_cpu = env._compute_allocation
    original_bw = env._compute_bandwidth_allocation

    def timed_cpu(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_cpu(*args, **kwargs)
        finally:
            counters["cpu_seconds"] += time.perf_counter() - start
            counters["cpu_calls"] += 1

    def timed_bw(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_bw(*args, **kwargs)
        finally:
            counters["bandwidth_seconds"] += time.perf_counter() - start
            counters["bandwidth_calls"] += 1

    env._compute_allocation = timed_cpu
    env._compute_bandwidth_allocation = timed_bw
    return counters


def _solve_one_uav(indices, tasks, ratios, cpu_cf, bw_cf, uav_pos, ue_pos,
                   energy_budget_j, channel_gain, constants, maxiter):
    """Solve one convex epigraph subproblem in normalized CPU/BW shares."""
    import numpy as np
    from scipy.optimize import minimize

    F_UAV = constants["F_UAV"]
    BANDWIDTH = constants["BANDWIDTH"]
    F_UE = constants["F_UE"]
    S_CYCLES = constants["S_CYCLES"]
    KAPPA = constants["KAPPA"]
    P_TX = constants["P_TX"]
    SIGMA2 = constants["SIGMA2"]

    ids = np.asarray(indices, dtype=int)
    n = len(ids)
    if n == 0:
        return {"success": True, "trivial": True, "seconds": 0.0, "solver_seconds": 0.0,
                "indices": [], "cpu": [], "bandwidth": [], "delay": [], "residuals": {},
                "message": "no offloaded users", "iterations": 0}

    start_total = time.perf_counter()
    sizes = np.asarray(tasks[ids], dtype=float)
    x = np.asarray(ratios[ids], dtype=float)
    cycles = sizes * S_CYCLES
    base = cycles / F_UE
    local = (1.0 - x) * base
    comp_coeff = x * cycles / F_UAV

    snr_logs = []
    for i in ids:
        gain = float(channel_gain(uav_pos, ue_pos[i]))
        snr = P_TX * gain / SIGMA2
        if snr <= 0.0:
            snr = 1e-10
        snr_logs.append(math.log2(1.0 + snr))
    snr_logs = np.asarray(snr_logs, dtype=float)
    tx_coeff = x * sizes / (BANDWIDTH * snr_logs)
    energy_coeff = KAPPA * x * cycles * (F_UAV ** 2)
    budget = float(energy_budget_j)
    if not math.isfinite(budget) or budget < -1e-8:
        return {"success": False, "trivial": False, "seconds": time.perf_counter() - start_total,
                "solver_seconds": 0.0, "indices": ids.tolist(), "delay": [],
                "message": "invalid negative energy budget", "iterations": 0, "residuals": {}}
    budget = max(0.0, budget)

    u0 = np.asarray(cpu_cf[ids], dtype=float) / F_UAV
    v0 = np.asarray(bw_cf[ids], dtype=float) / BANDWIDTH
    floor = 1e-8
    if np.any(~np.isfinite(u0)) or np.any(~np.isfinite(v0)) or np.any(u0 <= 0.0) or np.any(v0 <= 0.0):
        return {"success": False, "trivial": False, "seconds": time.perf_counter() - start_total,
                "solver_seconds": 0.0, "indices": ids.tolist(), "delay": [],
                "message": "CF warm start has nonpositive allocation", "iterations": 0, "residuals": {}}
    if np.sum(u0) > 1.0 + 1e-7 or np.sum(v0) > 1.0 + 1e-7:
        return {"success": False, "trivial": False, "seconds": time.perf_counter() - start_total,
                "solver_seconds": 0.0, "indices": ids.tolist(), "delay": [],
                "message": "CF warm start exceeds resource cap", "iterations": 0, "residuals": {}}

    # Scale each epigraph delay by its own T_base. This keeps SLSQP variables
    # and constraints near O(1), including when task sizes vary across users.
    local_ratio = local / base
    comp_ratio = comp_coeff / base
    tx_ratio = tx_coeff / base
    edge0 = comp_ratio / u0 + tx_ratio / v0
    t0 = np.maximum(local_ratio, edge0)
    z0 = np.concatenate([u0, v0, t0])

    def objective(z):
        return float(np.sum(z[2*n:3*n]))

    def objective_jac(z):
        grad = np.zeros(3*n, dtype=float)
        grad[2*n:3*n] = 1.0
        return grad

    def cpu_cap(z):
        return 1.0 - float(np.sum(z[:n]))

    def cpu_cap_jac(z):
        g = np.zeros(3*n, dtype=float)
        g[:n] = -1.0
        return g

    def bw_cap(z):
        return 1.0 - float(np.sum(z[n:2*n]))

    def bw_cap_jac(z):
        g = np.zeros(3*n, dtype=float)
        g[n:2*n] = -1.0
        return g

    energy_scale = max(budget, 1.0)

    def energy_cap(z):
        return (budget - float(np.dot(energy_coeff, z[:n] ** 2))) / energy_scale

    def energy_cap_jac(z):
        g = np.zeros(3*n, dtype=float)
        g[:n] = -2.0 * energy_coeff * z[:n] / energy_scale
        return g

    def local_epigraph(z):
        return z[2*n:3*n] - local_ratio

    def local_epigraph_jac(z):
        g = np.zeros((n, 3*n), dtype=float)
        g[:, 2*n:3*n] = np.eye(n)
        return g

    def edge_epigraph(z):
        u = np.maximum(z[:n], floor)
        v = np.maximum(z[n:2*n], floor)
        return z[2*n:3*n] - comp_ratio / u - tx_ratio / v

    def edge_epigraph_jac(z):
        u = np.maximum(z[:n], floor)
        v = np.maximum(z[n:2*n], floor)
        g = np.zeros((n, 3*n), dtype=float)
        g[:, :n] = np.diag(comp_ratio / (u ** 2))
        g[:, n:2*n] = np.diag(tx_ratio / (v ** 2))
        g[:, 2*n:3*n] = np.eye(n)
        return g

    constraints = [
        {"type": "ineq", "fun": cpu_cap, "jac": cpu_cap_jac},
        {"type": "ineq", "fun": bw_cap, "jac": bw_cap_jac},
        {"type": "ineq", "fun": energy_cap, "jac": energy_cap_jac},
        {"type": "ineq", "fun": local_epigraph, "jac": local_epigraph_jac},
        {"type": "ineq", "fun": edge_epigraph, "jac": edge_epigraph_jac},
    ]
    bounds = [(floor, 1.0)] * (2*n) + [(0.0, None)] * n
    def assess(result):
        z = np.asarray(result.x, dtype=float)
        if z.shape != (3*n,) or np.any(~np.isfinite(z)):
            return {"success": False, "z": z, "candidate_j": float("inf"),
                    "initial_j": float(np.sum(t0)), "normalized_resid": float("inf")}
        u = np.maximum(z[:n], floor)
        v = np.maximum(z[n:2*n], floor)
        t = z[2*n:3*n]
        edge = comp_coeff / u + tx_coeff / v
        energy_used = float(np.dot(energy_coeff, u ** 2))
        cpu_resid = max(0.0, float(np.sum(u) - 1.0))
        bw_resid = max(0.0, float(np.sum(v) - 1.0))
        energy_resid_j = max(0.0, energy_used - budget)
        local_ratio_check = local / base
        edge_ratio_check = edge / base
        epigraph_resid = max(0.0, float(np.max(local_ratio_check - t)),
                             float(np.max(edge_ratio_check - t)))
        normalized_resid = max(cpu_resid, bw_resid,
                               energy_resid_j / energy_scale,
                               epigraph_resid)
        epigraph_slack = max(0.0, float(np.max(t - np.maximum(local_ratio_check, edge_ratio_check))))
        candidate_j = float(np.sum(np.maximum(local, edge) / base))
        # edge0 and t0 are already normalized by T_base.
        initial_j = float(np.sum(t0))
        no_worse_than_cf = candidate_j <= initial_j + 1e-7
        return {
            "success": bool(result.success and normalized_resid <= 1e-5 and no_worse_than_cf),
            "z": z, "u": u, "v": v, "t": t, "edge": edge,
            "candidate_j": candidate_j, "initial_j": initial_j,
            "cpu_resid": cpu_resid, "bw_resid": bw_resid,
            "energy_resid_j": energy_resid_j, "epigraph_resid": epigraph_resid,
            "epigraph_slack": epigraph_slack, "normalized_resid": normalized_resid,
            "no_worse_than_cf": no_worse_than_cf,
        }

    solver_start = time.perf_counter()
    result = minimize(objective, z0, method="SLSQP", jac=objective_jac, bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": int(maxiter), "ftol": 1e-10, "disp": False})
    solver_seconds = time.perf_counter() - solver_start
    assessment = assess(result)
    attempts = [{"message": str(result.message), "iterations": int(getattr(result, "nit", 0)),
                 "success_flag": bool(result.success)}]
    if not assessment["success"]:
        # SLSQP can stop on a line-search warning at an otherwise feasible
        # near-optimum. Restart from that allocation with a fresh epigraph,
        # then try the original feasible CF point if needed.
        starts = []
        if assessment.get("u") is not None and np.all(np.isfinite(assessment["u"])):
            u_start = np.clip(assessment["u"], floor, 1.0)
            v_start = np.clip(assessment["v"], floor, 1.0)
            if np.sum(u_start) <= 1.0 + 1e-6 and np.sum(v_start) <= 1.0 + 1e-6:
                edge_start = comp_ratio / u_start + tx_ratio / v_start
                starts.append(np.concatenate([u_start, v_start,
                                              np.maximum(local_ratio, edge_start)]))
        starts.append(z0)
        for retry_start in starts:
            retry_started = time.perf_counter()
            retry = minimize(objective, retry_start, method="SLSQP", jac=objective_jac,
                             bounds=bounds, constraints=constraints,
                             options={"maxiter": max(int(maxiter) * 2, 500),
                                      "ftol": 1e-9, "disp": False})
            solver_seconds += time.perf_counter() - retry_started
            retry_assessment = assess(retry)
            attempts.append({"message": str(retry.message),
                             "iterations": int(getattr(retry, "nit", 0)),
                             "success_flag": bool(retry.success)})
            retry_primal_valid = (retry_assessment["normalized_resid"] <= 1e-5
                                  and retry_assessment.get("no_worse_than_cf", False))
            if retry_assessment["success"] or (
                    retry_primal_valid and retry_assessment["candidate_j"] < assessment["candidate_j"]):
                result, assessment = retry, retry_assessment
            if assessment["success"]:
                break

    z = assessment["z"]
    u = assessment.get("u", np.full(n, np.nan))
    v = assessment.get("v", np.full(n, np.nan))
    edge = assessment.get("edge", np.full(n, np.nan))
    candidate_j = assessment["candidate_j"]
    initial_j = assessment["initial_j"]
    cpu_resid = assessment.get("cpu_resid", float("inf"))
    bw_resid = assessment.get("bw_resid", float("inf"))
    energy_resid_j = assessment.get("energy_resid_j", float("inf"))
    epigraph_resid = assessment.get("epigraph_resid", float("inf"))
    epigraph_slack = assessment.get("epigraph_slack", float("inf"))
    normalized_resid = assessment["normalized_resid"]
    success = assessment["success"]
    total_seconds = time.perf_counter() - start_total
    return {
        "success": success,
        "trivial": False,
        "solver_success_flag": bool(result.success),
        "seconds": float(total_seconds),
        "solver_seconds": float(solver_seconds),
        "indices": ids.tolist(),
        "cpu": (u * F_UAV).tolist(),
        "bandwidth": (v * BANDWIDTH).tolist(),
        "delay": np.maximum(local, edge).tolist(),
        "candidate_j": candidate_j,
        "initial_cf_j": initial_j,
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "attempts": attempts,
        "residuals": {
            "cpu_capacity_fraction": cpu_resid,
            "bandwidth_capacity_fraction": bw_resid,
            "energy_excess_j": energy_resid_j,
            "epigraph_understatement_normalized": epigraph_resid,
            "epigraph_slack_normalized": epigraph_slack,
            "normalized_max": normalized_resid,
        },
    }


def solve_reference(active, tasks, ratios, assoc, cpu_cf, bw_cf, uav_pos, ue_pos,
                    energy_budgets, channel_gain, constants, maxiter):
    import numpy as np

    active_ids = np.flatnonzero(active)
    opt_delays = np.zeros_like(tasks, dtype=float)
    cpu_opt = np.zeros_like(tasks, dtype=float)
    bw_opt = np.zeros_like(tasks, dtype=float)
    base = tasks[active_ids] * constants["S_CYCLES"] / constants["F_UE"]
    if len(active_ids):
        opt_delays[active_ids] = base

    offload_ids = [int(i) for i in active_ids
                   if assoc[i] >= 0 and ratios[i] > constants["OFFLOAD_EPS"]]
    by_uav = []
    total_seconds = 0.0
    solver_seconds = 0.0
    all_success = True
    for k in range(constants["K"]):
        ids = [i for i in offload_ids if int(assoc[i]) == k]
        result = _solve_one_uav(ids, tasks, ratios, cpu_cf, bw_cf,
                                uav_pos[k], ue_pos, energy_budgets[k],
                                channel_gain, constants, maxiter)
        by_uav.append(result)
        total_seconds += float(result.get("seconds", 0.0))
        solver_seconds += float(result.get("solver_seconds", 0.0))
        all_success = all_success and bool(result["success"])
        if not result["success"]:
            continue
        for j, i in enumerate(result["indices"]):
            cpu_opt[i] = result["cpu"][j]
            bw_opt[i] = result["bandwidth"][j]
            opt_delays[i] = result["delay"][j]

    has_numerical_solve = bool(offload_ids)
    if not offload_ids:
        total_seconds = 0.0
        solver_seconds = 0.0
    return {
        "success": bool(all_success),
        "has_numerical_solve": has_numerical_solve,
        "offload_ids": offload_ids,
        "delays": opt_delays,
        "cpu": cpu_opt,
        "bandwidth": bw_opt,
        "seconds": float(total_seconds),
        "solver_seconds": float(solver_seconds),
        "uav_results": by_uav,
    }


def cf_delays_from_snapshot(active, tasks, ratios, assoc, cpu, bandwidth,
                            uav_pos, ue_pos, channel_gain, constants):
    import numpy as np
    result = np.zeros_like(tasks, dtype=float)
    for i in np.flatnonzero(active):
        cycles = float(tasks[i]) * constants["S_CYCLES"]
        base = cycles / constants["F_UE"]
        x = float(ratios[i])
        k = int(assoc[i])
        local = (1.0 - x) * base
        if k < 0 or x <= constants["OFFLOAD_EPS"]:
            result[i] = base
            continue
        gain = float(channel_gain(uav_pos[k], ue_pos[i]))
        snr = constants["P_TX"] * gain / constants["SIGMA2"]
        if snr <= 0.0:
            snr = 1e-10
        rate = float(bandwidth[i]) * math.log2(1.0 + snr)
        if cpu[i] <= 0.0 or rate <= 0.0:
            result[i] = float("inf")
        else:
            edge = x * cycles / float(cpu[i]) + x * float(tasks[i]) / rate
            result[i] = max(local, edge)
    return result


def summarize_results(states, episodes, manifest):
    import numpy as np
    valid = [r for r in states if r.get("reference_valid")]
    # Rejected proposals do not advance the environment and are not paired
    # decision states. Keep them in rejection counts, out of coverage's base.
    active_states = [r for r in states if int(r.get("active_users", 0)) > 0
                     and not r.get("environment_rejected", False)]
    solver_states = [r for r in states if r.get("solver_required")]
    solver_success_states = [r for r in solver_states if r.get("solver_success")]
    gaps = [r["relative_gap_percent"] for r in valid]
    slrr = [r["slrr_gain_mean_active"] for r in valid]
    sum_rho = [r["sum_rho_gain"] for r in valid]
    episode_gains = [r["slrr_fixed_horizon_gain"] for r in episodes
                     if r.get("episode_reference_valid") and r.get("slrr_fixed_horizon_gain") is not None]
    residual_rows = [r.get("constraint_residuals", {}) for r in valid if r.get("solver_required")]
    all_uav = [u for r in states for u in r.get("uav_solver", []) if not u.get("trivial", False)]
    return {
        "status": "completed" if len(episodes) == manifest["episodes"] else "partial",
        "episodes_completed": len(episodes),
        "requested_episodes": int(manifest["episodes"]),
        "active_state_count": len(active_states),
        "reference_valid_state_count": len(valid),
        "reference_invalid_state_count": int(len(active_states) - len(valid)),
        "reference_valid_coverage": float(len(valid) / max(1, len(active_states))),
        "relative_gap_percent": summarize(gaps),
        "slrr_gain_mean_active_user": summarize(slrr),
        "sum_rho_gain_per_state": summarize(sum_rho),
        "episode_slrr_fixed_horizon_gain": summarize(episode_gains),
        "numerical_solver_state_success_rate": {
            "success": len(solver_success_states), "attempted": len(solver_states),
            "rate": float(len(solver_success_states) / len(solver_states)) if solver_states else None,
        },
        "numerical_solver_uav_subproblem_success_rate": {
            "success": sum(bool(u.get("success")) for u in all_uav), "attempted": len(all_uav),
            "rate": float(sum(bool(u.get("success")) for u in all_uav) / len(all_uav)) if all_uav else None,
        },
        "allocation_time_ms_per_valid_state": {
            "cf_closed_form": summarize([r["cf_allocation_ms"] for r in valid]),
            "opt_formulation_plus_slsqp": summarize([r["opt_allocation_ms"] for r in valid]),
            "opt_solver_only_on_numeric_states": summarize([r["opt_solver_ms"] for r in solver_success_states]),
        },
        "constraint_residuals": {
            "max_cpu_capacity_fraction": max((v.get("cpu_capacity_fraction", 0.0) for v in residual_rows), default=0.0),
            "max_bandwidth_capacity_fraction": max((v.get("bandwidth_capacity_fraction", 0.0) for v in residual_rows), default=0.0),
            "max_energy_excess_j": max((v.get("energy_excess_j", 0.0) for v in residual_rows), default=0.0),
            "max_epigraph_understatement_normalized": max((v.get("epigraph_understatement_normalized", 0.0) for v in residual_rows), default=0.0),
            "p95_normalized_max": float(np.percentile([v.get("normalized_max", 0.0) for v in residual_rows], 95)) if residual_rows else 0.0,
            "max_normalized_max": max((v.get("normalized_max", 0.0) for v in residual_rows), default=0.0),
            "max_epigraph_slack_normalized": max((v.get("epigraph_slack_normalized", 0.0) for v in residual_rows), default=0.0),
        },
        "cf_delay_reconstruction_max_abs_error_s": max((r.get("cf_delay_reconstruction_max_abs_error_s", 0.0) for r in states), default=0.0),
        "edge_branch_dominant_share_among_offloaded_user_slots": summarize(
            [r["edge_branch_dominant_fraction"] for r in valid if r.get("edge_branch_dominant_fraction") is not None]),
        "offload_projection_state_share": float(
            sum(bool(r.get("offload_projected")) for r in active_states) / max(1, len(active_states))),
        "rejected_environment_attempts": int(sum(bool(r.get("environment_rejected")) for r in states)),
        "total_elapsed_seconds": float(time.time() - manifest["started_epoch"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, help="Exact training source snapshot directory")
    parser.add_argument("--model", required=True, help="Frozen DARS checkpoint (.pth)")
    parser.add_argument("--model-manifest", required=True, help="Manifest for the checkpoint")
    parser.add_argument("--env-config", required=True, help="Training-distribution environment JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-start", type=int, default=200000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--hot-prob", type=float, default=None,
                        help="Optional hotspot task-arrival probability override")
    parser.add_argument("--cold-prob", type=float, default=None,
                        help="Optional coldspot task-arrival probability override")
    parser.add_argument("--initial-energy", type=float, default=None,
                        help="Optional UAV initial-energy override in joules")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    model_path = Path(args.model).resolve()
    model_manifest_path = Path(args.model_manifest).resolve()
    env_config_path = Path(args.env_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not source_dir.is_dir() or not model_path.is_file() or not model_manifest_path.is_file():
        raise FileNotFoundError("Source snapshot, model, or model manifest missing")
    if not env_config_path.is_file():
        raise FileNotFoundError(env_config_path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    sys.path.insert(0, str(source_dir))

    import numpy as np
    import torch
    from scipy import __version__ as scipy_version
    from DARS import PERSACAgent
    from UAV_env import UAVEnv, OFFLOAD_EPS
    from config import (BANDWIDTH, F_UAV, F_UE, HOTSPOT_COLD_PROB, HOTSPOT_HOT_PROB,
                        K, KAPPA, P_TX, S_CYCLES, SIGMA2)
    from evaluation import configure_policy_environment, frozen_policy, model_digest, policy_input
    from experiment_config import EnvironmentConfig
    from state_normalization import StateNormalization

    torch.set_num_threads(1)
    cfg = EnvironmentConfig.from_json(env_config_path)
    if args.initial_energy is not None:
        cfg = replace(cfg, initial_energy=float(args.initial_energy))
    hot_prob = HOTSPOT_HOT_PROB if args.hot_prob is None else float(args.hot_prob)
    cold_prob = HOTSPOT_COLD_PROB if args.cold_prob is None else float(args.cold_prob)
    if not (0.0 <= hot_prob <= 1.0 and 0.0 <= cold_prob <= 1.0):
        raise ValueError("Hotspot and coldspot probabilities must be in [0, 1]")
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest.get("status") != "completed" or model_manifest.get("algorithm") != "per_sac_cvx":
        raise RuntimeError("Expected a completed per_sac_cvx model manifest")
    test_seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    train_seeds = set(int(s) for s in model_manifest.get("training_episode_seeds", []))
    validation_seeds = set(int(s) for s in model_manifest.get("validation_seeds", []))
    overlap = sorted(set(test_seeds) & (train_seeds | validation_seeds))
    if overlap:
        raise RuntimeError(f"Test seeds overlap training/validation seeds: {overlap[:10]}")

    output_dir.mkdir(parents=True)
    manifest = {
        "experiment": "frozen_DARS_direct_original_objective_resource_allocation",
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "started_epoch": time.time(),
        "episodes": int(args.episodes),
        "seed_start": int(args.seed_start),
        "test_seeds": test_seeds,
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "training_manifest": str(model_manifest_path),
        "training_source_snapshot": str(source_dir),
        "training_source_train_unified_sha256": model_manifest.get("source_sha256", {}).get("train_unified.py"),
        "env_config": str(env_config_path),
        "environment": cfg.to_dict(),
        "task_arrival_probabilities": {"hot": hot_prob, "cold": cold_prob},
        "battery_initial_energy_j": float(cfg.initial_energy),
        "battery_capacity_j": float(cfg.battery_capacity),
        "solver": {"name": "SciPy SLSQP", "version": scipy_version, "maxiter": int(args.maxiter),
                   "formulation": "convex epigraph of sum_i max(Tloc,Tedge)/Tbase"},
        "policy": "frozen; no replay writes, learning updates, or OPT writeback to the environment",
        "paired_conditions": ["state", "UAV positions", "UE positions", "tasks", "channel gain",
                              "association", "safety-projected offload ratio", "per-UAV CPU/BW caps",
                              "per-UAV remaining computation-energy budget"],
        "timing_scope": "CPU allocator only; CF call duration inside the real environment step; OPT includes per-UAV formulation plus SLSQP; excludes policy inference and environment stepping",
        "valid_state_rule": "active users > 0, positive J_OPT, and a feasible successful reference; trivial no-offload states are valid with zero gap; rejected steps and failed references excluded",
        "episodes_completed": 0,
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                    "scipy": scipy_version, "device": "CPU", "torch_threads": torch.get_num_threads()},
    }
    write_json(output_dir / "manifest.json", manifest)
    state_path = output_dir / "state_results.jsonl"
    episode_path = output_dir / "episode_results.jsonl"
    state_path.write_text("", encoding="utf-8")
    episode_path.write_text("", encoding="utf-8")

    constants = {
        "F_UAV": float(F_UAV), "BANDWIDTH": float(BANDWIDTH), "F_UE": float(F_UE),
        "S_CYCLES": float(S_CYCLES), "KAPPA": float(KAPPA), "P_TX": float(P_TX),
        "SIGMA2": float(SIGMA2), "K": int(K), "OFFLOAD_EPS": float(OFFLOAD_EPS),
    }
    agent = PERSACAgent()
    agent.load(str(model_path))
    normalizer = StateNormalization(cfg.battery_capacity)
    digest_before = model_digest(agent)
    all_states = []
    all_episodes = []
    counts = {"environment_attempts": 0, "environment_rejected": 0, "active_states": 0,
              "accepted_states": 0, "reference_valid": 0, "solver_required": 0,
              "solver_success": 0, "cf_reconstruction_failures": 0}

    try:
        with frozen_policy(agent):
            for ep_idx, seed in enumerate(test_seeds):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                env = UAVEnv(use_cvx_alloc=True, experiment_config=cfg, seed=seed)
                # reset() installs the module defaults before the first task draw.
                import UAV_env as evaluation_environment
                evaluation_environment.HOTSPOT_HOT_PROB = hot_prob
                evaluation_environment.HOTSPOT_COLD_PROB = cold_prob
                configure_policy_environment(agent, env)
                state = env.reset(seed=seed)
                assert (env._hot_prob, env._cold_prob) == (hot_prob, cold_prob)
                timer = make_timer_wrappers(env)
                ep_active_states = 0
                ep_valid_states = 0
                ep_rho_cf_sum = 0.0
                ep_rho_opt_sum = 0.0
                ep_solver_failures = 0
                ep_executed_steps = 0
                ep_rejected_attempts = 0

                for step_idx in range(env.slot_num):
                    action = agent.select_action(policy_input(agent, normalizer, state), noise_std=0.0)
                    active_before = env.active_mask.copy()
                    task_before = env.task_list.copy()
                    ue_before = env.loc_ue.copy()
                    uav_before = env.loc_uav.copy()
                    timer["cpu_seconds"] = 0.0
                    timer["bandwidth_seconds"] = 0.0
                    timer["cpu_calls"] = 0
                    timer["bandwidth_calls"] = 0
                    next_state, reward, done, info = env.step(action)
                    counts["environment_attempts"] += 1
                    if not info.get("action_executed", False):
                        counts["environment_rejected"] += 1
                        ep_rejected_attempts += 1
                        row = {"seed": int(seed), "step": int(step_idx), "active_users": int(np.sum(active_before)),
                               "reference_valid": False, "environment_rejected": True,
                               "active_mask": active_before.astype(int).tolist(),
                               "task_sizes": task_before.tolist(),
                               "uav_positions_before_action": uav_before.tolist(),
                               "ue_positions_for_attempt": ue_before.tolist(),
                               "termination_reason": info.get("termination_reason")}
                        append_jsonl(state_path, row)
                        all_states.append(row)
                        if done:
                            break
                        state = next_state
                        continue
                    ep_executed_steps += 1
                    active_ids = np.flatnonzero(active_before)
                    if len(active_ids) == 0:
                        row = {"seed": int(seed), "step": int(step_idx), "active_users": 0,
                               "reference_valid": False, "environment_rejected": False,
                               "reason": "no_active_tasks"}
                        append_jsonl(state_path, row)
                        all_states.append(row)
                        state = next_state
                        if done:
                            break
                        continue

                    counts["active_states"] += 1
                    ep_active_states += 1
                    assoc = np.asarray(info["association"], dtype=int).copy()
                    ratios = np.asarray(info["offload_ratios"], dtype=float).copy()
                    cpu_cf = np.asarray(info["cpu_allocation"], dtype=float).copy()
                    bw_cf = np.asarray(info["bandwidth_allocation"], dtype=float).copy()
                    uav_after = env.loc_uav.copy()
                    budgets = np.asarray(info["computation_budget"], dtype=float).copy()
                    channel_gain_matrix = np.asarray([
                        [float(env._channel_gain(uav_after[k], ue_before[i])) for i in range(len(task_before))]
                        for k in range(K)
                    ], dtype=float)
                    cf_delays_formula = cf_delays_from_snapshot(
                        active_before, task_before, ratios, assoc, cpu_cf, bw_cf,
                        uav_after, ue_before, env._channel_gain, constants)
                    env_delays = np.asarray(info["delays"], dtype=float)
                    cf_reconstruction_error = float(np.max(np.abs(cf_delays_formula[active_ids] - env_delays[active_ids])))
                    counts["cf_reconstruction_failures"] += int(cf_reconstruction_error > 1e-6)

                    # The exact reference solve is a sidecar: it never changes env or next_state.
                    ref_start = time.perf_counter()
                    ref = solve_reference(active_before, task_before, ratios, assoc, cpu_cf, bw_cf,
                                          uav_after, ue_before, budgets, env._channel_gain,
                                          constants, args.maxiter)
                    opt_wall_seconds = time.perf_counter() - ref_start
                    base_times = task_before[active_ids] * S_CYCLES / F_UE
                    j_cf = float(np.sum(env_delays[active_ids] / base_times))
                    cf_rho = float(np.mean(np.asarray(info["rho"])[active_ids]))
                    constraint_residuals = {}
                    for uav_result in ref["uav_results"]:
                        if not uav_result.get("trivial", False):
                            for key, value in uav_result.get("residuals", {}).items():
                                old = constraint_residuals.get(key, 0.0)
                                constraint_residuals[key] = max(float(old), float(value))
                    j_opt = None
                    opt_rho = None
                    raw_gap = None
                    relative_gap = None
                    slrr_gain = None
                    sum_rho_gain = None
                    branch_fraction = None
                    if ref["success"] and cf_reconstruction_error <= 1e-6:
                        j_opt = float(np.sum(ref["delays"][active_ids] / base_times))
                        if j_opt <= 0.0 or not math.isfinite(j_opt):
                            ref["success"] = False
                        else:
                            raw_gap = (j_cf - j_opt) / j_opt
                            if raw_gap < -1e-6:
                                ref["success"] = False
                            else:
                                relative_gap = max(0.0, raw_gap) * 100.0
                                opt_rho = float(np.mean(1.0 - ref["delays"][active_ids] / base_times))
                                slrr_gain = opt_rho - cf_rho
                                sum_rho_gain = j_cf - j_opt
                                off_ids = np.asarray(ref["offload_ids"], dtype=int)
                                if len(off_ids):
                                    cf_loc = ((1.0 - ratios[off_ids]) * task_before[off_ids] * S_CYCLES / F_UE)
                                    cf_edge = cf_delays_formula[off_ids]
                                    branch_fraction = float(np.mean(cf_edge > cf_loc))
                    reference_valid = bool(ref["success"])
                    counts["accepted_states"] += 1
                    counts["reference_valid"] += int(reference_valid)
                    counts["solver_required"] += int(ref["has_numerical_solve"])
                    counts["solver_success"] += int(ref["has_numerical_solve"] and reference_valid)
                    if reference_valid:
                        ep_valid_states += 1
                        ep_rho_cf_sum += cf_rho
                        ep_rho_opt_sum += float(opt_rho)
                    elif ref["has_numerical_solve"]:
                        ep_solver_failures += 1

                    cf_cpu_s = float(timer["cpu_seconds"])
                    cf_bw_s = float(timer["bandwidth_seconds"])
                    row = {
                        "seed": int(seed), "step": int(step_idx), "active_users": int(len(active_ids)),
                        "offloaded_users": int(len(ref["offload_ids"])),
                        "active_mask": active_before.astype(int).tolist(),
                        "task_sizes": task_before.tolist(),
                        "uav_positions_before_action": uav_before.tolist(),
                        "uav_positions_for_allocation": uav_after.tolist(),
                        "ue_positions_for_allocation": ue_before.tolist(),
                        "channel_gain_matrix": channel_gain_matrix.tolist(),
                        "offload_projected": bool(info.get("offload_projected", False)),
                        "offload_ratios_before_energy_projection": np.asarray(
                            info.get("initial_offload_ratios", ratios), dtype=float).tolist(),
                        "offload_safety_scale_per_uav": np.asarray(
                            info.get("offload_scale", np.ones(K)), dtype=float).tolist(),
                        "association": assoc.tolist(), "safety_projected_offload_ratios": ratios.tolist(),
                        "computation_budget_j": budgets.tolist(),
                        "cpu_cf_hz": cpu_cf.tolist(), "bandwidth_cf_hz": bw_cf.tolist(),
                        "cpu_opt_hz": ref["cpu"].tolist(), "bandwidth_opt_hz": ref["bandwidth"].tolist(),
                        "delays_cf_s": env_delays.tolist(), "delays_opt_s": ref["delays"].tolist() if reference_valid else None,
                        "cf_j": j_cf, "opt_j": j_opt, "raw_relative_gap_fraction": raw_gap,
                        "relative_gap_percent": relative_gap, "slrr_cf_mean_active": cf_rho,
                        "slrr_opt_mean_active": opt_rho, "slrr_gain_mean_active": slrr_gain,
                        "sum_rho_gain": sum_rho_gain,
                        "edge_branch_dominant_fraction": branch_fraction,
                        "reference_valid": reference_valid,
                        "solver_required": bool(ref["has_numerical_solve"]),
                        "solver_success": bool(reference_valid) if ref["has_numerical_solve"] else None,
                        "solver_messages": [u.get("message") for u in ref["uav_results"]],
                        "solver_iterations": [int(u.get("iterations", 0)) for u in ref["uav_results"]],
                        "uav_solver": ref["uav_results"],
                        "cf_delay_reconstruction_max_abs_error_s": cf_reconstruction_error,
                        "constraint_residuals": constraint_residuals,
                        "cf_cpu_allocation_ms": cf_cpu_s * 1000.0,
                        "cf_bandwidth_allocation_ms": cf_bw_s * 1000.0,
                        "cf_allocation_ms": (cf_cpu_s + cf_bw_s) * 1000.0,
                        "opt_allocation_ms": opt_wall_seconds * 1000.0,
                        "opt_solver_ms": ref["solver_seconds"] * 1000.0,
                        "cf_cpu_allocator_calls": int(timer["cpu_calls"]),
                        "cf_bandwidth_allocator_calls": int(timer["bandwidth_calls"]),
                        "environment_rejected": False,
                    }
                    append_jsonl(state_path, row)
                    all_states.append(row)
                    state = next_state
                    if done:
                        break

                episode_valid = (ep_solver_failures == 0 and ep_valid_states == ep_active_states
                                 and ep_executed_steps == env.slot_num and ep_rejected_attempts == 0)
                episode_row = {
                    "seed": int(seed), "episode_index": int(ep_idx), "executed_steps": int(ep_executed_steps),
                    "active_states": int(ep_active_states), "reference_valid_states": int(ep_valid_states),
                    "episode_reference_valid": bool(episode_valid),
                    "slrr_cf_fixed_horizon": float(ep_rho_cf_sum / env.slot_num) if episode_valid else None,
                    "slrr_opt_fixed_horizon": float(ep_rho_opt_sum / env.slot_num) if episode_valid else None,
                    "slrr_fixed_horizon_gain": float((ep_rho_opt_sum - ep_rho_cf_sum) / env.slot_num) if episode_valid else None,
                    "solver_failures": int(ep_solver_failures),
                    "rejected_attempts": int(ep_rejected_attempts),
                    "survived_under_cf_rollout": bool(ep_executed_steps == env.slot_num),
                }
                append_jsonl(episode_path, episode_row)
                all_episodes.append(episode_row)
                manifest["episodes_completed"] = ep_idx + 1
                write_json(output_dir / "manifest.json", manifest)
                if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
                    print(f"[{ep_idx+1}/{args.episodes}] seed={seed} active-states={counts['active_states']} "
                          f"valid={counts['reference_valid']} solver={counts['solver_success']}/{counts['solver_required']}",
                          flush=True)

        digest_after = model_digest(agent)
        if digest_before != digest_after:
            raise RuntimeError("Frozen checkpoint digest changed during evaluation")
        manifest["model_digest_before"] = digest_before
        manifest["model_digest_after"] = digest_after
        summary = summarize_results(all_states, all_episodes, manifest)
        summary["counts"] = counts
        summary["state_seed_range"] = [int(args.seed_start), int(args.seed_start + args.episodes - 1)]
        summary["valid_gap_rule"] = "g is computed only for active-task states with J_OPT > 0 and a successful feasible reference."
        summary["slrr_difference_rule"] = "slrr_gain_mean_active = mean(active rho under OPT) - mean(active rho under CF); episode fixed-horizon gain follows repository slrr_fixed_horizon aggregation."
        write_json(output_dir / "summary.json", summary)
        manifest["status"] = summary["status"]
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest["model_digest_before"] = digest_before
        manifest["model_digest_after"] = digest_after
        manifest["summary"] = str(output_dir / "summary.json")
        write_json(output_dir / "manifest.json", manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
        write_json(output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
