"""Frozen hotspot evaluation. Imports the archived training code, never edits it.

The scenario bank contains action-independent arrivals and UE positions for the
entire horizon, including slots following policy failure. Future data never enter
the observation. Each algorithm replays exactly the same bank entry.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

# This evaluator is deliberately a CPU-only, single-thread companion to training.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/original_main_seed42_gpu_retry_20261029_234000/source_snapshot"
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
import config
from UAV_env import UAVEnv, OFFLOAD_EPS
from experiment_config import EnvironmentConfig
from evaluation import (EpisodeMetrics, policy_input,
                        configure_policy_environment, summarize, jsonable)
from state_normalization import StateNormalization
from train_unified import ALGO_REGISTRY

CASES = ("natural_id", "matched_forward", "matched_reverse", "nonperiodic",
         "fast_switch", "slow_switch", "unseen_pairs")
PAIRS = ((0, 1), (1, 2), (2, 3))
NOVEL_PAIRS = ((0, 2), (1, 3), (0, 3))
CORE_FILES = ("UAV_env.py", "config.py", "experiment_config.py", "evaluation.py",
              "state_normalization.py", "train_unified.py", "DARS.py", "sac_agent.py",
              "esac_per_agent.py", "td3_agent.py", "maddpg_agent.py")


def policy_objects(agent):
    def walk(name, value):
        if isinstance(value, (torch.nn.Module, torch.Tensor)):
            yield name, value
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                yield from walk(f"{name}[{i}]", item)
        elif isinstance(value, dict):
            for key, item in sorted(value.items(), key=lambda item: str(item[0])):
                yield from walk(f"{name}.{key}", item)
    for name, value in sorted(vars(agent).items()):
        yield from walk(name, value)


def model_digest(agent):
    h = hashlib.sha256()
    for name, obj in policy_objects(agent):
        tensors = obj.state_dict().items() if isinstance(obj, torch.nn.Module) else [("", obj)]
        for key, tensor in sorted(tensors):
            h.update((name + key).encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def frozen_policy(agent):
    modules = [obj for _, obj in policy_objects(agent) if isinstance(obj, torch.nn.Module)]
    modes = [obj.training for obj in modules]
    for module in modules:
        module.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, mode in zip(modules, modes):
            module.train(mode)


def select_action(agent, algorithm, state):
    if algorithm == "sac_cvx":
        # The archived vanilla SAC ignores noise_std=0 and always samples.
        # Explicitly apply the same deterministic evaluation convention as DARS.
        return agent.actor.deterministic(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).cpu().numpy()[0]
    return agent.select_action(state, noise_std=0.0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=jsonable,
                              allow_nan=False) + "\n", encoding="utf-8")
    # Windows readers may briefly hold a file without FILE_SHARE_DELETE. Retry
    # only the atomic log replacement; never recompute an episode or change data.
    for attempt in range(32):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 31:
                raise
            time.sleep(min(0.01 * 2**attempt, 0.5))


def base_config():
    return EnvironmentConfig.from_json(SOURCE / "experiments/train_diverse.json")


def make_schedule(case, seed, index):
    """Return H+1 phase indices and metadata, without looking at any policy."""
    cfg = base_config()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 96031]))
    if case in ("natural_id", "unseen_pairs"):
        env = UAVEnv(experiment_config=cfg, seed=seed)
        env.reset(seed=seed)
        return env.hotspot_schedule.copy(), dict(segments=env.hotspot_segments, draws=1)
    if case in ("matched_forward", "matched_reverse"):
        feasible = [d for d in itertools.product(range(12, 21), repeat=3) if sum(d) == 50]
        durations = feasible[int(rng.integers(len(feasible)))]
        first = index % 3
        direction = 1 if case == "matched_forward" else -1
        order = [(first + direction * j) % 3 for j in range(3)]
        phases, segments = [], []
        for phase in order:
            segments.append(dict(start=len(phases), phase=phase, duration=durations[phase]))
            phases.extend([phase] * durations[phase])
        # Terminal observation is never acted on; preserve the final phase there.
        return np.array(phases + [phases[-1]]), dict(segments=segments, draws=1,
                                                    phase_durations=durations)
    bounds = {"nonperiodic": (12, 20), "fast_switch": (4, 8), "slow_switch": (24, 32)}[case]
    for draws in range(1, 10001):
        phase = int(rng.integers(3))
        phases, segments = [], []
        while len(phases) <= cfg.horizon:
            dwell = int(rng.integers(bounds[0], bounds[1] + 1))
            segments.append(dict(start=len(phases), phase=phase, duration=dwell))
            phases.extend([phase] * dwell)
            phase = (phase + (int(rng.integers(1, 3)) if case == "nonperiodic" else 1)) % 3
        observed = [s["phase"] for s in segments if s["start"] < cfg.horizon]
        directions = {(b - a) % 3 for a, b in zip(observed, observed[1:])}
        if case != "nonperiodic" or directions == {1, 2}:
            return np.asarray(phases[:cfg.horizon + 1]), dict(segments=segments, draws=draws)
    raise RuntimeError("Nonperiodic schedule rejection sampler did not terminate")


class ScheduledEnv(UAVEnv):
    def __init__(self, schedule, pairs=PAIRS, **kwargs):
        self.eval_schedule = np.asarray(schedule, dtype=int)
        self.eval_pairs = tuple(tuple(pair) for pair in pairs)
        super().__init__(**kwargs)

    def _build_hotspot_schedule(self):
        self.hotspot_schedule = self.eval_schedule.copy()
        self.hotspot_segments = []

    def _generate_tasks(self):
        # reset() installs the training pairs just before calling this hook.
        self._hotspot_phases = self.eval_pairs
        super()._generate_tasks()


def make_bank_entry(case, seed, index):
    schedule, metadata = make_schedule(case, seed, index)
    pairs = NOVEL_PAIRS if case == "unseen_pairs" else PAIRS
    env = ScheduledEnv(schedule, pairs, experiment_config=base_config(), seed=seed,
                       use_cvx_alloc=True)
    env.reset(seed=seed)
    initial = env.loc_uav.copy()
    users, tasks, active = [], [], []
    for t in range(env.slot_num + 1):
        if t:
            env.current_step = t
            env._update_ue()
        users.append(env.loc_ue.copy())
        tasks.append(env.task_list.copy())
        active.append(env.active_mask.copy())
    arrays = dict(schedule=schedule, pairs=np.array(pairs), initial_uav=initial,
                  ue=np.array(users), tasks=np.array(tasks), active=np.array(active),
                  cluster_ids=env.user_cluster_ids, centers=env.cluster_centers)
    if case == "matched_reverse":
        forward, _ = make_bank_entry("matched_forward", seed, index)
        # Match actual task sets by (phase, within-phase slot), not just expected
        # loads. Arrivals are iid conditional on phase, so this preserves each
        # case's marginal law. UE positions remain indexed by global time.
        for phase in range(3):
            dst = np.flatnonzero(schedule[:50] == phase)
            src = np.flatnonzero(forward["schedule"][:50] == phase)
            for key in ("tasks", "active"):
                arrays[key][dst] = forward[key][src]
    metadata.update(case=case, seed=seed, index=index,
                    phase_slots=np.bincount(schedule[:50], minlength=3).tolist(),
                    active_user_slots=int(np.sum(arrays["active"][:50])),
                    task_bits=float(np.sum(arrays["tasks"][:50])))
    return arrays, metadata


class ReplayEnv(ScheduledEnv):
    def __init__(self, bank, **kwargs):
        self.bank = bank
        super().__init__(bank["schedule"], bank["pairs"], **kwargs)

    def _init_ue_positions(self):
        return self.bank["ue"][0].copy()

    def _generate_tasks(self):
        self._hotspot_phases = self.eval_pairs
        self.task_list = self.bank["tasks"][self.current_step].copy()
        self.active_mask = self.bank["active"][self.current_step].copy()

    def _update_ue(self):
        self.loc_ue = self.bank["ue"][self.current_step].copy()
        self._generate_tasks()


def validate_manifest(path, seeds):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or manifest["training"]["episodes"] != 15000:
        raise ValueError("Require completed original 15000-episode checkpoint")
    if EnvironmentConfig(**manifest["environment"]) != base_config():
        raise ValueError("Reject reverse-trained/changed-distribution checkpoint")
    if set(seeds) & set(manifest["training_episode_seeds"] + manifest["validation_seeds"]):
        raise ValueError("Test seeds overlap training/validation")
    for name in CORE_FILES:
        if digest(SOURCE / name) != manifest["source_sha256"][name]:
            raise ValueError(f"Archived source mismatch: {name}")
    for key, value in manifest["system_parameters"].items():
        if not np.array_equal(np.asarray(getattr(config, key)), np.asarray(value)):
            raise ValueError(f"Physical parameter mismatch: {key}")
    if manifest["env_kwargs"] != {"use_cvx_alloc": True}:
        raise ValueError("Unexpected algorithm environment options")
    return manifest


def cluster_sum(values, ids):
    return np.bincount(ids, weights=values, minlength=4)


def delay_parts(bank, t, env, info):
    active, bits = bank["active"][t], bank["tasks"][t]
    x = info["offload_ratios"]
    local = (1.0 - x) * bits * config.S_CYCLES / config.F_UE
    edge = np.zeros(config.M)
    mask = active & (info["association"] >= 0) & (x > OFFLOAD_EPS)
    for i in np.where(mask)[0]:
        k = info["association"][i]
        gain = env._channel_gain(env.loc_uav[k], bank["ue"][t, i])
        rate = info["bandwidth_allocation"][i] * np.log2(1 + max(config.P_TX * gain / config.SIGMA2, 1e-10))
        edge[i] = x[i] * bits[i] / max(rate, 1e-10) + x[i] * bits[i] * config.S_CYCLES / max(info["cpu_allocation"][i], 1e-10)
    if not np.allclose(np.maximum(local, edge)[active], info["delays"][active], rtol=1e-9, atol=1e-9):
        raise AssertionError("Trace delay reconstruction differs from environment")
    return local, edge, mask


def switch_events(bank, trace):
    """Four slots after each planned switch; failed policies contribute no service.

    Distances/delays after failure are missing, not zero. Both pre/post distances
    target the same newly active clusters. Eligibility depends only on schedule.
    """
    schedule = bank["schedule"][:50]
    changes = (np.flatnonzero(schedule[1:] != schedule[:-1]) + 1).tolist()
    by_slot = {r["slot"]: r for r in trace if r["executed"]}
    attempted_slots = {r["slot"] for r in trace}
    events = []
    for index, t in enumerate(changes):
        end = changes[index + 1] if index + 1 < len(changes) else 50
        new = sorted(set(bank["pairs"][schedule[t]]) - set(bank["pairs"][schedule[t-1]]))
        eligible = end - t >= 4
        event = dict(slot=t, new_clusters=new, planned_dwell=end-t, eligible_four_slots=eligible,
                     reached=t in attempted_slots, first_slot_executed=t in by_slot,
                     complete_four_slots=eligible and all(s in by_slot for s in range(t, t+4)))
        if eligible:
            users = np.isin(bank["cluster_ids"], new)
            offered = float(bank["tasks"][t:t+4, users].sum())
            arrivals = int(bank["active"][t:t+4, users].sum())
            observed = [by_slot[s] for s in range(t, t+4) if s in by_slot]
            event.update(offered_bits=offered, arrivals=arrivals,
                         offloaded_bits=float(sum(np.asarray(r["offload_bits_by_cluster"])[new].sum() for r in observed)),
                         rho_sum=float(sum(np.asarray(r["rho_sum_by_cluster"])[new].sum() for r in observed)),
                         executed_slots=len(observed),
                         distance_before_m=None, distance_after_m=[None]*4,
                         delay_sum_s=float(sum(np.asarray(r["delay_sum_by_cluster"])[new].sum() for r in observed)),
                         executed_arrivals=int(sum(np.asarray(r["active_by_cluster"])[new].sum() for r in observed)))
            if t in by_slot:
                centers = bank["centers"][new]
                event["distance_before_m"] = float(np.linalg.norm(centers[:, None] - by_slot[t]["uav_before"][None], axis=2).min(axis=1).mean())
                event["distance_after_m"] = [float(np.linalg.norm(centers[:, None] - by_slot[s]["uav_after"][None], axis=2).min(axis=1).mean()) if s in by_slot else None for s in range(t, t+4)]
        events.append(event)
    return events


def evaluate_episode(agent, manifest, bank, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = ReplayEnv(bank, experiment_config=base_config(), seed=seed, **manifest["env_kwargs"])
    state = env.reset(seed=seed)
    if not np.array_equal(env.loc_uav, bank["initial_uav"]):
        raise AssertionError("Initial UAV mismatch")
    configure_policy_environment(agent, env)
    normalizer = StateNormalization(env.battery_capacity)
    metrics = EpisodeMetrics(env, seed, keep_trace=False)
    trace = []
    ids = bank["cluster_ids"]
    for t in range(env.slot_num):
        if not np.array_equal(env.task_list, bank["tasks"][t]) or not np.array_equal(env.loc_ue, bank["ue"][t]):
            raise AssertionError("Exogenous scenario mismatch")
        before = env.loc_uav.copy()
        start = time.perf_counter()
        action = select_action(agent, manifest["algorithm"], policy_input(agent, normalizer, state))
        policy_seconds = time.perf_counter() - start
        start = time.perf_counter()
        state, reward, done, info = env.step(action)
        metrics.record(info, reward, policy_seconds, time.perf_counter() - start)
        entry = dict(slot=t, executed=bool(info["action_executed"]),
                     uav_before=before, uav_after=env.loc_uav.copy(),
                     battery=env.e_battery_uav.copy(), phase=int(bank["schedule"][t]),
                     movement_projected=bool(info["movement_projected"]),
                     offload_projected=bool(info["offload_projected"]))
        if info["action_executed"]:
            local, edge, mask = delay_parts(bank, t, env, info)
            active = bank["active"][t]
            bits = bank["tasks"][t]
            d = np.linalg.norm(bank["ue"][t, :, None] - env.loc_uav[None, :, :], axis=2).min(axis=1)
            entry.update(rho_mean=float(info["rho_mean"]), rho_sum=float(np.sum(info["rho"])),
                         offload_bits_by_cluster=cluster_sum(bits * info["offload_ratios"], ids),
                         rho_sum_by_cluster=cluster_sum(info["rho"], ids),
                         delay_sum_by_cluster=cluster_sum(info["delays"], ids),
                         active_by_cluster=cluster_sum(active.astype(int), ids),
                         covered_bits_by_cluster=cluster_sum(bits * (d <= config.R_COV), ids),
                         offload_ratio_active_mean=float(np.mean(info["offload_ratios"][active])) if active.any() else None,
                         local_delay_sum_s=float(local[active].sum()), active_count=int(active.sum()),
                         edge_delay_sum_s=float(edge[mask].sum()), offloading_count=int(mask.sum()),
                         energy_comp_j=float(np.sum(info["energy_comp"])))
        trace.append(entry)
        if done:
            break
    row = metrics.finish()
    executed = [x for x in trace if x["executed"]]
    active_n = sum(x["active_count"] for x in executed)
    offload_n = sum(x["offloading_count"] for x in executed)
    rho_sum = sum(x["rho_sum"] for x in executed)
    row.update(slrr_sum_per_horizon=rho_sum / env.slot_num,
               slrr_per_arriving_task=rho_sum / max(int(bank["active"][:50].sum()), 1),
               offloaded_bits_fraction=float(sum(np.sum(x["offload_bits_by_cluster"]) for x in executed) / bank["tasks"][:50].sum()),
               local_delay_active_mean_s=sum(x["local_delay_sum_s"] for x in executed) / active_n if active_n else None,
               edge_delay_offloading_mean_s=sum(x["edge_delay_sum_s"] for x in executed) / offload_n if offload_n else None,
               active_user_slots_planned=int(bank["active"][:50].sum()),
               active_user_slots_executed=active_n, offloading_user_slots=offload_n,
               offered_task_bits=float(bank["tasks"][:50].sum()), trace=trace,
               switch_events=switch_events(bank, trace))
    return row


def paired_stats(values, seed=906):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(x[rng.integers(len(x), size=(2000, len(x)))], axis=1)
    return dict(n=len(x), mean=float(np.mean(x)),
                bootstrap_episode_ci95=np.quantile(means, [0.025, 0.975]).tolist(),
                scope="conditional on this trained checkpoint; not training-seed uncertainty")


def save_summary(output, case_rows, manifest, checkpoint_hash, bank_hash):
    summaries = {}
    for case, rows in case_rows.items():
        if not rows:
            continue
        stats = summarize(rows)
        stats["slrr_fixed_horizon"].update(paired_stats([r["slrr_fixed_horizon"] for r in rows]))
        for key in ("slrr_sum_per_horizon", "slrr_per_arriving_task", "offloaded_bits_fraction",
                    "local_delay_active_mean_s", "edge_delay_offloading_mean_s", "energy_comp_j"):
            vals = [r[key] for r in rows if r[key] is not None]
            stats[key] = dict(mean=float(np.mean(vals)) if vals else None, count=len(vals))
        events = [e for r in rows for e in r["switch_events"]]
        eligible = [e for e in events if e["eligible_four_slots"]]
        offered = sum(e["offered_bits"] for e in eligible)
        arrivals = sum(e["arrivals"] for e in eligible)
        stats["switch_response"] = dict(planned_events=len(events), eligible_four_slot_events=len(eligible),
            reached_events=sum(e["reached"] for e in eligible), first_slot_executed_events=sum(e["first_slot_executed"] for e in eligible),
            complete_four_slot_events=sum(e["complete_four_slots"] for e in eligible),
            newly_hot_offloaded_fraction=sum(e["offloaded_bits"] for e in eligible)/offered if offered else None,
            newly_hot_slrr_per_arrival=sum(e["rho_sum"] for e in eligible)/arrivals if arrivals else None,
            note="All eligible planned events included; postfailure service score is zero, distance/delay missing.")
        summaries[case] = stats
    contrasts = {}
    for case in CASES:
        ref = "matched_forward" if case == "matched_reverse" else "natural_id"
        if case == ref or case not in case_rows or ref not in case_rows:
            continue
        a = {r["seed"]: r for r in case_rows[case]}
        b = {r["seed"]: r for r in case_rows[ref]}
        common = sorted(a.keys() & b.keys())
        if common:
            contrasts[case + "_minus_" + ref] = {
                "slrr": paired_stats([a[s]["slrr_fixed_horizon"] - b[s]["slrr_fixed_horizon"] for s in common]),
                "slrr_sum_per_horizon": paired_stats([a[s]["slrr_sum_per_horizon"] - b[s]["slrr_sum_per_horizon"] for s in common]),
                "slrr_per_arriving_task": paired_stats([a[s]["slrr_per_arriving_task"] - b[s]["slrr_per_arriving_task"] for s in common]),
                "survival_probability": paired_stats([float(a[s]["survived"]) - float(b[s]["survived"]) for s in common])}
    dump(output / "summary.json", dict(algorithm=manifest["algorithm"], training_seed=manifest["training_seed"],
         checkpoint_sha256=checkpoint_hash, bank_sha256=bank_hash, cases=summaries, paired_contrasts=contrasts,
         interpretation="Positive contrast favors the shifted case. Episode intervals condition on one trained model."))


def run_worker(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() or list(output.glob("*.episodes.jsonl")):
        raise FileExistsError("Refusing to overwrite an existing evaluation")
    bank_dir = Path(args.bank)
    bank_index = json.loads((bank_dir / "index.json").read_text(encoding="utf-8"))
    seeds = bank_index["seeds"]
    if EnvironmentConfig(**bank_index["config"]) != base_config() or bank_index["cases"] != list(CASES):
        raise ValueError("Scenario-bank protocol mismatch")
    if bank_index["evaluator_sha256"] != digest(__file__):
        raise ValueError("Scenario bank was built by a different evaluator version")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate test seeds")
    manifest = validate_manifest(args.manifest, seeds)
    checkpoint = Path(args.manifest).with_name(manifest["algorithm"] + ".pth")
    checkpoint_hash, bank_hash = digest(checkpoint), digest(bank_dir / "index.json")
    agent, _, _ = ALGO_REGISTRY[manifest["algorithm"]]()
    agent.load(str(checkpoint))
    if torch.cuda.is_available():
        raise RuntimeError("Evaluation must not contend with GPU training")
    before = model_digest(agent)
    # Fail loudly if a future edit accidentally introduces learning/replay writes.
    def forbidden(*a, **k):
        raise RuntimeError("Training is forbidden during frozen-policy evaluation")
    agent.learn = forbidden
    agent.store = forbidden
    dump(output / "provenance.json", dict(manifest=json.loads(Path(args.manifest).read_text(encoding="utf-8")),
         manifest_path=str(Path(args.manifest).resolve()), checkpoint_sha256=checkpoint_hash,
         manifest_sha256=digest(args.manifest), bank_sha256=bank_hash,
         evaluator_sha256=digest(__file__), archived_source=str(SOURCE), device="cpu", threads=1,
         model_digest_before=before, test_seeds=seeds, inference_mode="deterministic for all algorithms"))
    rows_by_case = {}
    started = time.time()
    with frozen_policy(agent):
        for case in bank_index["cases"]:
            rows = rows_by_case[case] = []
            with (output / (case + ".episodes.jsonl")).open("x", encoding="utf-8") as stream:
                for i, seed in enumerate(seeds):
                    item = bank_index["entries"][case][i]
                    if item["seed"] != seed or item["case"] != case:
                        raise ValueError("Bank entry seed/case mismatch")
                    path = bank_dir / item["file"]
                    if digest(path) != item["sha256"]:
                        raise ValueError("Scenario-bank hash mismatch")
                    with np.load(path) as source:
                        bank = {k: source[k] for k in source.files}
                    if bank["schedule"].shape != (51,) or bank["tasks"].shape != (51, config.M) or bank["ue"].shape != (51, config.M, 2):
                        raise ValueError("Invalid scenario dimensions")
                    row = evaluate_episode(agent, manifest, bank, seed)
                    row.update(case=case, bank_entry_sha256=item["sha256"])
                    stream.write(json.dumps(row, default=jsonable, allow_nan=False, separators=(",", ":")) + "\n")
                    stream.flush()
                    # Summary retains episode aggregates, not the large traces.
                    rows.append({k: v for k, v in row.items() if k != "trace"})
                    dump(output / "progress.json", dict(status="running", pid=os.getpid(), case=case,
                         case_completed=i+1, case_total=len(seeds), completed_episodes=sum(map(len, rows_by_case.values())),
                         total_episodes=len(seeds)*len(bank_index["cases"]), elapsed_seconds=time.time()-started,
                         updated_unix=time.time()))
            save_summary(output, rows_by_case, manifest, checkpoint_hash, bank_hash)
            print(f"{manifest['algorithm']} seed={manifest['training_seed']} {case}: {len(rows)} episodes done", flush=True)
    after = model_digest(agent)
    if before != after or checkpoint_hash != digest(checkpoint):
        raise AssertionError("Frozen model/checkpoint changed")
    dump(output / "completion.json", dict(status="completed", episodes=sum(map(len, rows_by_case.values())),
         model_digest_before=before, model_digest_after=after, elapsed_seconds=time.time()-started,
         checkpoint_sha256=checkpoint_hash))
    dump(output / "progress.json", dict(status="completed", pid=os.getpid(), updated_unix=time.time(),
         completed_episodes=sum(map(len, rows_by_case.values())), total_episodes=len(seeds)*len(bank_index["cases"])))


def build_bank(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Scenario bank must be new")
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed, args.seed + args.episodes))
    index = dict(schema=2, seeds=seeds, cases=list(CASES), entries={}, config=base_config().to_dict(),
                 evaluator_sha256=digest(__file__), created_unix=time.time())
    for case in CASES:
        entries = index["entries"][case] = []
        for i, seed in enumerate(seeds):
            arrays, metadata = make_bank_entry(case, seed, i)
            path = output / f"{case}_{seed}.npz"
            np.savez_compressed(path, **arrays)
            entries.append(dict(file=path.name, sha256=digest(path), **metadata))
        print(f"Bank: {case}, {len(seeds)} episodes", flush=True)
    dump(output / "index.json", index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bank", "worker"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=700000)
    parser.add_argument("--episodes", type=int, default=201)
    parser.add_argument("--manifest")
    parser.add_argument("--bank")
    args = parser.parse_args()
    if args.mode == "bank":
        build_bank(args)
    else:
        run_worker(args)
