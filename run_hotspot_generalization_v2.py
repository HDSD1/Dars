"""Bounded, sequential CPU evaluation queue alongside the existing GPU training.

Completed original models are evaluated once; unfinished models remain pending.
Worker exits, heartbeats, and a 24h worker deadline are monitored. Failed jobs
are recorded and never silently retried. This queue does not launch training.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
EVALUATOR = ROOT / "hotspot_generalization_v2.py"
ALGORITHMS = ("per_sac_cvx", "esac_per_cvx", "sac_cvx", "td3_cvx", "maddpg_cvx")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(32):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 31:
                raise
            time.sleep(min(0.01 * 2**attempt, 0.5))


def registry():
    entries = []
    for seed in (42, 43, 44):
        for algo in ALGORITHMS:
            if seed == 42:
                folder = ("results/original_main_seed42_gpu_retry_20261029_234000" if algo in ALGORITHMS[:3]
                          else "results/remaining_td3_maddpg_seed42_20261030/source_snapshot/results/remaining_td3_maddpg_seed42_20261030")
            else:
                folder = ("results/formal_all_five_seeds43_44_restart_20261031/source_snapshot/results/"
                          f"formal_all_five_seeds43_44_session2_20261031_seed{seed}")
            entries.append(dict(id=f"{algo}_seed{seed}", algorithm=algo, training_seed=seed,
                                manifest=str(ROOT / folder / (algo + ".manifest.json"))))
    return entries


def environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")


def flags():
    return (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0


def prepare(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new run folder")
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(version=2, locked_utc=datetime.now(timezone.utc).isoformat(),
        target="Zero-shot generalization of original trained policies to held-out hotspot dynamics/combinations",
        training_episodes=15000, training_seeds=[42, 43, 44], algorithms=list(ALGORITHMS),
        test_seeds=list(range(700000, 700201)), smoke_seeds=[699990, 699991],
        cases=["natural_id", "matched_forward", "matched_reverse", "nonperiodic", "fast_switch", "slow_switch", "unseen_pairs"],
        models=registry(), evaluator_sha256=sha(EVALUATOR), runner_sha256=sha(__file__),
        archived_source="results/original_main_seed42_gpu_retry_20261029_234000/source_snapshot",
        policy="Final checkpoint; deterministic; no learning, checkpoint selection, or fine tuning on these tests",
        primary=["slrr_sum_per_horizon", "survival_probability"],
        additional_slrr=["slrr_fixed_horizon", "slrr_per_arriving_task"],
        order_control="matched_reverse minus matched_forward; phase durations in12..20 sum50; same task table per phase",
        nonperiodic="12..20 dwell; next among other two; condition on both directions observed in slots0..49",
        unseen_pairs=[[0,2],[1,3],[0,3]], train_pairs=[[0,1],[1,2],[2,3]],
        timing_bounds=dict(fast=[4,8], slow=[24,32]), switch_window=4,
        trajectory_example=dict(algorithm="per_sac_cvx", training_seed=42, test_seed=700000, case="matched_reverse"),
        training_seed_inference="Show each seed and common completed seed set; episode CIs conditional on fixed checkpoints",
        prior_exposure="Original seed42 order/timing failures were already examined. This is a prospectively locked follow-up, not historical preregistration.",
        execution=dict(device="cpu", threads=1, workers=1, process_priority="below_normal", worker_timeout_hours=24, queue_deadline_days=7),
        seed_audit="102 existing JSON files checked by integer seed fields before creation; no new seed overlap found",
        no_automatic_retry=True)
    write(output / "protocol.json", protocol)
    (output / "source_tools").mkdir()
    for path in (EVALUATOR, Path(__file__), ROOT / "test_hotspot_generalization_v2.py"):
        (output / "source_tools" / path.name).write_bytes(path.read_bytes())
    subprocess.run([sys.executable, "-u", str(EVALUATOR), "bank", "--seed", "700000", "--episodes", "201",
                    "--output", str(output / "bank")], cwd=ROOT, env=environment(), check=True,
                    creationflags=flags(), timeout=1800)
    write(output / "protocol_lock.json", dict(protocol_sha256=sha(output / "protocol.json"),
                                              bank_index_sha256=sha(output / "bank/index.json")))


def run(output, smoke=False):
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    lock = json.loads((output / "protocol_lock.json").read_text(encoding="utf-8"))
    if sha(output / "protocol.json") != lock["protocol_sha256"] or sha(EVALUATOR) != protocol["evaluator_sha256"]:
        raise RuntimeError("Locked protocol/evaluator changed")
    if sha(output / "bank/index.json") != lock["bank_index_sha256"]:
        raise RuntimeError("Locked bank index changed")
    runroot = output / "smoke" if smoke else output
    runroot.mkdir(exist_ok=True)
    if (runroot / "batch_status.json").exists():
        raise FileExistsError("Queue already started; inspect its status before any restart")
    bank = runroot / "bank"
    if smoke:
        subprocess.run([sys.executable, "-u", str(EVALUATOR), "bank", "--seed", "699990", "--episodes", "2",
                        "--output", str(bank)], cwd=ROOT, env=environment(), check=True,
                        creationflags=flags(), timeout=600)
    else:
        smoke_status = json.loads((output / "smoke/batch_status.json").read_text(encoding="utf-8"))
        if smoke_status["status"] != "completed":
            raise RuntimeError("All five algorithm smoke checks must complete first")
    jobs = [dict(item, status="pending") for item in protocol["models"] if not smoke or item["training_seed"] == 42]
    state = dict(status="running", pid=os.getpid(), created_unix=time.time(), mode="smoke" if smoke else "formal",
                 jobs=jobs, current_job=None, device="cpu", threads=1, protocol_sha256=lock["protocol_sha256"])
    deadline = time.time() + (7200 if smoke else 7*24*3600)
    status_path = runroot / "batch_status.json"
    def save():
        state["updated_unix"] = time.time()
        write(status_path, state)
    save()
    while time.time() < deadline:
        available = None
        for job in jobs:
            if job["status"] != "pending":
                continue
            path = Path(job["manifest"])
            if not path.exists():
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue  # Trainer may be atomically finishing/writing its manifest.
            if manifest.get("status") == "completed" and path.with_name(job["algorithm"] + ".pth").exists():
                available = job
                break
        if available is None:
            pending = [j for j in jobs if j["status"] == "pending"]
            if not pending:
                state["status"] = "completed" if all(j["status"] == "completed" for j in jobs) else "completed_with_failures"
                state["current_job"] = None
                save()
                return
            state.update(status="waiting_for_training", current_job=None)
            save()
            time.sleep(60)
            continue
        job = available
        if sha(EVALUATOR) != protocol["evaluator_sha256"]:
            raise RuntimeError("Evaluator changed during locked run")
        destination = runroot / job["id"]
        logpath = runroot / (job["id"] + ".console.log")
        cmd = [sys.executable, "-u", str(EVALUATOR), "worker", "--manifest", job["manifest"],
               "--bank", str(bank), "--output", str(destination)]
        job.update(status="running", started_unix=time.time(), log=str(logpath))
        state.update(status="running", current_job=job["id"])
        with logpath.open("xb") as stream:
            process = subprocess.Popen(cmd, cwd=ROOT, env=environment(), stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=flags())
            job["pid"] = process.pid
            save()
            timeout = 1200 if smoke else 24*3600
            while process.poll() is None:
                job["elapsed_seconds"] = time.time() - job["started_unix"]
                progress = destination / "progress.json"
                if progress.exists():
                    current = json.loads(progress.read_text(encoding="utf-8"))
                    job["progress"] = current
                    job["heartbeat_age_seconds"] = time.time() - current["updated_unix"]
                    job["stalled_advisory"] = job["heartbeat_age_seconds"] > 1800
                if job["elapsed_seconds"] > timeout:
                    job["status"] = "timeout"
                    process.kill()  # Only this evaluator child, never training.
                    break
                save()
                time.sleep(5)
            job["exit_code"] = process.wait()
        completion = destination / "completion.json"
        if job["status"] != "timeout":
            job["status"] = "completed" if job["exit_code"] == 0 and completion.exists() else "failed"
        job["ended_unix"] = time.time()
        save()
        print(f"{job['id']}: {job['status']}", flush=True)
        reporter = ROOT / "report_hotspot_generalization_v2.py"
        if not smoke and reporter.exists() and job["status"] == "completed":
            report = subprocess.run([sys.executable, str(reporter), "--output", str(output)], cwd=ROOT,
                                    env=environment(), creationflags=flags(), capture_output=True, timeout=300)
            (output / "report.console.log").write_bytes(report.stdout + report.stderr)
            state["last_report_exit_code"] = report.returncode
            save()
    state.update(status="deadline_reached", current_job=None)
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "smoke", "run"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            prepare(Path(args.output))
        else:
            run(Path(args.output), smoke=args.mode == "smoke")
    except BaseException:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / (args.mode + "_error.log")).write_text(traceback.format_exc(), encoding="utf-8")
        status_path = Path(args.output) / ("smoke/batch_status.json" if args.mode == "smoke" else "batch_status.json")
        if args.mode != "prepare" and status_path.exists():
            state = json.loads(status_path.read_text(encoding="utf-8"))
            state.update(status="failed", error=traceback.format_exc(), updated_unix=time.time())
            write(status_path, state)
        raise
