"""Report completed frozen-hotspot evaluations without importing training code.

Usage: python report_hotspot_generalization_v2.py --output ABSOLUTE_RUN_ROOT
The run root contains bank/index.json and <algorithm>_seedN worker directories.
Only this script's report artifacts are written; evaluation inputs stay intact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np


ALGORITHMS = ("per_sac_cvx", "esac_per_cvx", "sac_cvx", "td3_cvx", "maddpg_cvx")
LABELS = dict(zip(ALGORITHMS, ("DARS", "ESAC", "SAC", "TD3", "MADDPG")))
CASES = ("natural_id", "matched_forward", "matched_reverse", "nonperiodic",
         "fast_switch", "slow_switch", "unseen_pairs")
CASE_LABELS = dict(zip(CASES, ("Natural ID", "Matched\nforward", "Matched\nreverse",
                              "Nonperiodic", "Fast\nswitch", "Slow\nswitch", "Unseen\npairs")))
METRICS = ("slrr_sum_per_horizon", "slrr_fixed_horizon", "survived")
SCALAR_METRICS = METRICS + ("slrr_per_arriving_task", "offloaded_bits_fraction",
    "local_delay_active_mean_s", "edge_delay_offloading_mean_s", "energy_comp_j",
    "energy_out_j", "executed_steps", "decision_mean_ms", "offered_task_bits", "active_user_slots_planned")
FIGURE_SEED = 700000
FIGURE_TRAINING_SEED = 42
FIGURE_CASE = "matched_reverse"
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    for attempt in range(32):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 31:
                raise
            time.sleep(min(0.01 * 2**attempt, 0.5))


def finite_values(values):
    return np.asarray([float(x) for x in values if x is not None and np.isfinite(float(x))])


def stats(values, conditional_ci=False):
    x = finite_values(values)
    result = dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                  sd=float(x.std(ddof=1)) if len(x) > 1 else None)
    if conditional_ci and len(x):
        # Resample episodes within one fixed model, never as training replicates.
        rng = np.random.default_rng(960311)
        means = np.mean(x[rng.integers(len(x), size=(2000, len(x)))], axis=1)
        result["conditional_episode_bootstrap_ci95"] = np.quantile(means, [.025, .975]).tolist()
    return result


def response_stats(rows):
    events = [e for row in rows for e in row.get("switch_events", [])
              if e["eligible_four_slots"]]
    points = [[e.get("distance_before_m")] + e.get("distance_after_m", [None] * 4)
              for e in events]
    offered = sum(e["offered_bits"] for e in events)
    arrivals = sum(e["arrivals"] for e in events)
    return dict(eligible_events=len(events), reached_events=sum(e["reached"] for e in events),
                first_slot_executed_events=sum(e.get("first_slot_executed", e["distance_before_m"] is not None) for e in events),
                complete_four_slots=sum(e["complete_four_slots"] for e in events),
                offered_bits=offered, arrivals=arrivals,
                offloaded_bits=sum(e["offloaded_bits"] for e in events),
                rho_sum=sum(e["rho_sum"] for e in events),
                newly_hot_offloaded_fraction=(sum(e["offloaded_bits"] for e in events) / offered
                                             if offered else None),
                newly_hot_slrr_per_arrival=(sum(e["rho_sum"] for e in events) / arrivals
                                           if arrivals else None),
                distance=[stats([p[k] for p in points]) for k in range(5)],
                note="Service denominators include all eligible planned events; service after failure is zero. "
                     "Distance is conditional on execution and is missing after failure.")


def model_summary(rows_by_case):
    result = {}
    for case, rows in rows_by_case.items():
        result[case] = dict(episodes=len(rows), metrics={
            key: stats([r.get(key) for r in rows], conditional_ci=key in METRICS)
            for key in SCALAR_METRICS}, switch_response=response_stats(rows))
    contrasts = {}
    for case, rows in rows_by_case.items():
        reference = "matched_forward" if case == "matched_reverse" else "natural_id"
        if case == reference or reference not in rows_by_case:
            continue
        by_seed = {r["seed"]: r for r in rows_by_case[reference]}
        contrasts[case + "_minus_" + reference] = {
            key: stats([float(r[key]) - float(by_seed[r["seed"]][key]) for r in rows],
                       conditional_ci=True) for key in METRICS}
    return result, contrasts


def load_model(directory, bank, bank_hash):
    complete = read_json(directory / "completion.json")
    if complete.get("status") != "completed":
        raise ValueError("completion status is not completed")
    if not complete.get("model_digest_before") or complete["model_digest_before"] != complete.get("model_digest_after"):
        raise ValueError("frozen-policy digest check missing or failed")
    summary = read_json(directory / "summary.json")
    provenance = read_json(directory / "provenance.json")
    algorithm, seed = summary["algorithm"], int(summary["training_seed"])
    if algorithm not in ALGORITHMS or directory.name != f"{algorithm}_seed{seed}":
        raise ValueError("worker directory and summary identity differ")
    if summary["bank_sha256"] != bank_hash or provenance["bank_sha256"] != bank_hash:
        raise ValueError("worker used a different scenario bank")
    if summary["checkpoint_sha256"] != complete["checkpoint_sha256"] or summary["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
        raise ValueError("checkpoint hashes differ between worker records")
    if provenance["model_digest_before"] != complete["model_digest_before"]:
        raise ValueError("initial model digest differs between worker records")
    manifest = provenance["manifest"]
    if provenance["evaluator_sha256"] != bank["evaluator_sha256"] or manifest["environment"] != bank["config"]:
        raise ValueError("evaluation code or original training environment differs from the locked bank")
    if manifest["algorithm"] != algorithm or int(manifest["training_seed"]) != seed:
        raise ValueError("provenance identity differs from summary")
    if list(provenance["test_seeds"]) != list(bank["seeds"]):
        raise ValueError("provenance test seeds differ from scenario bank")
    if manifest.get("status") != "completed" or manifest["training"]["episodes"] != 15000:
        raise ValueError("checkpoint is not a completed original 15000-episode model")
    if set(bank["seeds"]) & set(manifest["training_episode_seeds"] + manifest["validation_seeds"]):
        raise ValueError("test seeds overlap training or validation")
    rows_by_case, chosen = {}, None
    for case in bank["cases"]:
        expected = {entry["seed"]: entry["sha256"] for entry in bank["entries"][case]}
        rows = []
        path = directory / f"{case}.episodes.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["case"] != case or row["seed"] not in expected or row["bank_entry_sha256"] != expected[row["seed"]]:
                    raise ValueError(f"scenario/seed/hash mismatch in {path.name}")
                for key in METRICS:
                    if row[key] is None or not np.isfinite(float(row[key])):
                        raise ValueError(f"non-finite core metric {key} in {path.name}")
                if algorithm == "per_sac_cvx" and seed == FIGURE_TRAINING_SEED and case == FIGURE_CASE and row["seed"] == FIGURE_SEED:
                    chosen = row
                rows.append({k: v for k, v in row.items() if k != "trace"})
        seeds = [r["seed"] for r in rows]
        if len(set(seeds)) != len(seeds) or set(seeds) != set(bank["seeds"]):
            raise ValueError(f"missing or duplicate episodes in {path.name}")
        existing = summary["cases"][case]
        if existing["episodes"] != len(rows):
            raise ValueError(f"summary episode count mismatch: {case}")
        for metric in METRICS:
            stored = existing["survival_probability"] if metric == "survived" else existing[metric]["mean"]
            if not np.isclose(stored, np.mean([r[metric] for r in rows]), rtol=1e-8, atol=1e-10):
                raise ValueError(f"summary metric mismatch: {case}/{metric}")
        rows_by_case[case] = rows
    if complete["episodes"] != sum(map(len, rows_by_case.values())):
        raise ValueError("completion episode count mismatch")
    case_stats, contrasts = model_summary(rows_by_case)
    public = dict(algorithm=algorithm, label=LABELS[algorithm], training_seed=seed,
                  path=str(directory), checkpoint_sha256=summary["checkpoint_sha256"],
                  source_sha256=manifest["source_sha256"], training_episodes=15000,
                  completion=complete, cases=case_stats, paired_contrasts=contrasts,
                  uncertainty_scope="Episode intervals condition on this trained checkpoint; not training-seed uncertainty.")
    return dict(public=public, rows=rows_by_case, chosen=chosen, manifest=manifest)


def compare_models(models, cases):
    by_algorithm = {a: {m["public"]["training_seed"]: m for m in models
                        if m["public"]["algorithm"] == a} for a in ALGORITHMS}
    common = sorted(set.intersection(*(set(v) for v in by_algorithm.values())))
    comparison = {}
    for case in cases:
        comparison[case] = {}
        for algorithm in ALGORITHMS:
            comparison[case][algorithm] = {
                key: stats([by_algorithm[algorithm][s]["public"]["cases"][case]["metrics"][key]["mean"]
                            for s in common]) for key in METRICS}
    paired = {}
    if common:
        for case in cases:
            paired[case] = {}
            for algorithm in ALGORITHMS[1:]:
                paired[case][algorithm] = {}
                for key in METRICS:
                    differences = [by_algorithm["per_sac_cvx"][s]["public"]["cases"][case]["metrics"][key]["mean"] -
                                   by_algorithm[algorithm][s]["public"]["cases"][case]["metrics"][key]["mean"]
                                   for s in common]
                    paired[case][algorithm][key] = dict(stats(differences),
                        by_training_seed={str(s): float(v) for s, v in zip(common, differences)})
    return common, comparison, paired


def placeholder(plt, path, title, message):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    ax.text(.5, .66, title, ha="center", fontsize=17, weight="bold", transform=ax.transAxes)
    ax.text(.5, .38, message, ha="center", va="center", fontsize=12, transform=ax.transAxes)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def metric_figure(plt, path, models, cases, common, comparison, dataset_scope):
    if not common:
        placeholder(plt, path, "Balanced five-algorithm comparison pending",
                    "No training seed has completed all five algorithms.\n"
                    "Completed checkpoints are listed individually in report.md; partial workers are excluded.")
        return "pending: no common completed training seed across all five algorithms"
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    x = np.arange(len(cases))
    titles = ("Core objective: total user SLRR per planned slot",
              "Mean active-user SLRR per planned slot", "Full-horizon survival")
    ylabels = (r"$H^{-1}\sum_t\sum_i\rho_i$", r"$H^{-1}\sum_t\mathrm{mean}_{i\in A_t}\rho_i$", "Survival probability")
    for metric, ax, title, ylabel in zip(METRICS, axes, titles, ylabels):
        for ai, algorithm in enumerate(ALGORITHMS):
            positions = x + (ai - 2) * .13
            values = [comparison[c][algorithm][metric]["mean"] for c in cases]
            deviations = [comparison[c][algorithm][metric]["sd"] or 0 for c in cases]
            ax.errorbar(positions, values, yerr=deviations if len(common) > 1 else None,
                        fmt="o", markersize=6, capsize=3, color=COLORS[ai], label=LABELS[algorithm])
            if len(common) > 1:
                for model in models:
                    data = model["public"]
                    if data["algorithm"] == algorithm and data["training_seed"] in common:
                        ax.scatter(positions, [data["cases"][c]["metrics"][metric]["mean"] for c in cases],
                                   marker="_", s=45, color=COLORS[ai], alpha=.55)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_ylim(-.03, 1.04)
    axes[-1].set_xticks(x, [CASE_LABELS[c] for c in cases])
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.34), frameon=False)
    seeds = ", ".join(map(str, common))
    prefix = "SMOKE CHECK ONLY | " if dataset_scope == "smoke" else ""
    fig.suptitle(prefix + f"Frozen-policy hotspot evaluation | common training seeds: {seeds}", y=.995, fontsize=14)
    note = ("Each point is an equal-weight mean over trained models; error bars are training-seed SD, not confidence intervals."
            if len(common) > 1 else "One trained checkpoint per algorithm: test episodes do not establish training-seed robustness.")
    fig.text(.5, .01, note + "\nAll planned slots remain in the denominator; failed remainder contributes zero SLRR.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .965), h_pad=2)
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return "complete"


def coordinate(x):
    return f"({x[0]:.1f}, {x[1]:.1f})"


def trajectory_figure(plt, path, models, bank, bank_dir):
    model = next((m for m in models if m["public"]["algorithm"] == "per_sac_cvx" and
                  m["public"]["training_seed"] == FIGURE_TRAINING_SEED), None)
    row = model["chosen"] if model else None
    if row is None:
        placeholder(plt, path, "Prespecified three-stage trajectory pending",
                    f"DARS training seed {FIGURE_TRAINING_SEED}, test seed {FIGURE_SEED}, {FIGURE_CASE}.\n"
                    "This exact completed test episode is unavailable. No better-looking episode was substituted.")
        return dict(status="pending", training_seed=FIGURE_TRAINING_SEED, test_seed=FIGURE_SEED, case=FIGURE_CASE)
    entry = next(e for e in bank["entries"][FIGURE_CASE] if e["seed"] == FIGURE_SEED)
    source = (bank_dir / entry["file"]).resolve()
    if sha256(source) != entry["sha256"]:
        raise ValueError("Prespecified trajectory bank file hash mismatch")
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    horizon = int(row["horizon"])
    schedule = arrays["schedule"][:horizon]
    boundaries = [0] + (np.flatnonzero(schedule[1:] != schedule[:-1]) + 1).tolist() + [horizon]
    if len(boundaries) != 4:
        raise ValueError("Prespecified matched_reverse entry does not have exactly three stages")
    records = [r for r in row["trace"] if r["executed"]]
    if [r["slot"] for r in records] != list(range(len(records))):
        raise ValueError("Trajectory contains a gap or execution after failure")
    params = model["manifest"]["system_parameters"]
    wct_indices = model["manifest"]["environment"]["wct_indices"]
    wct = np.asarray(params["WCT_POSITIONS"])[wct_indices]
    initial = arrays["initial_uav"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5))
    stage_details = []
    from matplotlib.lines import Line2D
    for stage, (ax, start, stop) in enumerate(zip(axes, boundaries[:-1], boundaries[1:]), 1):
        phase = int(schedule[start])
        hot = list(map(int, arrays["pairs"][phase]))
        ue = arrays["ue"][start]
        active = arrays["active"][start].astype(bool)
        ax.scatter(ue[~active, 0], ue[~active, 1], s=11, color="#B8B8B8", alpha=.65, zorder=2)
        ax.scatter(ue[active, 0], ue[active, 1], s=22, color="#A86D00", alpha=.75, zorder=2)
        centers = arrays["centers"]
        for ci, center in enumerate(centers):
            ax.scatter(*center, marker="*" if ci in hot else "+", s=260 if ci in hot else 75,
                       color="#F0B000" if ci in hot else "#999999", edgecolors="black" if ci in hot else None,
                       linewidths=.7, zorder=4)
            ax.annotate(f"C{ci}" + (" HOT" if ci in hot else ""), center, xytext=(8, 9),
                        textcoords="offset points", fontsize=8, weight="bold" if ci in hot else "normal")
        ax.scatter(wct[:, 0], wct[:, 1], s=190, marker="s", facecolors="none", edgecolors="#B10026",
                   linewidths=1.6, zorder=5)
        for wi, p in zip(wct_indices, wct):
            ax.annotate(f"W{wi}", p, xytext=(8, -14), textcoords="offset points", fontsize=8, color="#B10026")
        segment = [r for r in records if start <= r["slot"] < stop]
        coordinates = []
        for k in range(len(initial)):
            ax.scatter(*initial[k], marker="+", s=95, color=COLORS[k], zorder=6)
            if segment:
                points = np.asarray([segment[0]["uav_before"][k]] + [r["uav_after"][k] for r in segment])
                ax.plot(points[:, 0], points[:, 1], color=COLORS[k], linewidth=2, zorder=6)
                ax.scatter(*points[0], marker="o", s=55, facecolors="white", edgecolors=COLORS[k], zorder=7)
                ax.scatter(*points[-1], marker="s", s=55, color=COLORS[k], zorder=7)
                if len(points) > 2:
                    middle = min(len(points) - 2, len(points) // 2)
                    ax.annotate("", xy=points[middle + 1], xytext=points[middle],
                                arrowprops=dict(arrowstyle="->", color=COLORS[k], lw=1.8), zorder=8)
                label = f"U{k+1}: initial {coordinate(initial[k])}\n  start {coordinate(points[0])}; end {coordinate(points[-1])}"
                coordinates.append(dict(uav=k, initial=initial[k].tolist(), start=points[0].tolist(), end=points[-1].tolist()))
            else:
                label = f"U{k+1}: initial {coordinate(initial[k])}\n  stage not reached; no trajectory"
                coordinates.append(dict(uav=k, initial=initial[k].tolist(), start=None, end=None))
            ax.text(.01, -.16 - .12*k, label, color=COLORS[k], transform=ax.transAxes, fontsize=8.6, va="top")
        executed_stop = segment[-1]["slot"] + 1 if segment else None
        status = "complete" if executed_stop == stop else (f"stopped at t={executed_stop}" if segment else "NOT REACHED")
        ax.set_title(f"Stage {stage}: slots {start+1}-{stop}\nHot clusters {hot} | {status}", fontsize=11)
        ax.set(xlim=(0, params["GROUND_LENGTH"]), ylim=(0, params["GROUND_WIDTH"]), xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=.12)
        stage_details.append(dict(stage=stage, slot_start=start, slot_stop=stop, hot_clusters=hot,
                                  executed_stop=executed_stop, status=status, coordinates=coordinates))
    legend = [Line2D([], [], marker="*", color="none", markerfacecolor="#F0B000", markeredgecolor="black", markersize=12, label="Hot cluster center"),
              Line2D([], [], marker="s", color="none", markeredgecolor="#B10026", markersize=10, label="WCT station"),
              Line2D([], [], marker=".", color="#A86D00", linestyle="none", label="UE with a task at stage start"),
              Line2D([], [], marker="+", color="black", linestyle="none", label="Episode initial UAV position"),
              Line2D([], [], marker="o", color="black", markerfacecolor="white", linestyle="none", label="Stage start"),
              Line2D([], [], marker="s", color="black", linestyle="none", label="Last executed stage position")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .005), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Prespecified DARS episode | train seed 42, test seed 700000 | matched reverse\n"
                 f"Executed {row['executed_steps']}/{horizon} slots; survived={row['survived']}. Trajectories stop at failure.", fontsize=14, y=.99)
    fig.subplots_adjust(left=.045, right=.985, top=.83, bottom=.30, wspace=.22)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return dict(status="complete", training_seed=42, test_seed=700000, case=FIGURE_CASE,
                selection="fixed before outcomes; no performance-based selection", bank_file=str(source),
                executed_steps=row["executed_steps"], survived=row["survived"], stages=stage_details)


def response_figure(plt, path, models, cases, dataset_scope):
    model = next((m["public"] for m in models if m["public"]["algorithm"] == "per_sac_cvx" and
                  m["public"]["training_seed"] == FIGURE_TRAINING_SEED), None)
    if model is None:
        placeholder(plt, path, "DARS four-slot switch response pending",
                    "The original DARS training-seed-42 worker has not completed all scenarios.")
        return "pending"
    fig = plt.figure(figsize=(15, 12))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1))
    distance_ax, service_ax, count_ax = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    palette = plt.get_cmap("tab10")
    x = np.arange(len(cases))
    for i, case in enumerate(cases):
        response = model["cases"][case]["switch_response"]
        label = CASE_LABELS[case].replace("\n", " ")
        values = [v["mean"] if v["mean"] is not None else np.nan for v in response["distance"]]
        counts = "/".join(str(v["n"]) for v in response["distance"])
        distance_ax.plot(range(5), values, marker="o", color=palette(i),
                         label=f"{label} [n at each point: {counts}]")
    distance_ax.set_xticks(range(5), ["Before switch", "After slot 1", "After slot 2", "After slot 3", "After slot 4"])
    distance_ax.set_ylabel("Distance to newly hot cluster center (m)\n(nearest UAV, mean across newly hot clusters)")
    distance_ax.set_title("Distance conditions on executed slots; decreasing sample counts are shown explicitly", loc="left")
    distance_ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, fontsize=8, frameon=False)
    responses = [model["cases"][c]["switch_response"] for c in cases]
    for offset, key, label, color in ((-.18, "newly_hot_offloaded_fraction", "Offloaded bits / offered bits", "#0072B2"),
                                     (.18, "newly_hot_slrr_per_arrival", "SLRR sum / arriving tasks", "#E69F00")):
        service_ax.bar(x+offset, [r[key] if r[key] is not None else np.nan for r in responses], .34, label=label, color=color)
    for offset, key, label, color in ((-.24, "eligible_events", "All eligible events", "#AAAAAA"),
                                     (0, "reached_events", "Reached (attempted)", "#0072B2"),
                                     (.24, "complete_four_slots", "Four slots executed", "#009E73")):
        bars = count_ax.bar(x+offset, [r[key] for r in responses], .23, label=label, color=color)
        count_ax.bar_label(bars, fontsize=6, rotation=90, padding=3)
    service_ax.set_ylabel("All-event service score")
    service_ax.set_title("All eligible events; post-failure service = 0", loc="left", fontsize=11)
    count_ax.set_ylabel("Planned switch events")
    count_ax.set_title("Reach and four-slot completion counts", loc="left", fontsize=11)
    for ax in (service_ax, count_ax):
        ax.set_xticks(x, [CASE_LABELS[c] for c in cases], fontsize=8)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.19), frameon=False)
    for ax in (distance_ax, service_ax, count_ax):
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    prefix = "SMOKE CHECK ONLY | " if dataset_scope == "smoke" else ""
    fig.suptitle(prefix + "DARS training seed 42 | four-slot response after each planned hotspot switch\n"
                 "Descriptive evidence conditional on one trained model; no training-seed robustness claim", fontsize=14)
    fig.subplots_adjust(left=.085, right=.98, top=.90, bottom=.13, hspace=.65, wspace=.20)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return "complete"


def number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_report(report):
    common, models = report["common_completed_training_seeds"], report["completed_models"]
    lines = ["# 未见热点模式：冻结策略评估", "", f"生成时间：{report['created_utc']}。结果目录：`{report['run_root']}`。", "",
        "**数据用途：" + ("SMOKE 功能检查；不能纳入论文正式结果或泛化结论。" if report["dataset_scope"] == "smoke"
                            else "已锁定协议的后续正式测试（已看过此前诊断，不声称历史预注册）。" if report["dataset_scope"] == "protocol_locked_evaluation"
                            else "未匹配锁定协议，仅供检查；不能自动视为正式测试。") + "**", "",
        "## 结论边界", "",
        "本报告只纳入所有场景均完成、完成标记有效且模型摘要校验通过的原始 15000 回合模型。"
        "训练支持为正向顺序 012/120/201、驻留 12–20 时隙；评估不更新策略。反向重新训练模型不纳入本报告。", "",
        "**单训练种子阶段不能宣称稳定泛化。** 测试回合提供固定策略面对不同环境随机性的证据，不能冒充独立训练重复。"
        "逐模型 bootstrap 区间只反映该检查点的测试回合不确定性；跨算法汇总仅采用五种算法共同完成的训练种子，等权平均各训练种子的均值。", "",
        f"共同完成的训练种子：**{', '.join(map(str, common)) if common else '暂无'}**；正式比较需要 DARS、ESAC、SAC、TD3、MADDPG 全部完成同一训练种子。", "",
        f"测试种子：{report['bank']['seed_min']}–{report['bank']['seed_max']}，共 {report['bank']['episodes_per_case']} 个/场景。"
        "所有算法复用同一外生场景库。图示回合事先固定为 DARS/train42/test700000/matched_reverse，缺失时不换用其他回合。", "",
        "## 指标口径", "",
        "- 核心总目标 `slrr_sum_per_horizon`：所有用户的 SLRR 总和除以计划时隙数 H。",
        "- 平均指标 `slrr_fixed_horizon`：每时隙活跃用户的平均 SLRR 相加，再除以 H。两者不可互相替代。",
        "- 存活率：完成全部 H 个时隙的测试回合比例。提前终止后的剩余时隙保留在分母内，SLRR 计零。",
        "- 切换响应：仅按场景计划确定至少驻留 4 时隙的事件；全体合格事件的服务分数包含策略失败后的零服务。"
        "距离只对实际执行的时隙有定义，失败后缺失，逐点样本数保留。到达事件表示已尝试切换首时隙；成功执行首时隙另行计数。", "",
        "## 场景与解释", "",
        "| 场景 | 与训练支持的关系 |", "|---|---|",
        "| natural_id | 原始训练分布，新测试种子 |",
        "| matched_forward | 训练支持内顺序与驻留；三阶段总长恰为 50，是条件化对照 |",
        "| matched_reverse | 未见反向顺序；与 matched_forward 匹配各阶段总时长及任务集合，UE 位置按全局时间复用 |",
        "| nonperiodic | 单回合同时包含两种切换方向，驻留仍为 12–20 |",
        "| fast_switch | 正向顺序，未见驻留 4–8 |",
        "| slow_switch | 正向顺序，未见驻留 24–32；短回合可能只有少量切换 |",
        "| unseen_pairs | 未见热点簇组合；负载结构可能随簇人数变化，需结合提供任务量解释 |", "",
        "matched_reverse 的主要配对对照为 matched_forward；其余分布变化主要与 natural_id 比较。"
        "训练/测试支持、模型来源、所有逐回合结果和失败数据均保留；不能凭轨迹观感或边缘分支占优单独宣称泛化。", "",
        "## 逐训练种子结果", ""]
    if not models:
        lines += ["暂无符合完成与一致性条件的模型。", ""]
    for model in sorted(models, key=lambda m: (m["training_seed"], ALGORITHMS.index(m["algorithm"]))):
        lines += [f"### {model['label']} · 训练种子 {model['training_seed']}", "",
            "| 场景 | 测试回合 | 总 SLRR/H | 平均 SLRR/H | 存活率 | 卸载比（bits） | 计算能耗 J |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for case, detail in model["cases"].items():
            m = detail["metrics"]
            lines.append(f"| {case} | {detail['episodes']} | {number(m['slrr_sum_per_horizon']['mean'])} | "
                         f"{number(m['slrr_fixed_horizon']['mean'])} | {number(m['survived']['mean'])} | "
                         f"{number(m['offloaded_bits_fraction']['mean'])} | {number(m['energy_comp_j']['mean'], 1)} |")
        lines += ["", "配对变化（测试场景减参考场景；95% 区间条件于此训练模型）：", "",
                  "| 配对 | 总 SLRR/H 差值 | 条件 bootstrap 95% 区间 | 存活率差值 |", "|---|---:|---|---:|"]
        for name, contrast in model["paired_contrasts"].items():
            ci = contrast["slrr_sum_per_horizon"].get("conditional_episode_bootstrap_ci95")
            ci_text = f"[{number(ci[0])}, {number(ci[1])}]" if ci else "—"
            lines.append(f"| {name} | {number(contrast['slrr_sum_per_horizon']['mean'])} | {ci_text} | {number(contrast['survived']['mean'])} |")
        lines += [""]
    lines += ["## 共同训练种子的跨算法比较", ""]
    if common:
        lines += ["每个表格值先在每个训练种子内平均测试回合，再对共同训练种子等权平均。图中若有误差线，是训练种子均值的标准差，非置信区间。", "",
                  "| 场景 | 算法 | 训练种子数 | 总 SLRR/H | 平均 SLRR/H | 存活率 |", "|---|---|---:|---:|---:|---:|"]
        for case, algorithms in report["balanced_comparison"].items():
            for algorithm, m in algorithms.items():
                lines.append(f"| {case} | {LABELS[algorithm]} | {len(common)} | {number(m['slrr_sum_per_horizon']['mean'])} | "
                             f"{number(m['slrr_fixed_horizon']['mean'])} | {number(m['survived']['mean'])} |")
    else:
        lines += ["暂无五种算法共同完成的训练种子；暂不作完整基线排名。"]
    lines += ["", "## 图", ""]
    for key in ("metrics", "trajectory", "switch_response"):
        art = report["artifacts"][key]
        lines += [f"![{key}](<{art['path']}>)", ""]
    lines += ["## 数据完整性与复现", "",
              f"场景库索引 SHA-256：`{report['bank']['sha256']}`。", "",
              "原始 summary.json、completion.json、provenance.json 与逐回合 JSONL 的种子、检查点、场景库和核心均值相互核验。"
              "完整汇总、逐训练种子配对差值、条件区间、距离逐点样本数和预指定轨迹坐标见 report_summary.json。", ""]
    if report["excluded_workers"]:
        lines += ["未纳入正式统计的目录：", ""]
        lines += [f"- `{item['path']}`：{item['reason']}" for item in report["excluded_workers"]]
    else:
        lines += ["当前没有被排除的已发现 worker 目录。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Evaluation run root; report artifacts are written here")
    args = parser.parse_args()
    root = Path(args.output).expanduser().resolve()
    candidates = [root / "bank" / "index.json", root / "scenario_bank" / "index.json"]
    index_path = next((p for p in candidates if p.is_file()), None)
    if index_path is None:
        raise FileNotFoundError(f"No bank/index.json or scenario_bank/index.json under {root}")
    bank = read_json(index_path)
    if not bank["seeds"] or len(bank["seeds"]) != len(set(bank["seeds"])):
        raise ValueError("Scenario bank has empty/duplicate test seeds")
    if set(bank["cases"]) != set(CASES):
        raise ValueError("Expected all seven protocol-locked hotspot cases")
    cases = [case for case in CASES if case in bank["cases"]]
    for case in cases:
        if [e["seed"] for e in bank["entries"][case]] != bank["seeds"]:
            raise ValueError(f"Scenario bank seed sequence mismatch: {case}")
    bank_hash = sha256(index_path)
    protocol_path = root / "protocol.json"
    protocol = read_json(protocol_path) if protocol_path.is_file() else None
    if root.name.lower() == "smoke" or set(bank["seeds"]).issubset({699990, 699991}):
        dataset_scope = "smoke"
    elif protocol is not None and protocol.get("test_seeds") == bank["seeds"]:
        lock = read_json(root / "protocol_lock.json")
        if lock["protocol_sha256"] != sha256(protocol_path) or lock["bank_index_sha256"] != bank_hash:
            raise ValueError("Locked protocol or bank index hash mismatch")
        if protocol["evaluator_sha256"] != bank["evaluator_sha256"]:
            raise ValueError("Locked evaluator and scenario bank code hashes differ")
        dataset_scope = "protocol_locked_evaluation"
    else:
        dataset_scope = "unregistered_evaluation"
    models, excluded = [], []
    directories = [p for p in sorted(root.iterdir()) if p.is_dir() and re.fullmatch(r"(?:" + "|".join(ALGORITHMS) + r")_seed\d+", p.name)]
    for directory in directories:
        if not (directory / "completion.json").is_file():
            excluded.append(dict(path=str(directory), reason="尚无 completion.json；进行中的部分结果不纳入正式汇总"))
            continue
        try:
            models.append(load_model(directory, bank, bank_hash))
        except (ValueError, KeyError, TypeError, OSError) as error:
            excluded.append(dict(path=str(directory), reason=f"完整性校验失败：{error}"))
    common, comparison, paired = compare_models(models, cases)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.facecolor": "white"})
    figures = {"metrics": root / "hotspot_metrics_v2.png", "trajectory": root / "hotspot_three_stage_v2.png",
               "switch_response": root / "hotspot_switch_response_v2.png"}
    status = metric_figure(plt, figures["metrics"], models, cases, common, comparison, dataset_scope)
    trajectory = trajectory_figure(plt, figures["trajectory"], models, bank, index_path.parent)
    switch_status = response_figure(plt, figures["switch_response"], models, cases, dataset_scope)
    report = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(), run_root=str(root), dataset_scope=dataset_scope,
        protocol_path=str(protocol_path) if protocol is not None else None,
        report_script_sha256=sha256(__file__),
        bank=dict(path=str(index_path), sha256=bank_hash, seeds=bank["seeds"], seed_min=min(bank["seeds"]),
                  seed_max=max(bank["seeds"]), episodes_per_case=len(bank["seeds"]), cases=cases),
        expected_algorithms=list(ALGORITHMS), common_completed_training_seeds=common,
        completed_models=[m["public"] for m in models], excluded_workers=excluded,
        balanced_comparison=comparison, paired_dars_minus_baseline=paired,
        inference_scope="Average checkpoint means equally over the training seeds completed by all five algorithms. "
                        "Single-training-seed results cannot establish stable generalization. Test episodes are not independent training replicates.",
        artifacts=dict(metrics=dict(path=str(figures["metrics"]), status=status),
                       trajectory=dict(path=str(figures["trajectory"]), **trajectory),
                       switch_response=dict(path=str(figures["switch_response"]), status=switch_status)))
    write_json(root / "report_summary.json", report)
    (root / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(dict(run_root=str(root), completed_models=len(models), common_training_seeds=common,
                          excluded_workers=len(excluded), report=str(root / "report.md")), ensure_ascii=False))


if __name__ == "__main__":
    main()
