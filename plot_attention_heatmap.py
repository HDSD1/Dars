import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import GROUND_LENGTH, GROUND_WIDTH, M, N_CLUSTERS, SLOT_NUM
from DARS import PERSACAgent, decompose_state, DEVICE
from state_normalization import StateNormalization
from UAV_env import UAVEnv


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(ROOT, "results", "new_v13_ms_seed43", "per_sac_cvx.pth")
OUT_DIR = os.path.join(ROOT, "results", "sensitivity")

PHASE_NAMES = [
    "Phase I: clusters 0 & 1 hot",
    "Phase II: clusters 1 & 2 hot",
    "Phase III: clusters 2 & 3 hot",
]

CLUSTER_COLORS = ["#D62728", "#2CA02C", "#1F77B4", "#FF7F0E"]


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.unicode_minus": False,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def phase_steps():
    third = SLOT_NUM / 3
    return [
        max(0, int(third) - 2),
        min(SLOT_NUM - 1, int(third) + 2),
        min(SLOT_NUM - 1, int(2 * third) + 2),
    ]


def get_actor_attention(agent, norm_state):
    state_tensor = torch.as_tensor(norm_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    _, user_feat = decompose_state(state_tensor)
    attn = agent.actor.user_attn
    with torch.no_grad():
        x = attn.user_proj(user_feat)
        _, weights = attn.mha(
            x, x, x,
            need_weights=True,
            average_attn_weights=False,
        )
    # weights: (batch, heads, query_users, key_users).
    weights = weights.squeeze(0).detach().cpu().numpy()
    received = weights.mean(axis=(0, 1))
    received = received / max(received.sum(), 1e-12)
    return received, weights


def run_episode_and_capture(model_path, seed, capture_steps):
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = PERSACAgent()
    agent.load(model_path)
    agent.actor.eval()

    env = UAVEnv(use_cvx_alloc=True, use_hotspot=True)
    normalizer = StateNormalization()
    state = env.reset()

    captures = {}
    for step in range(SLOT_NUM):
        norm_state = normalizer.normalize(state)
        if step in capture_steps:
            score, matrix = get_actor_attention(agent, norm_state)
            captures[step] = {
                "score": score,
                "matrix": matrix,
                "ue_pos": env.loc_ue.copy(),
                "active": env.active_mask.copy(),
                "task": env.task_list.copy(),
                "phase": env._get_current_phase(),
                "hot_clusters": list(env._hotspot_phases[env._get_current_phase()]),
                "cluster_ids": env.user_cluster_ids.copy(),
            }
        action = agent.select_action(norm_state, noise_std=0.0)
        state, _, done, _ = env.step(action)
        if done:
            break
    return captures, env


def draw_panel(ax, capture, title, vmin, vmax):
    pos = capture["ue_pos"]
    score = capture["score"]
    active = capture["active"]
    cluster_ids = capture["cluster_ids"]
    hot_clusters = set(capture["hot_clusters"])

    ax.set_facecolor("#FAFAFA")
    sc = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=score,
        s=70 + 360 * score / max(score.max(), 1e-12),
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        edgecolor=np.where(active, "#111111", "#A0A0A0"),
        linewidth=0.7,
        alpha=np.where(active, 0.96, 0.42),
        zorder=3,
    )

    for c in range(N_CLUSTERS):
        mask = cluster_ids == c
        if not np.any(mask):
            continue
        cx, cy = pos[mask].mean(axis=0)
        is_hot = c in hot_clusters
        ax.scatter(
            [cx], [cy],
            marker="*",
            s=250 if is_hot else 130,
            color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)] if is_hot else "#777777",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        ax.text(
            cx + 9,
            cy + 9,
            f"C{c}{' hot' if is_hot else ''}",
            fontsize=9,
            fontweight="bold" if is_hot else "normal",
            color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)] if is_hot else "#555555",
            zorder=6,
        )

    ax.set_xlim(0, GROUND_LENGTH)
    ax.set_ylim(0, GROUND_WIDTH)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    return sc


def plot_heatmaps(captures, save_prefix):
    ordered_steps = sorted(captures)
    scores = np.concatenate([captures[s]["score"] for s in ordered_steps])
    vmin, vmax = float(scores.min()), float(scores.max())

    fig, axes = plt.subplots(1, len(ordered_steps), figsize=(16, 5.3), constrained_layout=True)
    if len(ordered_steps) == 1:
        axes = [axes]

    sc = None
    for ax, step in zip(axes, ordered_steps):
        cap = captures[step]
        phase = cap["phase"]
        title = f"t={step}, {PHASE_NAMES[phase]}"
        sc = draw_panel(ax, cap, title, vmin, vmax)

    cbar = fig.colorbar(sc, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("Received self-attention score")

    outputs = []
    for ext in ("png", "pdf"):
        path = f"{save_prefix}.{ext}"
        plt.savefig(path)
        outputs.append(path)
    plt.close(fig)

    # Save individual plots
    for step in ordered_steps:
        cap = captures[step]
        fig, ax = plt.subplots(figsize=(6, 5.3))
        # No title for individual plots
        sc = draw_panel(ax, cap, "", vmin, vmax)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label("Received self-attention score")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            path = f"{save_prefix}_step{step}.{ext}"
            plt.savefig(path)
            outputs.append(path)
        plt.close(fig)
    return outputs


def plot_phase_shift(captures, save_prefix):
    steps = sorted(captures)
    if len(steps) < 2:
        return []
    first, second = steps[0], steps[1]
    cap0, cap1 = captures[first], captures[second]
    delta = cap1["score"] - cap0["score"]
    pos = cap1["ue_pos"]
    cluster_ids = cap1["cluster_ids"]

    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    lim = float(np.max(np.abs(delta)))
    sc = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=delta,
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        s=95 + 450 * np.abs(delta) / max(lim, 1e-12),
        edgecolor="black",
        linewidth=0.55,
        alpha=0.9,
    )
    for c in range(N_CLUSTERS):
        mask = cluster_ids == c
        if np.any(mask):
            cx, cy = pos[mask].mean(axis=0)
            ax.text(cx + 9, cy + 9, f"C{c}", fontsize=10, fontweight="bold")
    ax.set_xlim(0, GROUND_LENGTH)
    ax.set_ylim(0, GROUND_WIDTH)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    # No title for shift plot
    # ax.set_title(f"Attention Shift: t={first} to t={second}")
    ax.grid(True, linestyle="--", alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Attention score change")
    plt.tight_layout()

    outputs = []
    for ext in ("png", "pdf"):
        path = f"{save_prefix}.{ext}"
        plt.savefig(path)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_attention_csv(captures, path):
    fields = [
        "step", "phase", "user_id", "cluster_id", "is_hot_cluster",
        "is_active", "task_bits", "x", "y", "attention_score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in sorted(captures):
            cap = captures[step]
            hot_clusters = set(cap["hot_clusters"])
            for user_id in range(M):
                writer.writerow({
                    "step": step,
                    "phase": cap["phase"],
                    "user_id": user_id,
                    "cluster_id": int(cap["cluster_ids"][user_id]),
                    "is_hot_cluster": int(cap["cluster_ids"][user_id] in hot_clusters),
                    "is_active": int(cap["active"][user_id]),
                    "task_bits": float(cap["task"][user_id]),
                    "x": float(cap["ue_pos"][user_id, 0]),
                    "y": float(cap["ue_pos"][user_id, 1]),
                    "attention_score": float(cap["score"][user_id]),
                })
    return path


def main():
    parser = argparse.ArgumentParser(description="Plot DARS user-attention spatial heatmaps.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--steps", type=str, default="",
                        help="Comma-separated slots. Default captures one slot in each hotspot phase.")
    parser.add_argument("--out", type=str, default=os.path.join(OUT_DIR, "fig_attention_heatmap"))
    args = parser.parse_args()

    steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()] if args.steps else phase_steps()
    captures, _ = run_episode_and_capture(args.model, args.seed, set(steps))
    if not captures:
        raise RuntimeError("No attention snapshots were captured.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    outputs = plot_heatmaps(captures, args.out)
    outputs += plot_phase_shift(captures, args.out + "_shift")
    csv_path = write_attention_csv(captures, args.out + ".csv")
    print(f"Saved summary: {csv_path}")
    for path in outputs:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
