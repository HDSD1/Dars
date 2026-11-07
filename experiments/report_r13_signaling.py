"""Report measured R1.3 application bytes and conditional IP service budgets.

This postprocessor reads completed benchmark outputs only. It does not load a
policy, change a simulation, measure a radio, or modify the paper or bibliography.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGES = ("up", "peer", "down")
LABELS = ("UE reports", "UAV exchange", "UE grants")
COLORS = ("#3569A8", "#DE9C3F", "#4C9675")
RATES = (64_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000, 10_000_000)


def summarize(values):
    x = np.asarray(values, dtype=float)
    return dict(mean=float(x.mean()), p95=float(np.percentile(x, 95)),
                maximum=float(x.max()), minimum=float(x.min()))


def rate_label(rate):
    return f"{rate / 1_000_000:g} Mbit/s" if rate >= 1_000_000 else f"{rate / 1000:g} kbit/s"


def sensitivities(ip_bytes, slot_seconds):
    settings = [(f"equal_{rate}", (rate, rate, rate)) for rate in RATES]
    settings += [("a2a_bottleneck", (1_000_000, 100_000, 1_000_000)),
                 ("access_bottleneck", (100_000, 1_000_000, 100_000))]
    output = []
    for name, rates in settings:
        stage_times = 8.0 * ip_bytes / np.asarray(rates, dtype=float)
        total_times = stage_times.sum(axis=1)
        row = dict(scenario=name, up_rate_bps=rates[0], peer_rate_bps=rates[1],
                   down_rate_bps=rates[2], slot_seconds=slot_seconds)
        for j, stage in enumerate(STAGES):
            row[f"ip_{stage}_bytes_mean"] = float(ip_bytes[:, j].mean())
            row[f"ip_{stage}_service_ms_mean"] = float(stage_times[:, j].mean() * 1000)
        for statistic, value in summarize(total_times).items():
            row[f"ip_serial_service_ms_{statistic}"] = value * 1000
            row[f"slot_percent_{statistic}"] = value / slot_seconds * 100
        output.append(row)
    return output


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(output, app_bytes, ip_bytes, sensitivity, slot_seconds):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": (1, 1.35)})
    means = np.vstack((app_bytes.mean(axis=0), ip_bytes.mean(axis=0)))
    bottom = np.zeros(2)
    for j, (label, color) in enumerate(zip(LABELS, COLORS)):
        axes[0].bar([0, 1], means[:, j], bottom=bottom, width=0.58, color=color, label=label)
        for x in (0, 1):
            axes[0].text(x, bottom[x] + means[x, j] / 2, f"{means[x, j]:,.0f}",
                         ha="center", va="center", color="white", fontsize=10)
        bottom += means[:, j]
    for x in (0, 1):
        axes[0].text(x, bottom[x] + 110, f"{bottom[x]:,.0f} B", ha="center", fontweight="bold")
    axes[0].set(xticks=[0, 1], xticklabels=["Application\nserialized", "Application +\nIPv4/UDP headers"],
                ylabel="Bytes per attempted slot", ylim=(0, bottom.max() * 1.19),
                title="(a) Three-stage message volume")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].set_axisbelow(True)

    equal = sensitivity[:len(RATES)]
    rates = np.asarray(RATES, dtype=float) / 1_000_000
    time_ms = np.array([r["ip_serial_service_ms_mean"] for r in equal])
    axes[1].loglog(rates, time_ms, "o-", color="#3569A8", lw=1.8,
                   label="Equal effective IP service rates")
    for fraction, color in ((0.01, "#777777"), (0.05, "#AAAAAA")):
        budget_ms = fraction * slot_seconds * 1000
        axes[1].axhline(budget_ms, color=color, linestyle="--", linewidth=1,
                       label=f"{fraction * 100:g}% of {slot_seconds:g} s slot (reference)")
    axes[1].set(xticks=rates, xticklabels=[f"{r:g}" for r in rates],
                xlabel="Effective IP service rate per stage (Mbit/s)",
                ylabel="Serial IP service budget (ms)",
                title="(b) Conditional service-time budget")
    axes[1].minorticks_off()
    axes[1].grid(which="major", alpha=0.2)
    axes[1].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="white", framealpha=1, fontsize=8)
    slow_peer, slow_access = sensitivity[-2:]
    inset = ("Rates: reports / exchange / grants (Mbit/s)\n"
             f"1 / 0.1 / 1: {slow_peer['ip_serial_service_ms_mean']:.3f} ms\n"
             f"0.1 / 1 / 0.1: {slow_access['ip_serial_service_ms_mean']:.3f} ms")
    axes[1].text(0.04, 0.05, inset, transform=axes[1].transAxes, fontsize=8,
                 bbox=dict(facecolor="white", edgecolor="#DDDDDD", alpha=0.95, pad=5))
    fig.text(0.5, 0.014,
             "Application bytes measured; IP headers counted. Timing is a rate-conditioned budget, not measured radio latency.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.045, 1, 1), w_pad=2)
    fig.savefig(output / "r13_signaling.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "r13_signaling.pdf", bbox_inches="tight")
    plt.close(fig)


def make_text(summary, protocol, sensitivity, app_bytes, ip_bytes):
    metrics = summary["metrics"]
    slot = float(protocol["slot_seconds"])
    users, uavs = protocol["num_users"], protocol["num_uavs"]
    app_total, ip_total = app_bytes.sum(axis=1), ip_bytes.sum(axis=1)
    fixed = bool(np.all(app_total == app_total[0]) and np.all(ip_total == ip_total[0]))
    lines = [
        "R1.3 信令序列化与条件通信预算实验结果", "",
        "1. 实验对象与核验范围",
        f"冻结训练种子：{summary['independent_training_seeds']}；热点场景：{', '.join(summary['scenarios'])}。",
        f"用户数 {users}，UAV 数 {uavs}，环境时隙 {slot:g} s。",
        f"共 {summary['episodes']:,} 回合、{summary['attempted_states']:,} 次尝试决策；"
        f"其中 {summary['executed_states']:,} 次执行、{summary['aborted_states']:,} 次拒绝。拒绝状态未从信令统计剔除。",
        f"原始状态重构检查 {summary['state_reconstruction_checks']:,} 次；"
        f"最终安全决策检查 {summary['final_decision_checks']:,} 次；"
        f"已有轨迹对照 {summary['archived_trace_checks']:,} 次；不一致 {summary['mismatches']:,}。",
        "上述核验检查传输前后原始状态、Actor 输出和最终安全决策一致，并与既有轨迹对照；不是无线实机测试。",
        "两套接收端副本使用同一台电脑、相同权重和确定性 CPU 推理；本实验未检验异构机载硬件的一致性。",
        "这是现有冻结模型的通信接口核验，不是新增训练结果，也不是三训练种子性能对照。", "",
        "2. 每时隙实际序列化数据量与 IP 层计数",
        "阶段                          应用序列化 bytes       IPv4/UDP bytes",
    ]
    for j, name in enumerate(("用户上报", "两机聚合交换", "用户授权")):
        lines.append(f"{name:<18} {app_bytes[:, j].mean():>14,.0f} {ip_bytes[:, j].mean():>22,.0f}")
    lines += [
        f"合计                         {app_total.mean():>14,.0f} {ip_total.mean():>22,.0f}",
        f"每次尝试决策报文数：{metrics['packet_count']['mean']:g}；全部报文数：{summary['total_packets']:,}。",
        f"最大单 IP 数据报：{metrics['max_ip_datagram_bytes']['maximum']:,.0f} B；"
        f"参考 MTU：{protocol['ip_budget']['assumed_mtu_bytes']} B。",
        f"{app_total.mean():,.0f} B 为参考消息格式实际编码后的应用数据量；"
        f"{ip_total.mean():,.0f} B 为此数据量加每包 20 B IPv4 无选项首部和 8 B UDP 首部的计数。",
        "消息字段和报文布局是本实验定义的可复现参考协议；RFC 仅支持标准首部长度，不为自定义字段布局背书。",
        "用户上报包括 ID、位置、任务位数和活动标记；两机交换包括本机位置、电量及各自收到的用户记录。",
        "报告分组在部署时预置：报告 UAV = 用户 ID modulo 2；服务关联仍由原算法动态决定，无需为报告分组额外广播实时 UAV 位置。",
        "拒绝记录来自原策略的安全可行性检查；与既有轨迹一致，不是通信实验引起的拒绝或丢包。",
        "授权包括用户 ID、服务 UAV（或本地处理标记）、最终卸载比例、带宽、频段起点及执行/拒绝状态。",
        "两架 UAV 具有相同冻结模型和全局状态，各自重算完整联合决策；CPU 分配和本机运动指令不再重复跨机发送。",
        "冻结部署不交换梯度、经验池或每时隙模型更新；初始模型分发不属于此处稳态每时隙预算。",
        ("固定长度字段使每时隙字节数恒定；多回合重复用于验证消息内容和决策一致性，不构成字节数的随机置信区间。"
         if fixed else "字节数随观测内容变化；详细均值、P95 和最大值见原始 summary.json。"), "",
        "3. 通信服务时间敏感性（计算预算，不是实测端到端时延）",
        "使用 T_IP = 8 * (S_up/R_up + S_peer/R_peer + S_down/R_down)。",
        "S 是各阶段 IP 字节数，R 是成功交付的有效 IP 层服务率，三个阶段按完全串行相加。",
        "R_eff 已作为有效服务率，不另乘 PER 或重传系数；不把 MAC/PHY 开销、竞争等待、排队和推理时间假装成实测值。",
        "三阶段同速                 平均预算 ms       P95 ms       占8秒时隙百分比".replace("8秒", f"{slot:g}秒"),
    ]
    for row in sensitivity[:len(RATES)]:
        lines.append(f"{rate_label(row['up_rate_bps']):<23} {row['ip_serial_service_ms_mean']:>11.4f}"
                     f" {row['ip_serial_service_ms_p95']:>12.4f} {row['slot_percent_mean']:>17.5f}%")
    for label, row in zip(("空空链路较慢", "上下行接入链路较慢"), sensitivity[-2:]):
        rates = " / ".join(rate_label(row[f"{s}_rate_bps"]) for s in STAGES)
        lines.append(f"{label}（上报 / 机间 / 授权 = {rates}）："
                     f"{row['ip_serial_service_ms_mean']:.4f} ms，占时隙 {row['slot_percent_mean']:.5f}%。")
    lines += ["", "4. 归一化预算参考（不是合格阈值）"]
    for fraction in (0.01, 0.05):
        rates = 8 * ip_total / (fraction * slot)
        lines.append(f"若把通信串行服务预算设为时隙的 {fraction * 100:g}%（{fraction * slot * 1000:g} ms），"
                     f"三阶段同速所需服务率均值为 {rates.mean()/1000:.3f} kbit/s，"
                     f"P95 为 {np.percentile(rates,95)/1000:.3f} kbit/s。")
    lines += [
        "这些百分比只用于解释量级，不是论文合格线，也不是所选无线链路已能达到该有效速率的证据。", "",
        "5. 覆盖与单机收发负担",
        f"位于报告 UAV 的 150 m 任务服务半径以外的用户报告占比："
        f"{summary['outside_service_radius_report_fraction']*100:.4f}%。",
    ]
    for label, name in (("每时隙最远报告水平距离", "max_reporting_horizontal_m"),
                        ("每时隙最远报告斜距", "max_reporting_slant_m"),
                        ("两机间距离", "inter_uav_distance_m")):
        v = metrics[name]
        lines.append(f"{label}：均值 {v['mean']:.3f} m，P95 {v['p95']:.3f} m，最大 {v['maximum']:.3f} m。")
    for k in range(uavs):
        tx, rx = metrics[f"uav{k}_tx_ip_bytes"], metrics[f"uav{k}_rx_ip_bytes"]
        lines.append(f"UAV {k} 每时隙 IP 发送/接收：均值 {tx['mean']:.3f}/{rx['mean']:.3f} B；"
                     f"P95 {tx['p95']:.3f}/{rx['p95']:.3f} B；最大 {tx['maximum']:.0f}/{rx['maximum']:.0f} B。")
    lines += [
        "这里假设用户能连接报告 UAV 的控制链路；任务服务半径不能证明控制覆盖，距离统计本身也不能证明链路可用。",
        "单机收发统计按端点计数，不能将所有端点收发量再加到三阶段网络总字节数中。", "",
        "6. 分组透明性：训练种子 / 场景 / 尝试数 / 拒绝数 / 半径外报告占比",
    ]
    for group in summary["groups"]:
        lines.append(f"{group['training_seed']} / {group['case']} / {group['attempts']} / "
                     f"{group['abort_attempts']} / {group['outside_service_radius_report_fraction']*100:.4f}%")
    lines += [
        "", "7. 可支持与不可支持的结论",
        "可支持：在所列冻结模型与存档测试状态上，所设计参考消息能够无损重构决策所需状态，并复现最终安全决策；"
        "应用字节量、IP 首部预算和给定有效吞吐下的串行服务预算可复算。",
        "尚不可支持：真实无线端到端时延、所有链路可达、机载处理器实时性、无线/推理能耗、丢包鲁棒性，"
        "或通信时延不会降低 SLRR。该实验没有向环境注入延迟，也没有改变奖励或重新训练。",
        "此图不包含推理/关联/资源分配/安全层计算耗时；应与计算计时实验区分，再讨论整体在线决策耗时。", "",
        "计数依据：",
    ]
    for source in protocol["sources"]:
        lines.append(f"{source['url']}  ({source['supports']})")
    return "\n".join(lines) + "\n"


def make_tex(summary, protocol, sensitivity, app_bytes, ip_bytes):
    slot = float(protocol["slot_seconds"])
    app, ip = app_bytes.mean(axis=0), ip_bytes.mean(axis=0)
    total = float(ip.sum())
    seeds = ", ".join(str(x) for x in summary["independent_training_seeds"])
    lines = [
        "% Standalone English fragment; input manually if desired. No manuscript edits were made.",
        r"\par\noindent\textbf{R1.3: reproducible signaling accounting.}",
        f"We evaluated the reference message interface using frozen DARS training seeds {seeds}, "
        f"{len(summary['scenarios'])} hotspot scenarios, and {summary['episodes']} archived test episodes. "
        f"The evaluation covered {summary['attempted_states']} attempted decisions, including "
        f"{summary['executed_states']} executed decisions and {summary['aborted_states']} rejected decisions. "
        "Rejected decisions were retained in the signaling accounting.",
        f"There were {summary['state_reconstruction_checks']} exact raw-state reconstruction checks, "
        f"{summary['final_decision_checks']} final-decision comparisons, and "
        f"{summary['archived_trace_checks']} comparisons with archived trajectories, with "
        f"{summary['mismatches']} mismatches.",
        r"\par\noindent The reference protocol uses binary64 continuous fields and explicit user, slot, and sender identifiers. "
        "Reporting groups are preconfigured by user ID modulo 2 and do not alter dynamic service associations. "
        "Each user sends one state report to its reporting UAV; the two UAVs exchange two aggregate messages; "
        "each user then receives one execution grant or explicit abort message. "
        "Each UAV reconstructs the same global state and independently evaluates the same frozen joint policy "
        "and safety/resource-allocation routines. Thus, locally reproducible CPU allocations and UAV motion "
        "commands do not require a further command exchange in this reference design.",
        r"\begin{center}", r"\begin{tabular}{lrr}", r"\hline",
        r"Stage & Application bytes & IPv4/UDP bytes \\", r"\hline",
    ]
    for label, a, b in zip(LABELS, app, ip):
        lines.append(f"{label} & {a:.0f} & {b:.0f} " + r"\\")
    lines += [f"Total & {app.sum():.0f} & {total:.0f} " + r"\\", r"\hline",
              r"\end{tabular}", r"\end{center}",
        f"The {app.sum():.0f} application bytes per attempted slot were measured by actual serialization. "
        f"The {total:.0f} IP bytes were calculated by adding a 20-byte IPv4 header without options "
        "and an 8-byte UDP header per datagram (RFC 791 and RFC 768). "
        f"There are {summary['metrics']['packet_count']['mean']:.0f} datagrams per attempted slot. "
        "The fixed-length schema explains the constant byte count; repeated episodes verify the contents "
        "and decisions rather than estimating uncertainty in a deterministic packet size. "
        "The message layout is a reference implementation, not a layout prescribed by those RFCs.",
        r"\par\noindent For an explicitly assumed delivered IP-layer service rate $R_j$ in stage $j$, "
        "the fully serial service-time budget is",
        r"\[ T_{\mathrm{IP}} = 8\left(\frac{S_{\mathrm{up}}}{R_{\mathrm{up}}} + "
        r"\frac{S_{\mathrm{peer}}}{R_{\mathrm{peer}}} + \frac{S_{\mathrm{down}}}{R_{\mathrm{down}}}\right). \]",
        r"Here $S_j$ denotes IP bytes. Since $R_j$ is an effective delivered service rate, "
        "no separate packet-error or retransmission multiplier is applied.",
        r"\begin{center}", r"\begin{tabular}{rrr}", r"\hline",
        r"Equal rate (kbit/s) & IP service budget (ms) & Slot fraction (\%) \\", r"\hline",
    ]
    for row in sensitivity[:len(RATES)]:
        lines.append(f"{row['up_rate_bps']/1000:g} & {row['ip_serial_service_ms_mean']:.4f} & "
                     f"{row['slot_percent_mean']:.5f} " + r"\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{center}",
        f"For reports/peer exchange/grants at 1/0.1/1 Mbit/s, the budget is "
        f"{sensitivity[-2]['ip_serial_service_ms_mean']:.4f} ms; at 0.1/1/0.1 Mbit/s, "
        f"it is {sensitivity[-1]['ip_serial_service_ms_mean']:.4f} ms. "
        f"For the {slot:g}-s simulation slot, equal stage rates of "
        f"{8*total/(0.01*slot)/1000:.3f} and {8*total/(0.05*slot)/1000:.3f} kbit/s "
        r"correspond to reference service budgets of 1\% and 5\% of the slot, respectively. "
        "These percentages are descriptive budget references, not acceptance thresholds.",
        r"\par\noindent The calculation is conditional on control-link coverage and the specified effective service rates. "
        f"In the replayed states, {summary['outside_service_radius_report_fraction']*100:.4f}" + r"\% "
        "of user reports originate outside the reporting UAV's 150-m task-service radius. "
        "This radius does not establish control-link coverage. The analysis does not measure radio latency, "
        "contention, queueing, link establishment, onboard inference latency, or communication/inference energy. "
        "It does not establish that communication delay has no effect on SLRR, because no communication delay "
        "or packet loss was injected into the environment. Frozen deployment exchanges no per-slot gradients, "
        "replay samples, or model updates; initial model distribution is outside this steady-state accounting.",
    ]
    return "\n".join(lines) + "\n"


def run(args):
    results = Path(args.results).resolve()
    protocol = json.loads((results / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise ValueError("The benchmark must complete before generating the report.")
    with (results / "slots.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != summary["attempted_states"]:
        raise ValueError("slots.csv count differs from the completed summary.")
    app = np.array([[float(r[f"app_{s}_bytes"]) for s in STAGES] for r in rows])
    ip = np.array([[float(r[f"ip_{s}_bytes"]) for s in STAGES] for r in rows])
    for prefix, values in (("app", app), ("ip", ip)):
        if not np.isclose(values.sum(axis=1).mean(), summary["metrics"][f"{prefix}_total_bytes"]["mean"]):
            raise ValueError(f"{prefix} byte accounting differs from the completed summary.")
    sensitivity = sensitivities(ip, float(protocol["slot_seconds"]))
    write_csv(results / "sensitivity.csv", sensitivity)
    make_figure(results, app, ip, sensitivity, float(protocol["slot_seconds"]))
    (results / "report.txt").write_text(make_text(summary, protocol, sensitivity, app, ip), encoding="utf-8")
    (results / "report.tex").write_text(make_tex(summary, protocol, sensitivity, app, ip), encoding="ascii")
    print(json.dumps(dict(results=str(results), attempted_states=len(rows),
                         application_bytes_mean=float(app.sum(axis=1).mean()),
                         ip_bytes_mean=float(ip.sum(axis=1).mean()),
                         outputs=["sensitivity.csv", "r13_signaling.png", "r13_signaling.pdf", "report.txt", "report.tex"]),
                     ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Completed R1.3 benchmark output directory")
    run(parser.parse_args())
