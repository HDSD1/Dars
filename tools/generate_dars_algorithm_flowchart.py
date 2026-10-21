"""Generate the pictorial DARS algorithm flowchart.

The SVG is the editable source. A standalone HTML wrapper is also emitted so a
headless browser can export a high-resolution PNG and a vector PDF.
"""

from __future__ import annotations

import argparse
from pathlib import Path


W = 1800
H = 1120


PAPER = {
    "bg": "#ffffff",
    "ink": "#17212b",
    "muted": "#66727f",
    "line": "#263746",
    "panel": "#fbfcfd",
    "soft": "#eef2f5",
    "blue": "#4f82bd",
    "blue_soft": "#eaf2fb",
    "orange": "#df7a36",
    "orange_soft": "#fff0e5",
    "green": "#5f9f58",
    "green_soft": "#eaf5e7",
    "red": "#bd5552",
    "red_soft": "#faeaea",
    "gold": "#c69a2b",
    "gold_soft": "#fff7db",
    "violet": "#7868a6",
    "violet_soft": "#f0edf8",
}


THEME = {
    "bg": "var(--background)",
    "ink": "var(--foreground)",
    "muted": "var(--muted-foreground)",
    "line": "var(--foreground)",
    "panel": "var(--card)",
    "soft": "var(--muted)",
    "blue": "var(--viz-series-1)",
    "blue_soft": "var(--muted)",
    "orange": "var(--viz-series-2)",
    "orange_soft": "var(--muted)",
    "green": "var(--viz-series-3)",
    "green_soft": "var(--muted)",
    "red": "var(--viz-series-4)",
    "red_soft": "var(--muted)",
    "gold": "var(--viz-series-5)",
    "gold_soft": "var(--muted)",
    "violet": "var(--viz-series-6)",
    "violet_soft": "var(--muted)",
}


def build_svg(c: dict[str, str], include_background: bool) -> str:
    out: list[str] = []

    def add(s: str) -> None:
        out.append(s)

    def text(
        x: float,
        y: float,
        value: str,
        size: int = 28,
        anchor: str = "middle",
        fill: str | None = None,
        weight: int = 400,
        family: str = "Arial, sans-serif",
        italic: bool = False,
        rotate: float | None = None,
    ) -> None:
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        style = ' font-style="italic"' if italic else ""
        add(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill or c["ink"]}" letter-spacing="0"{style}{transform}>{value}</text>'
        )

    def rect(
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str | None = None,
        sw: float = 2,
        rx: float = 4,
        dash: str | None = None,
        opacity: float | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        opacity_attr = f' opacity="{opacity}"' if opacity is not None else ""
        add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"'
            f'{dash_attr}{opacity_attr}/>'
        )

    def circle(
        cx: float,
        cy: float,
        r: float,
        fill: str = "none",
        stroke: str | None = None,
        sw: float = 2,
        opacity: float | None = None,
    ) -> None:
        opacity_attr = f' opacity="{opacity}"' if opacity is not None else ""
        add(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
            f'stroke="{stroke or "none"}" stroke-width="{sw}"{opacity_attr}/>'
        )

    def line(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str | None = None,
        sw: float = 2,
        dash: str | None = None,
        marker: str | None = None,
        opacity: float | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = f' marker-end="url(#{marker})"' if marker else ""
        opacity_attr = f' opacity="{opacity}"' if opacity is not None else ""
        add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke or c["line"]}" stroke-width="{sw}" '
            f'stroke-linecap="round"{dash_attr}{marker_attr}{opacity_attr}/>'
        )

    def path(
        d: str,
        stroke: str | None = None,
        sw: float = 2,
        fill: str = "none",
        dash: str | None = None,
        marker: str | None = None,
        opacity: float | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = f' marker-end="url(#{marker})"' if marker else ""
        opacity_attr = f' opacity="{opacity}"' if opacity is not None else ""
        add(
            f'<path d="{d}" stroke="{stroke or c["line"]}" stroke-width="{sw}" '
            f'fill="{fill}" stroke-linecap="round" stroke-linejoin="round"'
            f'{dash_attr}{marker_attr}{opacity_attr}/>'
        )

    def badge(cx: float, cy: float, value: str, color: str) -> None:
        circle(cx, cy, 19, color)
        text(cx, cy + 8, value, 23, fill=c["bg"], weight=500)

    def battery(x: float, y: float, level: float, color: str, scale: float = 1.0) -> None:
        rect(x, y, 42 * scale, 20 * scale, "none", c["ink"], 2, 2)
        rect(
            x + 4 * scale,
            y + 4 * scale,
            30 * scale * max(0.0, min(1.0, level)),
            12 * scale,
            color,
            None,
            0,
            1,
        )
        rect(x + 42 * scale, y + 6 * scale, 5 * scale, 8 * scale, c["ink"], None, 0, 1)

    def user(x: float, y: float, active: bool = True, hot: bool = False) -> None:
        color = c["orange"] if active else c["muted"]
        circle(x, y - 7, 5.5, color)
        path(
            f"M{x - 9},{y + 12} Q{x},{y + 1} {x + 9},{y + 12}",
            color,
            3,
        )
        if hot:
            circle(x, y, 17, "none", c["orange"], 2, 0.45)

    def drone(x: float, y: float, scale: float = 1.0, color: str | None = None) -> None:
        col = color or c["ink"]
        line(x - 18 * scale, y, x + 18 * scale, y, col, 3 * scale)
        line(x - 11 * scale, y, x - 22 * scale, y - 12 * scale, col, 3 * scale)
        line(x + 11 * scale, y, x + 22 * scale, y - 12 * scale, col, 3 * scale)
        circle(x - 25 * scale, y - 14 * scale, 8 * scale, "none", col, 2.5 * scale)
        circle(x + 25 * scale, y - 14 * scale, 8 * scale, "none", col, 2.5 * scale)
        rect(
            x - 12 * scale,
            y - 5 * scale,
            24 * scale,
            13 * scale,
            col,
            col,
            1,
            3 * scale,
        )
        path(
            f"M{x - 7 * scale},{y + 8 * scale} L{x - 3 * scale},{y + 16 * scale} "
            f"M{x + 7 * scale},{y + 8 * scale} L{x + 3 * scale},{y + 16 * scale}",
            col,
            2.5 * scale,
        )

    def wct(x: float, y: float, scale: float = 1.0) -> None:
        col = c["green"]
        path(
            f"M{x},{y + 19 * scale} L{x},{y - 12 * scale} "
            f"M{x - 10 * scale},{y + 19 * scale} L{x + 10 * scale},{y + 19 * scale}",
            col,
            3 * scale,
        )
        path(
            f"M{x - 7 * scale},{y - 7 * scale} Q{x},{y - 14 * scale} {x + 7 * scale},{y - 7 * scale}",
            col,
            2.5 * scale,
        )
        path(
            f"M{x - 13 * scale},{y - 13 * scale} Q{x},{y - 27 * scale} {x + 13 * scale},{y - 13 * scale}",
            col,
            2 * scale,
            dash="5 4",
        )

    def neural_net(
        x: float,
        y: float,
        w: float,
        h: float,
        color: str,
        layers: tuple[int, ...] = (4, 5, 3),
    ) -> None:
        xs = [x + i * w / (len(layers) - 1) for i in range(len(layers))]
        nodes: list[list[tuple[float, float]]] = []
        for layer_index, n in enumerate(layers):
            ys = [y + (j + 1) * h / (n + 1) for j in range(n)]
            nodes.append([(xs[layer_index], yy) for yy in ys])
        for left, right in zip(nodes, nodes[1:]):
            for x1, y1 in left:
                for x2, y2 in right:
                    line(x1, y1, x2, y2, c["line"], 1.1, opacity=0.32)
        for layer in nodes:
            for nx, ny in layer:
                circle(nx, ny, 5.5, color, c["ink"], 1)

    def map_scene(
        x: float,
        y: float,
        w: float,
        h: float,
        compact: bool = False,
        show_labels: bool = True,
    ) -> None:
        rect(x, y, w, h, c["panel"], c["line"], 2, 4)
        spacing = 34 if not compact else 24
        gx = x + spacing
        while gx < x + w:
            line(gx, y, gx, y + h, c["muted"], 0.7, opacity=0.18)
            gx += spacing
        gy = y + spacing
        while gy < y + h:
            line(x, gy, x + w, gy, c["muted"], 0.7, opacity=0.18)
            gy += spacing

        circle(x + w * 0.27, y + h * 0.28, min(w, h) * 0.18, c["orange_soft"], None, 0, 0.55)
        circle(x + w * 0.71, y + h * 0.68, min(w, h) * 0.17, c["blue_soft"], None, 0, 0.7)
        circle(x + w * 0.77, y + h * 0.22, min(w, h) * 0.11, c["red_soft"], None, 0, 0.5)

        user(x + w * 0.18, y + h * 0.21, True, True)
        user(x + w * 0.30, y + h * 0.35, True, True)
        user(x + w * 0.20, y + h * 0.42, False)
        user(x + w * 0.43, y + h * 0.22, True)
        user(x + w * 0.62, y + h * 0.63, True, True)
        user(x + w * 0.74, y + h * 0.74, True, True)
        user(x + w * 0.84, y + h * 0.59, False)
        user(x + w * 0.79, y + h * 0.25, True, True)

        wct(x + w * 0.10, y + h * 0.84, 0.75 if compact else 1.0)
        wct(x + w * 0.90, y + h * 0.84, 0.75 if compact else 1.0)

        circle(x + w * 0.39, y + h * 0.48, min(w, h) * 0.23, "none", c["blue"], 2, 0.45)
        circle(x + w * 0.68, y + h * 0.40, min(w, h) * 0.23, "none", c["blue"], 2, 0.45)
        drone(x + w * 0.39, y + h * 0.48, 0.72 if compact else 1.0, c["blue"])
        drone(x + w * 0.68, y + h * 0.40, 0.72 if compact else 1.0, c["blue"])

        path(
            f"M{x + w * 0.39},{y + h * 0.48} "
            f"C{x + w * 0.48},{y + h * 0.28} {x + w * 0.58},{y + h * 0.25} "
            f"{x + w * 0.68},{y + h * 0.40}",
            c["orange"],
            3,
            dash="8 7",
            marker="arrow-orange",
        )
        if show_labels:
            text(x + 14, y + 30, "dynamic hotspots", 22, "start", c["muted"])
            battery(x + w - 72, y + 14, 0.58, c["green"], 0.8)

    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="dars-title dars-desc">'
    )
    add("<title id=\"dars-title\">Pictorial flow of the DARS algorithm</title>")
    add(
        "<desc id=\"dars-desc\">The environment state is encoded by task-demand "
        "attention, processed by a movement-only SAC actor, passed through user "
        "association and joint offloading/resource optimization, and learned with "
        "n-step dual-priority replay and "
        "twin critics.</desc>"
    )
    add("<defs>")
    for name, color in (
        ("arrow", c["line"]),
        ("arrow-blue", c["blue"]),
        ("arrow-orange", c["orange"]),
        ("arrow-green", c["green"]),
        ("arrow-red", c["red"]),
        ("arrow-gold", c["gold"]),
        ("arrow-violet", c["violet"]),
    ):
        add(
            f'<marker id="{name}" markerWidth="10" markerHeight="10" refX="8" refY="5" '
            f'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,5 L0,10 z" fill="{color}"/></marker>'
        )
    add(
        f'<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
        f'<feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="{c["line"]}" flood-opacity="0.12"/>'
        f"</filter>"
    )
    add("</defs>")

    if include_background:
        rect(0, 0, W, H, c["bg"], None, 0, 0)

    # Section rails
    line(42, 64, 1758, 64, c["line"], 2)
    text(55, 50, "DECISION AND EXECUTION", 26, "start", c["muted"], 500)
    line(42, 650, 1758, 650, c["line"], 2)
    text(55, 636, "LEARNING LOOP", 26, "start", c["muted"], 500)

    # 1. Environment
    badge(75, 100, "1", c["blue"])
    text(105, 110, "Environment", 31, "start", c["ink"], 500)
    map_scene(55, 130, 370, 420)

    # State channels
    path("M425,335 C452,335 456,240 478,240", c["blue"], 4, marker="arrow-blue")
    text(456, 218, "s<tspan baseline-shift=\"sub\" font-size=\"20\">t</tspan>", 31, fill=c["blue"], italic=True)
    rect(478, 145, 145, 385, c["soft"], c["line"], 2, 4)
    text(550, 179, "state", 23, fill=c["muted"], weight=500)
    # Four compact visual state channels
    rect(495, 198, 111, 64, c["panel"], c["line"], 1.5, 3)
    battery(513, 220, 0.62, c["green"], 0.85)
    battery(565, 220, 0.35, c["red"], 0.85)
    rect(495, 277, 111, 64, c["panel"], c["line"], 1.5, 3)
    drone(530, 312, 0.55, c["blue"])
    path("M557,325 Q575,292 595,306", c["orange"], 2.2, marker="arrow-orange")
    rect(495, 356, 111, 64, c["panel"], c["line"], 1.5, 3)
    user(520, 391, True, True)
    user(551, 391, False)
    user(582, 391, True, True)
    rect(495, 435, 111, 64, c["panel"], c["line"], 1.5, 3)
    wct(520, 473, 0.7)
    circle(575, 468, 17, "none", c["violet"], 3)
    line(575, 468, 575, 457, c["violet"], 2)
    line(575, 468, 584, 474, c["violet"], 2)

    # 2. Task-demand attention
    badge(660, 100, "2", c["orange"])
    text(690, 110, "Demand attention", 31, "start", c["ink"], 500)
    rect(650, 145, 275, 385, c["orange_soft"], c["orange"], 2.5, 4)
    # User tokens
    token_y = [205, 260, 315, 370, 425, 480]
    token_r = [11, 18, 9, 20, 12, 16]
    for idx, (ty, tr) in enumerate(zip(token_y, token_r)):
        token_color = c["orange"] if idx in (1, 3, 5) else c["muted"]
        circle(690, ty, tr, token_color, c["ink"], 1, 0.9)
        line(708, ty, 786, 335, token_color, 1.5 + tr / 10, opacity=0.42)
    # Attention lens and weighted embedding
    circle(790, 335, 50, c["panel"], c["orange"], 4)
    circle(790, 335, 25, c["orange_soft"], c["orange"], 2)
    path("M821,372 L852,410", c["orange"], 7)
    for i, bar_w in enumerate((44, 77, 55, 89, 63)):
        rect(835, 245 + i * 38, bar_w, 17, c["orange"], None, 0, 2, 0.32 + i * 0.1)
    text(790, 185, "MHA", 23, fill=c["orange"], weight=500)
    text(790, 507, "h<tspan baseline-shift=\"sub\" font-size=\"18\">t</tspan>", 29, fill=c["orange"], italic=True)

    path("M623,335 L650,335", c["line"], 3, marker="arrow")
    path("M925,335 L965,335", c["orange"], 4, marker="arrow-orange")

    # 3. Actor and twin critic heads
    badge(1000, 100, "3", c["violet"])
    text(1030, 110, "SAC policy", 31, "start", c["ink"], 500)
    rect(970, 145, 275, 175, c["violet_soft"], c["violet"], 2.5, 4)
    neural_net(1000, 177, 155, 100, c["violet"], (4, 5, 3))
    text(1190, 215, "Actor", 25, fill=c["ink"], weight=500)
    text(
        1190,
        250,
        "&#960;<tspan baseline-shift=\"sub\" font-size=\"18\">&#952;</tspan>",
        30,
        fill=c["violet"],
        italic=True,
        family="Cambria Math, Times New Roman, serif",
    )

    rect(970, 355, 275, 175, c["blue_soft"], c["blue"], 2.5, 4)
    neural_net(995, 385, 126, 105, c["blue"], (3, 4, 2))
    neural_net(1080, 385, 126, 105, c["blue"], (3, 4, 2))
    text(1192, 414, "Twin", 23, fill=c["ink"], weight=500)
    text(1192, 445, "critics", 23, fill=c["ink"], weight=500)
    text(1192, 485, "Q<tspan baseline-shift=\"sub\" font-size=\"17\">1</tspan>  Q<tspan baseline-shift=\"sub\" font-size=\"17\">2</tspan>", 27, fill=c["blue"], italic=True)

    # 4. Movement-only policy action
    badge(1305, 100, "4", c["orange"])
    text(1335, 110, "Movement-only action", 31, "start", c["ink"], 500)
    rect(1285, 145, 460, 175, c["panel"], c["orange"], 2.5, 4)
    # Trajectory cell
    rect(1305, 170, 125, 120, c["orange_soft"], c["line"], 1.5, 3)
    drone(1354, 220, 0.65, c["blue"])
    path("M1320,270 Q1355,235 1408,190", c["orange"], 3.5, dash="7 6", marker="arrow-orange")
    text(1368, 286, "c<tspan baseline-shift=\"sub\" font-size=\"17\">k</tspan>, s<tspan baseline-shift=\"sub\" font-size=\"17\">k</tspan>, v<tspan baseline-shift=\"sub\" font-size=\"17\">k</tspan>", 24, fill=c["muted"], italic=True)
    # Action-size cell: offloading is solved after association.
    rect(1448, 170, 125, 120, c["orange_soft"], c["line"], 1.5, 3)
    text(1510, 220, "3K", 35, fill=c["orange"], weight=600)
    text(1510, 264, "movement", 17, fill=c["muted"], weight=500)
    # Entropy cell
    rect(1590, 170, 135, 120, c["violet_soft"], c["line"], 1.5, 3)
    path("M1612,257 C1625,207 1641,278 1656,218 C1671,174 1685,258 1703,202", c["violet"], 4)
    text(1658, 286, "&#945;<tspan baseline-shift=\"sub\" font-size=\"17\">H</tspan>", 25, fill=c["muted"], italic=True, family="Cambria Math, Times New Roman, serif")
    path("M1245,230 L1285,230", c["orange"], 4, marker="arrow-orange")
    text(1264, 211, "a<tspan baseline-shift=\"sub\" font-size=\"18\">t</tspan>", 29, fill=c["orange"], italic=True)

    # 5. Middle and inner layer, shown pictorially
    badge(1305, 355, "5", c["green"])
    text(1335, 365, "Feasible scheduling", 31, "start", c["ink"], 500)
    rect(1285, 390, 460, 160, c["green_soft"], c["green"], 2.5, 4)
    # Association mini-map
    rect(1305, 410, 195, 118, c["panel"], c["line"], 1.5, 3)
    drone(1364, 453, 0.52, c["blue"])
    drone(1444, 477, 0.52, c["blue"])
    for ux, uy, target_x, target_y in (
        (1325, 500, 1364, 453),
        (1368, 511, 1364, 453),
        (1420, 430, 1444, 477),
        (1477, 439, 1444, 477),
        (1482, 512, 1444, 477),
    ):
        user(ux, uy, True)
        line(ux, uy - 7, target_x, target_y, c["green"], 1.8, opacity=0.6)
    text(1402, 522, "S<tspan baseline-shift=\"sub\" font-size=\"16\">k</tspan>", 23, fill=c["muted"], italic=True)
    # Joint inner optimizer as proportional gauges
    rect(1520, 410, 205, 118, c["panel"], c["line"], 1.5, 3)
    for j, (label, widths, col) in enumerate(
        (
            ("CPU", (45, 75, 31), c["orange"]),
            ("BW", (61, 34, 72), c["blue"]),
        )
    ):
        yy = 438 + j * 47
        text(1535, yy + 11, label, 18, "start", c["muted"], 500)
        start = 1580
        for width in widths:
            rect(start, yy - 4, width, 16, col, None, 0, 2, 0.74)
            start += width + 5
    text(1622, 518, "x*   f*   b*", 22, fill=c["muted"], italic=True, family="Cambria Math, Times New Roman, serif")
    path("M1515,320 L1515,390", c["orange"], 4, marker="arrow-orange")
    path("M1285,470 C1195,470 1235,560 1105,560 C820,560 660,555 425,520", c["green"], 4, marker="arrow-green")
    text(905, 584, "r<tspan baseline-shift=\"sub\" font-size=\"18\">t</tspan>, s<tspan baseline-shift=\"sub\" font-size=\"18\">t+1</tspan>", 29, fill=c["green"], italic=True)

    # Learning loop: n-step rollout
    badge(75, 690, "6", c["gold"])
    text(105, 700, "n-step return", 31, "start", c["ink"], 500)
    rect(55, 730, 520, 300, c["panel"], c["gold"], 2.5, 4)
    # Three trajectory frames
    frame_xs = (82, 238, 394)
    rewards = ("r<tspan baseline-shift=\"sub\" font-size=\"16\">t</tspan>", "r<tspan baseline-shift=\"sub\" font-size=\"16\">t+1</tspan>", "r<tspan baseline-shift=\"sub\" font-size=\"16\">t+2</tspan>")
    for idx, fx in enumerate(frame_xs):
        rect(fx, 770, 130, 155, c["soft"], c["line"], 1.5, 3)
        circle(fx + 38, 815 + idx * 5, 22, c["orange_soft"], None, 0, 0.8)
        user(fx + 38, 815 + idx * 5, True, True)
        drone(fx + 91, 812 - idx * 4, 0.48, c["blue"])
        path(
            f"M{fx + 75},{850 - idx * 2} Q{fx + 93},{830 - idx * 4} {fx + 111},{846 + idx * 2}",
            c["orange"],
            2.5,
            marker="arrow-orange",
        )
        battery(fx + 18, 882, 0.72 - idx * 0.18, c["green"] if idx < 2 else c["red"], 0.72)
        text(fx + 96, 900, rewards[idx], 21, fill=c["muted"], italic=True)
        if idx < 2:
            line(fx + 132, 847, fx + 154, 847, c["gold"], 3, marker="arrow-gold")
    text(315, 970, "G<tspan baseline-shift=\"super\" font-size=\"17\">(n)</tspan><tspan baseline-shift=\"sub\" font-size=\"17\">t</tspan> = r<tspan baseline-shift=\"sub\" font-size=\"17\">t</tspan> + &#947;r<tspan baseline-shift=\"sub\" font-size=\"17\">t+1</tspan> + ...", 28, fill=c["gold"], italic=True, family="Cambria Math, Times New Roman, serif")
    path("M315,730 C315,675 275,665 275,620", c["gold"], 3, dash="8 7", marker="arrow-gold")

    # Dual-priority replay buffer
    badge(635, 690, "7", c["red"])
    text(665, 700, "Dual-priority replay", 31, "start", c["ink"], 500)
    rect(615, 730, 385, 300, c["red_soft"], c["red"], 2.5, 4)
    # Cylinder
    path("M660,800 C660,774 865,774 865,800 L865,930 C865,956 660,956 660,930 Z", c["line"], 2.5, c["panel"])
    path("M660,800 C660,826 865,826 865,800", c["line"], 2)
    path("M660,855 C660,881 865,881 865,855", c["muted"], 1.5, opacity=0.45)
    path("M660,910 C660,936 865,936 865,910", c["muted"], 1.5, opacity=0.45)
    # Two priority streams
    line(905, 803, 905, 924, c["red"], 9)
    line(947, 835, 947, 924, c["gold"], 9)
    text(905, 788, "|&#948;|", 23, fill=c["red"], italic=True, family="Cambria Math, Times New Roman, serif")
    text(947, 788, "|r-r&#772;|", 23, fill=c["gold"], italic=True, family="Cambria Math, Times New Roman, serif")
    text(807, 987, "p<tspan baseline-shift=\"sub\" font-size=\"16\">i</tspan> = p<tspan baseline-shift=\"super\" font-size=\"16\">TD</tspan><tspan baseline-shift=\"sub\" font-size=\"16\">i</tspan> + &#955;<tspan baseline-shift=\"sub\" font-size=\"16\">sur</tspan> p<tspan baseline-shift=\"super\" font-size=\"16\">sur</tspan><tspan baseline-shift=\"sub\" font-size=\"16\">i</tspan>", 25, fill=c["muted"], italic=True, family="Cambria Math, Times New Roman, serif")
    line(575, 875, 615, 875, c["red"], 4, marker="arrow-red")

    # Importance-sampled mini-batch
    badge(1030, 690, "8", c["red"])
    text(1060, 700, "IS mini-batch", 31, "start", c["ink"], 500)
    for offset in (24, 12, 0):
        rect(1040 + offset, 790 - offset, 180, 155, c["panel"], c["line"], 1.8, 3)
    for j, val in enumerate((0.32, 0.78, 0.55, 0.91)):
        yy = 810 + j * 30
        rect(1075, yy, 105 * val, 12, c["red"] if j in (1, 3) else c["gold"], None, 0, 2, 0.78)
        text(1192, yy + 12, f"w{j + 1}", 18, "start", c["muted"], italic=True)
    line(1000, 875, 1040, 875, c["red"], 4, marker="arrow-red")

    # Twin critic update and target
    badge(1250, 690, "9", c["blue"])
    text(1280, 700, "Value update", 31, "start", c["ink"], 500)
    rect(1260, 745, 245, 235, c["blue_soft"], c["blue"], 2.5, 4)
    neural_net(1290, 785, 105, 65, c["blue"], (3, 4, 2))
    neural_net(1290, 875, 105, 65, c["blue"], (3, 4, 2))
    text(1445, 817, "Q<tspan baseline-shift=\"sub\" font-size=\"17\">1</tspan>", 27, fill=c["blue"], italic=True)
    text(1445, 907, "Q<tspan baseline-shift=\"sub\" font-size=\"17\">2</tspan>", 27, fill=c["blue"], italic=True)
    text(1383, 963, "min(Q<tspan baseline-shift=\"sub\" font-size=\"15\">1</tspan>,Q<tspan baseline-shift=\"sub\" font-size=\"15\">2</tspan>)", 23, fill=c["muted"], italic=True)
    line(1224, 875, 1260, 875, c["blue"], 4, marker="arrow-blue")

    # Actor, entropy and target updates
    badge(1540, 690, "10", c["violet"])
    text(1570, 700, "Policy update", 31, "start", c["ink"], 500)
    rect(1540, 745, 205, 235, c["violet_soft"], c["violet"], 2.5, 4)
    neural_net(1570, 782, 95, 65, c["violet"], (3, 4, 2))
    text(1690, 817, "&#960;<tspan baseline-shift=\"sub\" font-size=\"16\">&#952;</tspan>", 28, fill=c["violet"], italic=True, family="Cambria Math, Times New Roman, serif")
    path("M1580,895 C1592,850 1607,927 1621,868 C1638,824 1650,913 1668,858", c["violet"], 3.5)
    text(1690, 884, "&#945;<tspan baseline-shift=\"sub\" font-size=\"16\">H</tspan>", 27, fill=c["violet"], italic=True, family="Cambria Math, Times New Roman, serif")
    circle(1618, 944, 22, "none", c["green"], 4)
    circle(1618, 944, 12, "none", c["green"], 3)
    text(1690, 953, "Q&#8242;", 27, fill=c["green"], italic=True, family="Cambria Math, Times New Roman, serif")
    line(1505, 875, 1540, 875, c["violet"], 4, marker="arrow-violet")
    # Return updates to top networks
    path("M1640,745 C1640,690 1705,690 1705,600 L1705,580 C1705,570 1660,570 1605,570 C1350,570 1220,555 1160,530", c["violet"], 3.5, dash="9 7", marker="arrow-violet")

    # Tiny legend emphasizing that arrows are the explanation, not prose.
    circle(80, 1072, 7, c["blue"])
    text(98, 1080, "interaction", 20, "start", c["muted"])
    circle(220, 1072, 7, c["red"])
    text(238, 1080, "priority sampling", 20, "start", c["muted"])
    circle(425, 1072, 7, c["green"])
    text(443, 1080, "parameter feedback", 20, "start", c["muted"])
    text(1745, 1080, "DARS", 27, "end", c["muted"], 500)

    add("</svg>")
    return "\n".join(out)


def make_standalone(svg: str) -> str:
    return (
        "<!doctype html>\n"
        "<html><head><meta charset=\"utf-8\">\n"
        "<style>\n"
        "@page { size: 7.2in 4.48in; margin: 0; }\n"
        "html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; }\n"
        "body { background: #ffffff; }\n"
        "svg { display: block; width: 100vw; height: 100vh; }\n"
        "</style></head><body>\n"
        f"{svg}\n"
        "</body></html>\n"
    )


def make_fragment(svg: str) -> str:
    return (
        '<div id="dars-algorithm-flow" style="width:100%;color:var(--foreground)">\n'
        f"{svg}\n"
        "</div>\n"
        "<style>\n"
        "#dars-algorithm-flow svg { display:block; width:100%; height:auto; }\n"
        "</style>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--viz-fragment", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    figures = repo / "results" / "figures"
    pdf_output = repo / "output" / "pdf"
    render_dir = repo / "tmp" / "dars-flowchart"
    figures.mkdir(parents=True, exist_ok=True)
    pdf_output.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    args.viz_fragment.parent.mkdir(parents=True, exist_ok=True)

    paper_svg = build_svg(PAPER, include_background=True)
    theme_svg = build_svg(THEME, include_background=False)

    svg_path = figures / "dars_algorithm_flowchart_pictorial.svg"
    svg_path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + paper_svg, encoding="utf-8")

    standalone_path = render_dir / "dars_algorithm_flowchart_pictorial.html"
    standalone_path.write_text(make_standalone(paper_svg), encoding="utf-8")

    args.viz_fragment.write_text(make_fragment(theme_svg), encoding="utf-8")

    print(svg_path)
    print(standalone_path)
    print(pdf_output / "dars_algorithm_flowchart_pictorial.pdf")


if __name__ == "__main__":
    main()
