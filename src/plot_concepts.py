"""Render publication-ready concept diagrams from data/concept_figures.json.

Run from the repository root with:
    .venv/Scripts/python.exe src/plot_concepts.py

Each figure is exported as a high-resolution PNG and as editable SVG/PDF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "concept_figures.json"
FIGURE_DIR = ROOT / "figures"

PALETTE = {
    "blue": {"edge": "#477AA5", "fill": "#EAF1F7", "text": "#244B69"},
    "teal": {"edge": "#418982", "fill": "#EAF4F2", "text": "#235A56"},
    "purple": {"edge": "#7A6FA7", "fill": "#F0EEF7", "text": "#514873"},
    "gray": {"edge": "#77838E", "fill": "#F1F3F5", "text": "#48545E"},
    "policy": {"edge": "#5682A5", "fill": "#EDF3F8", "text": "#284C67"},
    "judge": {"edge": "#558B82", "fill": "#EDF5F2", "text": "#2A5D55"},
    "update": {"edge": "#46867C", "fill": "#E8F2F0", "text": "#275A53"},
    "data": {"edge": "#7893A8", "fill": "#F0F4F7", "text": "#3C5568"},
    "loss": {"edge": "#8176A8", "fill": "#F2F0F8", "text": "#534A75"},
    "mask": {"edge": "#909AA3", "fill": "#F2F4F5", "text": "#4E5962"},
}


def load_content() -> dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def configure_fonts() -> str:
    installed = {font.name for font in fm.fontManager.ttflist}
    candidates = ["Noto Sans SC", "Microsoft YaHei", "DengXian", "SimHei", "sans-serif"]
    selected = next((name for name in candidates if name in installed), "sans-serif")
    plt.rcParams.update(
        {
            "font.family": selected,
            "font.sans-serif": [selected, "Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    return selected


def add_rounded_box(
    ax: plt.Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str = "blue",
    *,
    fontsize: float = 9.0,
    title: str | None = None,
    detail: str | None = None,
    linewidth: float = 1.0,
    rounding: float = 0.12,
    zorder: float = 3,
    weight: str = "normal",
) -> FancyBboxPatch:
    style = PALETTE[color]
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.035,rounding_size={rounding}",
        facecolor=style["fill"],
        edgecolor=style["edge"],
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if title is None:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            color=style["text"],
            fontsize=fontsize,
            fontweight=weight,
            linespacing=1.23,
            zorder=zorder + 1,
        )
    else:
        if detail is None:
            title_y = y
            detail_y = y
        else:
            title_y = y + min(0.13, height * 0.18)
            detail_y = y - min(0.16, height * 0.24)
        ax.text(
            x,
            title_y,
            title,
            ha="center",
            va="center",
            color=style["text"],
            fontsize=fontsize,
            fontweight="semibold",
            linespacing=1.2,
            zorder=zorder + 1,
        )
        if detail is not None:
            ax.text(
                x,
                detail_y,
                detail,
                ha="center",
                va="center",
                color="#46535E",
                fontsize=max(6.7, fontsize - 1.45),
                linespacing=1.15,
                zorder=zorder + 1,
            )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#687986",
    linewidth: float = 1.2,
    linestyle: str = "solid",
    zorder: float = 1,
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def add_polyline_arrow(
    ax: plt.Axes,
    vertices: list[tuple[float, float]],
    *,
    color: str = "#687986",
    linewidth: float = 1.2,
    linestyle: str = "solid",
    zorder: float = 1,
) -> None:
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 1)
    ax.add_patch(
        FancyArrowPatch(
            path=MplPath(vertices, codes),
            arrowstyle="-|>",
            mutation_scale=9,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            FIGURE_DIR / f"{stem}.{extension}",
            dpi=160,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.12,
        )
    plt.close(fig)


def draw_timeline(content: dict[str, Any]) -> None:
    section = content["timeline"]
    fig, ax = plt.subplots(figsize=(17.4, 9.4), layout="constrained")
    ax.set_xlim(0, 17.4)
    ax.set_ylim(0.45, 9.45)
    ax.axis("off")

    ax.text(
        0.12,
        9.17,
        "后训练方法谱系：分支扩展与训练范式共存",
        fontsize=19.5,
        fontweight="bold",
        color="#253746",
        ha="left",
        va="center",
    )
    ax.text(
        0.14,
        8.80,
        "时间标示代表性研究节点；箭头说明方法或数据关系，不表示后一方法全面取代前一方法。",
        fontsize=10.0,
        color="#5C6A75",
        ha="left",
        va="center",
    )

    x_centers = {year: 3.0 + i * 3.1 for i, year in enumerate(section["years"])}
    lane_centers = {index: 7.35 - index * 1.5 for index in range(len(section["lanes"]))}
    lane_x0, lane_x1 = 1.72, 17.20
    for i, lane in enumerate(section["lanes"]):
        y = lane_centers[i]
        ax.add_patch(
            FancyBboxPatch(
                (lane_x0, y - 0.62),
                lane_x1 - lane_x0,
                1.24,
                boxstyle="round,pad=0.01,rounding_size=0.08",
                facecolor="#FAFBFC" if i % 2 == 0 else "#FFFFFF",
                edgecolor="#E7EBEE",
                linewidth=0.65,
                zorder=0,
            )
        )
        ax.text(
            0.12,
            y,
            lane,
            ha="left",
            va="center",
            fontsize=8.6,
            color="#43535F",
            fontweight="semibold",
            linespacing=1.35,
        )

    for i, year in enumerate(section["years"]):
        x = x_centers[year]
        if i % 2:
            ax.axvspan(x - 1.52, x + 1.52, color="#F8FAFB", zorder=-1)
        ax.text(
            x,
            8.45,
            str(year),
            fontsize=12,
            fontweight="bold",
            color="#3F5362",
            ha="center",
            va="center",
        )
        ax.plot([x, x], [0.68, 8.18], color="#EEF1F3", linewidth=0.7, zorder=0)

    node_index: dict[tuple[int, int, float], dict[str, Any]] = {}
    for node in section["nodes"]:
        node_index[(node["year"], node["lane"], node["offset"])] = node

    card_w = 2.34
    default_h = 0.57
    # Relationship arrows are drawn before nodes so that they remain behind labels.
    for edge in section["edges"]:
        start_key = (edge["from"][0], edge["from"][1], edge["from"][2])
        end_key = (edge["to"][0], edge["to"][1], edge["to"][2])
        start_node, end_node = node_index[start_key], node_index[end_key]
        sx, ex = x_centers[start_node["year"]], x_centers[end_node["year"]]
        sy = lane_centers[start_node["lane"]] + start_node["offset"]
        ey = lane_centers[end_node["lane"]] + end_node["offset"]
        if abs(ex - sx) < 0.01:
            ydir = 1 if ey < sy else -1
            p0 = (sx, sy + ydir * default_h / 2)
            p1 = (ex, ey - ydir * default_h / 2)
        else:
            direction = 1 if ex > sx else -1
            p0 = (sx + direction * card_w / 2, sy)
            p1 = (ex - direction * card_w / 2, ey)
        edge_color = "#84939D" if "style" not in edge else "#8C7777"
        edge_style = "dashed" if edge.get("style") == "dashed" else "solid"
        add_arrow(ax, p0, p1, color=edge_color, linewidth=1.05, linestyle=edge_style)
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        if "label_xy" in edge:
            mx, my = edge["label_xy"]
        elif abs(ex - sx) < 0.01:
            mx += 0.88
        elif abs(ey - sy) < 0.15:
            my += 0.25
        ax.text(
            mx,
            my,
            edge["label"],
            ha="center",
            va="center",
            fontsize=7.0,
            color="#53616B",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.96},
            zorder=2,
        )

    for node in section["nodes"]:
        x = x_centers[node["year"]]
        y = lane_centers[node["lane"]] + node["offset"]
        siblings = sum(1 for item in section["nodes"] if item["year"] == node["year"] and item["lane"] == node["lane"])
        h = 0.50 if siblings > 1 else default_h
        add_rounded_box(
            ax,
            (x, y),
            card_w,
            h,
            "",
            color=node["color"],
            title=node["title"],
            detail=node["detail"],
            fontsize=8.25 if siblings > 1 else 8.7,
            linewidth=0.9,
            rounding=0.09,
            zorder=4,
        )

    ax.text(
        17.0,
        0.54,
        "方法与数据分支并行；灰色标注研究中的目标失配风险",
        fontsize=7.8,
        color="#697680",
        ha="right",
        va="center",
    )
    save_figure(fig, "timeline")


def draw_loop_panel(ax: plt.Axes, item: dict[str, Any]) -> None:
    center_x = ax.get_xlim()[1] / 2
    add_rounded_box(
        ax,
        (center_x, 8.63),
        3.70,
        0.48,
        "",
        color=item["color"],
        title=item["name"],
        detail=None,
        fontsize=12.0,
        linewidth=1.0,
        rounding=0.12,
    )
    ax.text(
        center_x,
        8.24,
        item["subtitle"],
        ha="center",
        va="center",
        fontsize=8.2,
        color="#5F6C76",
    )
    steps = item["steps"]
    spacing = 1.07 if len(steps) >= 6 else 1.42
    start_y = 7.62
    box_w, box_h = 3.22, 0.66
    positions: list[float] = []
    for index, step in enumerate(steps):
        y = start_y - index * spacing
        positions.append(y)
        kind = step["kind"]
        color = {
            "data": "data",
            "policy": "policy",
            "judge": "judge",
            "process": "gray",
            "loss": "loss",
            "update": "update",
        }[kind]
        add_rounded_box(
            ax,
            (center_x, y),
            box_w,
            box_h,
            step["text"],
            color=color,
            fontsize=8.3,
            rounding=0.09,
            linewidth=0.9,
        )
        if index < len(steps) - 1:
            add_arrow(
                ax,
                (center_x, y - box_h / 2 - 0.015),
                (center_x, y - spacing + box_h / 2 + 0.015),
                color="#84929B",
                linewidth=1.05,
                zorder=2,
            )

    if item["loop"]:
        y0, y1 = positions[-1], positions[1]
        right = center_x + 1.82
        path = MplPath(
            [
                (center_x + box_w / 2, y0),
                (right, y0),
                (right, y1),
                (center_x + box_w / 2 + 0.05, y1),
            ],
            [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
        )
        ax.add_patch(
            FancyArrowPatch(
                path=path,
                arrowstyle="-|>",
                mutation_scale=9,
                color="#84929B",
                linewidth=1.0,
                linestyle=(0, (3, 2)),
                zorder=1,
            )
        )
        ax.text(
            right - 0.02,
            (y0 + y1) / 2,
            "刷新采样",
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.0,
            color="#64727D",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.3},
            zorder=2,
        )

    if item.get("legend"):
        ax.text(
            center_x,
            0.57,
            "\n".join(item["legend"]),
            ha="center",
            va="bottom",
            fontsize=7.1,
            color="#4E5C66",
            linespacing=1.45,
        )
    elif not item["loop"]:
        note = "reference = 固定 SFT 策略；不从模型生成新回答" if item["name"] == "离线 DPO" else "样本固定；训练仅拟合目标回答"
        ax.text(center_x, 0.7, note, ha="center", va="center", fontsize=7.2, color="#65727B")


def draw_loops(content: dict[str, Any]) -> None:
    section = content["loops"]
    fig = plt.figure(figsize=(16.8, 9.7))
    fig.text(
        0.02,
        0.973,
        "训练回路比较：数据是否更新，决定采样与反馈位置",
        fontsize=19,
        fontweight="bold",
        color="#253746",
        ha="left",
        va="top",
    )
    fig.text(
        0.02,
        0.929,
        "固定示范 / 偏好对以静态样本训练；在线 DPO 与 GRPO 将策略采样和外部反馈纳入更新闭环。",
        fontsize=9.8,
        color="#5C6A75",
        ha="left",
        va="top",
    )
    grid = fig.add_gridspec(1, 4, left=0.015, right=0.985, bottom=0.025, top=0.86, wspace=0.10)
    for index, item in enumerate(section["columns"]):
        ax = fig.add_subplot(grid[0, index])
        ax.set_xlim(0, 4.0)
        ax.set_ylim(0.3, 9.0)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.05, 0.32),
                3.90,
                8.54,
                boxstyle="round,pad=0.01,rounding_size=0.08",
                facecolor="#FFFFFF",
                edgecolor="#E3E8EC",
                linewidth=0.85,
                zorder=0,
            )
        )
        draw_loop_panel(ax, item)
    save_figure(fig, "loops")


def draw_implementation(content: dict[str, Any]) -> None:
    section = content["implementation"]
    fig, ax = plt.subplots(figsize=(18.0, 9.6), layout="constrained")
    ax.set_xlim(0, 18)
    ax.set_ylim(0.1, 9.7)
    ax.axis("off")
    ax.text(
        0.15,
        9.38,
        "从训练语义到本地实现对象",
        fontsize=19.5,
        fontweight="bold",
        color="#253746",
        ha="left",
        va="center",
    )
    ax.text(
        0.17,
        9.01,
        "共享的 prompt、rollout 与 token mask 向奖励和概率计算分流；DPO 与 GRPO 使用不同的优化统计量。",
        fontsize=9.8,
        color="#5C6A75",
        ha="left",
        va="center",
    )

    top_xs = [3.0, 9.0, 15.0]
    top_y = 7.95
    top_w, top_h = 4.4, 1.03
    for i, item in enumerate(section["common"]):
        color = {"data": "data", "policy": "policy", "mask": "mask"}[item["kind"]]
        add_rounded_box(
            ax,
            (top_xs[i], top_y),
            top_w,
            top_h,
            "",
            color=color,
            title=item["title"],
            detail=item["detail"],
            fontsize=10.0,
            linewidth=1.0,
            rounding=0.12,
        )
        if i < len(top_xs) - 1:
            add_arrow(ax, (top_xs[i] + top_w / 2, top_y), (top_xs[i + 1] - top_w / 2, top_y), color="#74838E", linewidth=1.2)

    # The completion stream fans out to reward evaluation and log-probability roles.
    reward = section["reward"]
    add_polyline_arrow(
        ax,
        [(12.80, top_y - top_h / 2), (10.7, 7.22), (9.15, 6.95), (8.70, 6.34)],
        color="#82909A",
        linewidth=1.05,
    )
    add_arrow(ax, (15.00, top_y - top_h / 2), (15.00, 6.79), color="#82909A", linewidth=1.05)

    add_rounded_box(
        ax,
        (4.8, 5.75),
        7.8,
        1.18,
        "",
        color="judge",
        title=reward["title"],
        detail=reward["detail"],
        fontsize=10.0,
        linewidth=1.0,
        rounding=0.12,
    )
    # Distinct probability roles: only current log p carries gradients.
    ax.add_patch(
        FancyBboxPatch(
            (9.1, 5.12),
            8.3,
            1.62,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            facecolor="#FBFCFD",
            edgecolor="#DDE4E8",
            linewidth=0.85,
            zorder=0,
        )
    )
    ax.text(13.25, 6.51, "策略概率的三种角色", ha="center", va="center", fontsize=9.2, fontweight="semibold", color="#475762")
    logp_xs = [10.48, 13.25, 16.02]
    for item, x in zip(section["logps"], logp_xs):
        add_rounded_box(
            ax,
            (x, 5.75),
            2.45,
            1.0,
            "",
            color=item["color"],
            title=item["title"],
            detail=item["detail"],
            fontsize=8.7,
            linewidth=0.85,
            rounding=0.08,
        )

    # Two objective cards preserve the implementation-specific distinction.
    objective_y = 3.17
    dpo, grpo = section["objectives"]
    add_arrow(ax, (4.8, 5.12), (4.8, 4.07), color="#82909A", linewidth=1.1)
    add_arrow(ax, (13.25, 5.12), (13.25, 4.07), color="#82909A", linewidth=1.1)
    add_arrow(ax, (8.70, 5.16), (9.07, 4.07), color=PALETTE["judge"]["edge"], linewidth=1.1)
    add_rounded_box(
        ax,
        (4.8, objective_y),
        8.35,
        1.80,
        "",
        color=dpo["color"],
        title=dpo["title"],
        detail=dpo["detail"],
        fontsize=10.1,
        linewidth=1.05,
        rounding=0.13,
    )
    add_rounded_box(
        ax,
        (13.25, objective_y),
        8.35,
        1.80,
        "",
        color=grpo["color"],
        title=grpo["title"],
        detail=grpo["detail"],
        fontsize=9.6,
        linewidth=1.05,
        rounding=0.13,
    )
    add_arrow(ax, (4.8, 2.26), (8.5, 1.18), color=PALETTE["blue"]["edge"], linewidth=1.25)
    add_arrow(ax, (13.25, 2.26), (9.5, 1.18), color=PALETTE["teal"]["edge"], linewidth=1.25)
    add_rounded_box(
        ax,
        (9.0, 0.75),
        7.2,
        0.74,
        "",
        color="update",
        title="优化器更新 actor 参数 θ",
        detail="下一轮 rollout 使用更新后的策略",
        fontsize=10.0,
        linewidth=1.0,
        rounding=0.11,
    )
    save_figure(fig, "implementation")


def main() -> None:
    configure_fonts()
    content = load_content()
    draw_timeline(content)
    draw_loops(content)
    draw_implementation(content)


if __name__ == "__main__":
    main()
