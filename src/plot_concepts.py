"""Render the reusable concept figures for the post-training survey.

Run from this directory with the experiment environment:
    python src/plot_concepts.py

The script reads data/concept_figures.json and writes four PNG, SVG and PDF
figures. The diagrams contain literature-level concepts only; no experiment-
specific implementation objects are included.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "concept_figures.json"
FIGURE_DIR = ROOT / "figures"

PALETTE = {
    "blue": {"edge": "#477AA5", "fill": "#EAF1F7", "text": "#244B69"},
    "teal": {"edge": "#418982", "fill": "#EAF4F2", "text": "#235A56"},
    "purple": {"edge": "#7A6FA7", "fill": "#F0EEF7", "text": "#514873"},
    "gray": {"edge": "#77838E", "fill": "#F1F3F5", "text": "#48545E"},
    "ochre": {"edge": "#A98750", "fill": "#F7F2E9", "text": "#66522F"},
    "white": {"edge": "#DDE4E8", "fill": "#FFFFFF", "text": "#43535F"},
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


def add_box(
    ax: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    title: str,
    detail: str = "",
    tone: str = "blue",
    *,
    title_size: float = 10.0,
    detail_size: float | None = None,
    rounding: float = 0.10,
    zorder: float = 3,
) -> FancyBboxPatch:
    style = PALETTE[tone]
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.035,rounding_size={rounding}",
        facecolor=style["fill"],
        edgecolor=style["edge"],
        linewidth=0.95,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if detail:
        ax.text(
            x,
            y + min(0.17, height * 0.18),
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="semibold",
            color=style["text"],
            linespacing=1.12,
            zorder=zorder + 1,
        )
        ax.text(
            x,
            y - min(0.17, height * 0.20),
            detail,
            ha="center",
            va="center",
            fontsize=detail_size if detail_size is not None else max(7.2, title_size - 1.4),
            color="#46535E",
            linespacing=1.15,
            zorder=zorder + 1,
        )
    else:
        ax.text(
            x,
            y,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="semibold",
            color=style["text"],
            linespacing=1.15,
            zorder=zorder + 1,
        )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#71818D",
    width: float = 1.2,
    dashed: bool = False,
    zorder: float = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            color=color,
            linewidth=width,
            linestyle=(0, (3, 2)) if dashed else "solid",
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def add_polyline_arrow(
    ax: plt.Axes,
    vertices: list[tuple[float, float]],
    *,
    color: str = "#71818D",
    width: float = 1.2,
    dashed: bool = False,
) -> None:
    path = MplPath(
        vertices,
        [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 1),
    )
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=10,
            color=color,
            linewidth=width,
            linestyle=(0, (3, 2)) if dashed else "solid",
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def add_return_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    upper_y: float,
    end_x: float,
    color: str = "#83919A",
) -> None:
    right_x = 3.89
    path = MplPath(
        [start, (right_x, start[1]), (right_x, upper_y), (end_x, upper_y)],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
    )
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=9,
            color=color,
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            zorder=1,
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
            pad_inches=0.13,
        )
    plt.close(fig)


def draw_timeline(content: dict[str, Any]) -> None:
    section = content["timeline"]
    fig, ax = plt.subplots(figsize=(18.2, 8.55))
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0.35, 8.95)
    ax.axis("off")
    ax.text(0.15, 8.64, section["title"], fontsize=19.5, fontweight="bold", color="#253746", va="center")
    ax.text(0.17, 8.25, section["subtitle"], fontsize=10.0, color="#5C6A75", va="center")

    years = section["years"]
    x_centers = {year: 2.95 + i * 3.18 for i, year in enumerate(years)}
    lane_y = {index: 6.77 - index * 1.53 for index in range(len(section["lanes"]))}
    for index, year in enumerate(years):
        x = x_centers[year]
        if index % 2:
            ax.axvspan(x - 1.55, x + 1.55, color="#F8FAFB", zorder=-2)
        ax.text(x, 7.76, str(year), ha="center", va="center", fontsize=12.0, fontweight="bold", color="#405462")
        ax.plot([x, x], [0.74, 7.50], color="#E8EDF0", linewidth=0.75, zorder=-1)

    for index, lane in enumerate(section["lanes"]):
        y = lane_y[index]
        ax.add_patch(
            FancyBboxPatch(
                (1.60, y - 0.59),
                16.25,
                1.18,
                boxstyle="round,pad=0.01,rounding_size=0.08",
                facecolor="#FBFCFD" if index % 2 == 0 else "#FFFFFF",
                edgecolor="#E6EBEE",
                linewidth=0.7,
                zorder=-1,
            )
        )
        ax.text(0.14, y, lane, ha="left", va="center", fontsize=8.9, fontweight="semibold", color="#43535F", linespacing=1.3)

    nodes = section["nodes"]
    index_by_id = {item["id"]: item for item in nodes}
    card_w = 2.40
    edge_points: dict[str, tuple[float, float]] = {}
    for edge in section["edges"]:
        start_node = index_by_id[edge["from"]]
        end_node = index_by_id[edge["to"]]
        sx, ex = x_centers[start_node["year"]], x_centers[end_node["year"]]
        sy = lane_y[start_node["lane"]] + start_node.get("offset", 0.0)
        ey = lane_y[end_node["lane"]] + end_node.get("offset", 0.0)
        if abs(sx - ex) < 0.01:
            start, end = (sx, sy - 0.37), (ex, ey + 0.37)
            label_xy = (sx + 1.05, (sy + ey) / 2)
        else:
            direction = 1 if ex > sx else -1
            start, end = (sx + direction * card_w / 2, sy), (ex - direction * card_w / 2, ey)
            label_xy = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.22)
        add_arrow(ax, start, end, width=1.05, color="#82919A", dashed=edge.get("dashed", False))
        edge_points[edge["id"]] = label_xy
        ax.text(
            label_xy[0],
            label_xy[1],
            edge["label"],
            ha="center",
            va="center",
            fontsize=7.6,
            color="#53616B",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.97},
            zorder=2,
        )

    for node in nodes:
        year_group = [item for item in nodes if item["year"] == node["year"] and item["lane"] == node["lane"]]
        height = 0.54 if len(year_group) > 1 else 0.73
        center = (x_centers[node["year"]], lane_y[node["lane"]] + node.get("offset", 0.0))
        add_box(
            ax,
            center,
            (card_w, height),
            node["title"],
            node["detail"],
            node["tone"],
            title_size=8.6 if len(year_group) > 1 else 9.0,
            detail_size=7.55 if len(year_group) > 1 else 7.9,
            rounding=0.08,
        )

    ax.text(17.88, 0.48, section["footer"], ha="right", va="center", fontsize=8.2, color="#65727B")
    save_figure(fig, "timeline")


def draw_loop_panel(ax: plt.Axes, item: dict[str, Any]) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.06, 0.30),
            4.16,
            7.65,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            facecolor="#FFFFFF",
            edgecolor="#E1E7EB",
            linewidth=0.85,
            zorder=0,
        )
    )
    add_box(ax, (2.14, 7.57), (3.72, 0.47), item["name"], tone=item["tone"], title_size=11.5, rounding=0.10)
    ax.text(2.14, 7.16, item["subtitle"], ha="center", va="center", fontsize=8.4, color="#5F6C76")

    steps = item["steps"]
    if len(steps) == 3:
        y_positions = [6.08, 4.44, 2.80]
    elif len(steps) == 4:
        y_positions = [6.30, 5.00, 3.70, 2.40]
    else:
        y_positions = [6.28, 5.20, 4.12, 3.04, 1.96]
    box_size = (3.48, 0.76)
    tone_map = {"data": "blue", "policy": "blue", "judge": "teal", "process": "gray", "loss": "purple", "update": "teal"}
    for i, (step, y) in enumerate(zip(steps, y_positions)):
        add_box(
            ax,
            (2.12, y),
            box_size,
            step["title"],
            step.get("detail", ""),
            tone_map[step["kind"]],
            title_size=8.9,
            detail_size=7.8,
            rounding=0.075,
        )
        if i < len(steps) - 1:
            add_arrow(ax, (2.12, y - box_size[1] / 2 - 0.025), (2.12, y_positions[i + 1] + box_size[1] / 2 + 0.025), width=1.0)
    if item.get("loop"):
        candidate_y = y_positions[1]
        update_y = y_positions[-1]
        add_return_arrow(ax, (3.86, update_y), candidate_y, 3.86)
    if item.get("note"):
        ax.text(2.14, 0.67, item["note"], ha="center", va="center", fontsize=7.4, color="#56636D", linespacing=1.35)


def draw_loops(content: dict[str, Any]) -> None:
    section = content["loops"]
    fig = plt.figure(figsize=(18.2, 8.65))
    fig.text(0.018, 0.967, section["title"], fontsize=19.2, fontweight="bold", color="#253746", ha="left", va="top")
    fig.text(0.020, 0.921, section["subtitle"], fontsize=10.0, color="#5C6A75", ha="left", va="top")
    grid = fig.add_gridspec(1, 4, left=0.008, right=0.992, bottom=0.025, top=0.885, wspace=0.045)
    for index, item in enumerate(section["columns"]):
        ax = fig.add_subplot(grid[0, index])
        ax.set_xlim(0, 4.28)
        ax.set_ylim(0.2, 8.15)
        ax.axis("off")
        draw_loop_panel(ax, item)
    fig.text(0.986, 0.012, section["footer"], fontsize=8.0, color="#65727B", ha="right", va="bottom")
    save_figure(fig, "loops")


def draw_reward_layers(content: dict[str, Any]) -> None:
    section = content["reward_layers"]
    fig, ax = plt.subplots(figsize=(18.0, 8.2))
    ax.set_xlim(0, 18.0)
    ax.set_ylim(0.25, 8.45)
    ax.axis("off")
    ax.text(0.15, 8.14, section["title"], fontsize=19.3, fontweight="bold", color="#253746", va="center")
    ax.text(0.17, 7.78, section["subtitle"], fontsize=10.0, color="#5C6A75", va="center")

    header_y = 7.23
    col_x = {"unit": 5.12, "timing": 10.0, "blind": 15.0}
    ax.text(0.25, header_y, "反馈形式", fontsize=9.5, fontweight="semibold", color="#43535F", va="center")
    ax.text(col_x["unit"], header_y, "监督单元", fontsize=9.5, fontweight="semibold", color="#43535F", ha="center", va="center")
    ax.text(col_x["timing"], header_y, "信号进入训练的位置", fontsize=9.5, fontweight="semibold", color="#43535F", ha="center", va="center")
    ax.text(col_x["blind"], header_y, "需要留意的盲区", fontsize=9.5, fontweight="semibold", color="#43535F", ha="center", va="center")
    ax.plot([0.22, 17.78], [6.96, 6.96], color="#DFE5E9", linewidth=0.9)

    row_y = [6.31, 5.19, 4.07, 2.95, 1.83]
    tones = ["blue", "purple", "teal", "ochre", "gray"]
    for index, (row, y, tone) in enumerate(zip(section["rows"], row_y, tones)):
        if index % 2 == 0:
            ax.add_patch(
                FancyBboxPatch(
                    (0.20, y - 0.49),
                    17.58,
                    0.98,
                    boxstyle="round,pad=0.01,rounding_size=0.045",
                    facecolor="#FBFCFD",
                    edgecolor="none",
                    zorder=-1,
                )
            )
        add_box(ax, (1.52, y), (2.55, 0.70), row["label"], tone=tone, title_size=9.2, rounding=0.075)
        for key in ("unit", "timing", "blind"):
            ax.text(
                col_x[key],
                y,
                row[key],
                ha="center",
                va="center",
                fontsize=8.7,
                color="#3F4F5A",
                linespacing=1.28,
                wrap=True,
            )
    ax.text(17.80, 0.61, section["footer"], ha="right", va="center", fontsize=8.0, color="#65727B")
    save_figure(fig, "reward_layers")


def draw_r1_pipeline(content: dict[str, Any]) -> None:
    section = content["r1_pipeline"]
    fig, ax = plt.subplots(figsize=(19.2, 10.1))
    ax.set_xlim(0, 19.2)
    ax.set_ylim(0.25, 10.1)
    ax.axis("off")
    ax.text(0.16, 9.82, section["title"], fontsize=19.3, fontweight="bold", color="#253746", va="center")
    ax.text(0.18, 9.44, section["subtitle"], fontsize=9.8, color="#5C6A75", va="center")

    # The direct-RL R1-Zero recipe and the multi-stage R1 recipe are shown as
    # separate branches from their shared base-model family.
    ax.add_patch(
        FancyBboxPatch(
            (0.18, 6.83),
            18.84,
            2.22,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            facecolor="#FBFCFD",
            edgecolor="#E0E6EA",
            linewidth=0.85,
            zorder=0,
        )
    )
    ax.text(0.42, 8.78, section["zero_label"], fontsize=10.2, fontweight="semibold", color="#3D5F7A", va="center")
    zero_nodes = section["zero_nodes"]
    zero_centers = [2.70, 9.60, 16.45]
    zero_y = 7.82
    zero_size = (3.55, 0.96)
    for i, (item, x) in enumerate(zip(zero_nodes, zero_centers)):
        add_box(ax, (x, zero_y), zero_size, item["title"], item["detail"], item["tone"], title_size=9.8, detail_size=8.3)
        if i < len(zero_nodes) - 1:
            add_arrow(ax, (x + zero_size[0] / 2 + 0.04, zero_y), (zero_centers[i + 1] - zero_size[0] / 2 - 0.04, zero_y), width=1.25)

    ax.add_patch(
        FancyBboxPatch(
            (0.18, 3.20),
            18.84,
            3.18,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            facecolor="#FFFFFF",
            edgecolor="#E0E6EA",
            linewidth=0.85,
            zorder=0,
        )
    )
    ax.text(0.42, 6.06, section["r1_label"], fontsize=10.2, fontweight="semibold", color="#3D5F7A", va="center")
    main_nodes = section["main_nodes"]
    main_centers = [1.55 + i * 2.70 for i in range(len(main_nodes))]
    main_y = 4.73
    main_size = (2.24, 1.12)
    for item, x in zip(main_nodes, main_centers):
        add_box(
            ax,
            (x, main_y),
            main_size,
            item["title"],
            item["detail"],
            item["tone"],
            title_size=item.get("title_size", 8.8),
            detail_size=item.get("detail_size", 7.8),
            rounding=0.075,
        )
    for edge in section["main_edges"]:
        start_x = main_centers[edge["from"]]
        end_x = main_centers[edge["to"]]
        add_arrow(
            ax,
            (start_x + main_size[0] / 2 + 0.015, main_y),
            (end_x - main_size[0] / 2 - 0.015, main_y),
            width=1.1,
            dashed=edge["kind"] == "data",
        )

    init_edge = section["reinitialization_edge"]
    base_x = main_centers[init_edge["from"]]
    sft_x = main_centers[init_edge["to"]]
    init_y = 5.68
    add_polyline_arrow(
        ax,
        [
            (base_x, main_y + main_size[1] / 2),
            (base_x, init_y),
            (sft_x, init_y),
            (sft_x, main_y + main_size[1] / 2),
        ],
        width=1.0,
    )
    ax.text(
        (base_x + sft_x) / 2,
        init_y,
        init_edge["label"],
        ha="center",
        va="center",
        fontsize=7.7,
        color="#596672",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
        zorder=2,
    )

    # The B.3.3 mixed dataset is also the data source for student SFT.
    # Dashed arrows distinguish this dataset flow from model initialization.
    ax.add_patch(
        FancyBboxPatch(
            (0.18, 0.50),
            18.84,
            2.25,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            facecolor="#FBFCFD",
            edgecolor="#E0E6EA",
            linewidth=0.85,
            zorder=0,
        )
    )
    ax.text(0.42, 2.48, section["distill_label"], fontsize=10.2, fontweight="semibold", color="#514873", va="center")
    distill_nodes = section["distill_nodes"]
    distill_centers = [4.20, 9.85, 15.50]
    distill_y = 1.42
    distill_size = (3.25, 0.90)
    for item, x in zip(distill_nodes, distill_centers):
        add_box(ax, (x, distill_y), distill_size, item["title"], item["detail"], item["tone"], title_size=9.0, detail_size=7.8)
    data_x = main_centers[section["dataset_node"]]
    add_polyline_arrow(
        ax,
        [
            (data_x, main_y - main_size[1] / 2),
            (data_x, 2.98),
            (distill_centers[2], 2.98),
            (distill_centers[2], distill_y + distill_size[1] / 2 + 0.02),
        ],
        width=1.1,
        dashed=True,
    )
    ax.text(
        13.20,
        2.99,
        "同一精选数据集",
        ha="center",
        va="center",
        fontsize=7.7,
        color="#596672",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
        zorder=2,
    )
    add_arrow(ax, (distill_centers[2] - distill_size[0] / 2 - 0.05, distill_y), (distill_centers[1] + distill_size[0] / 2 + 0.05, distill_y), width=1.1, dashed=True)
    add_arrow(ax, (distill_centers[1] - distill_size[0] / 2 - 0.05, distill_y), (distill_centers[0] + distill_size[0] / 2 + 0.05, distill_y), width=1.1)
    ax.text(18.98, 0.34, section["footer"], ha="right", va="center", fontsize=7.9, color="#65727B")
    save_figure(fig, "r1_pipeline")


def main() -> None:
    configure_fonts()
    content = load_content()
    draw_timeline(content)
    draw_loops(content)
    draw_reward_layers(content)
    draw_r1_pipeline(content)


if __name__ == "__main__":
    main()
