"""Render the implementation map from data/concept_figures.json.

Run from the repository root with:
    .venv/Scripts/python.exe src/plot_concepts.py

The implementation map is exported as PNG, SVG, and PDF.
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
    draw_implementation(content)


if __name__ == "__main__":
    main()
