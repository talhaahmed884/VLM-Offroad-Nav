"""Report figures from the evaluation results: metrics by category, wins/losses, latency, example frames.

Colors follow the condition, never its position: baseline = neutral gray; the hybrids use the
first three slots of the validated default categorical palette (blue, orange, aqua), which
are colorblind-safe as a set. Text stays in ink colors; gridlines are hairlines.
Scripts using this should call matplotlib.use("Agg") before importing it if no display is available.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from offroad.data.calibration import CameraCalibration
from offroad.eval.compare import BASELINE, ConditionResult, paired_comparison
from offroad.eval.report_tables import PAIRED_COLUMNS, mean_of, select
from offroad.viz.baseline_figure import draw_path_on_camera
from offroad.viz.topdown import draw_camera_fov, label_colors, show_top_down

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"
BASELINE_GRAY = "#898781"
SERIES = {
    "DriveVLN": "#8e44ad", "8B": "#2a78d6",
    "oracle": "#eb6834", "4B": "#1baf7a"
}  # palette slots 1, 2, 3
WIN, LOSS = "#2a78d6", "#e34948"  # the palette's diverging poles (blue = better, red = worse)


def display_name(condition: str) -> str:
    """'VLM-8B_4bit' -> 'Hybrid 8B', 'oracle' -> 'Hybrid oracle', 'baseline' -> 'Baseline'."""
    if condition == BASELINE:
        return "Baseline"
    if condition == "oracle":
        return "Hybrid oracle"
    return "Hybrid " + condition.replace("VLM-", "").split("_")[0]


def condition_color(condition: str) -> str:
    if condition == BASELINE:
        return BASELINE_GRAY
    for key, color in SERIES.items():
        if key in condition:
            return color
    return INK_SECONDARY


def style_axes(ax):
    """Recessive chrome: no top/right spines, muted ticks, hairline horizontal grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def grouped_bars(ax, rows: list[dict], conditions: list[str], metric: str, kind: str, title: str):
    """One group of bars per category (plus "All"), one bar per condition. kind: "rate", "pct", or "num"."""
    categories = [*dict.fromkeys(r["category"] for r in rows), None]
    width = 0.8 / len(conditions)
    for k, condition in enumerate(conditions):
        values = []
        for category in categories:
            group = select(rows, condition, category)
            if kind == "rate":
                values.append(sum(r[metric] for r in group) / len(group) if group else np.nan)
            else:
                value = mean_of(group, metric)
                values.append(np.nan if value is None else value)
        x = np.arange(len(categories)) + (k - (len(conditions) - 1) / 2) * width
        ax.bar(x, values, width=width * 0.9, color=condition_color(condition), label=display_name(condition))
    ax.set_xticks(np.arange(len(categories)), [c or "all" for c in categories])
    if kind in ("rate", "pct"):
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    style_axes(ax)


def by_category_figure(rows: list[dict], conditions: list[str]) -> plt.Figure:
    """Small multiples: six key metrics, grouped by category."""
    panels = [
        ("success", "rate", "Success rate (higher is better)"),
        ("share_water", "pct", "Path through water (lower is better)"),
        ("share_mud", "pct", "Path through mud (lower is better)"),
        ("share_compliant", "pct", "Path through grass/bush (drivable)"),
        ("shortfall_m", "num", "Stops short of target, m (lower is better)"),
        ("hard_dist_min_m", "num", "Closest hard obstacle, m (higher is safer)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), facecolor=SURFACE)
    for ax, (metric, kind, title) in zip(axes.flat, panels):
        grouped_bars(ax, rows, conditions, metric, kind, title)
    handles = [Patch(color=condition_color(c), label=display_name(c)) for c in conditions]
    fig.legend(handles=handles, loc="upper center", ncol=len(conditions), frameon=False, fontsize=10,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def paired_figure(rows: list[dict], conditions: list[str]) -> plt.Figure:
    """Frame by frame vs the baseline: per hybrid and metric, frames better (right) and worse (left)."""
    hybrids = [c for c in conditions if c != BASELINE]
    frames = list(dict.fromkeys((r["sequence"], r["frame"]) for r in rows))
    keyed = {(r["sequence"], r["frame"], r["condition"]): r for r in rows}
    fig, axes = plt.subplots(1, len(hybrids), figsize=(5.2 * len(hybrids), 3.8), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, condition in zip(axes, hybrids):
        base = [keyed[(s, f, BASELINE)] for s, f in frames]
        other = [keyed[(s, f, condition)] for s, f in frames]
        titles, wins, losses, ties = [], [], [], []
        for title, metric, lower, tolerance in PAIRED_COLUMNS:
            c = paired_comparison([b[metric] for b in base], [o[metric] for o in other], lower, tolerance)
            titles.append(title)
            wins.append(c["wins"])
            losses.append(c["losses"])
            ties.append(c["ties"])
        y = np.arange(len(titles))
        ax.barh(y, wins, height=0.55, color=WIN, label="better than baseline")
        ax.barh(y, [-n for n in losses], height=0.55, color=LOSS, label="worse than baseline")
        for yi, w, l, t in zip(y, wins, losses, ties):
            ax.text(max(w, 0) + 0.4, yi, f"{w} / {l}  ({t} same)", va="center", fontsize=8.5, color=INK_SECONDARY)
        ax.axvline(0, color="#c3c2b7", linewidth=1)
        ax.set_yticks(y, titles)
        ax.invert_yaxis()
        limit = max(max(wins), max(losses), 1) + 9
        ax.set_xlim(-max(max(losses), 1) - 1, limit)
        ax.set_xlabel("frames (worse ← | → better)", color=INK_SECONDARY, fontsize=9)
        ax.set_title(display_name(condition), fontsize=11, color=INK, loc="left")
        style_axes(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    handles = [Patch(color=WIN, label="better than baseline"), Patch(color=LOSS, label="worse than baseline")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def latency_figure(rows: list[dict], conditions: list[str]) -> plt.Figure:
    """Seconds per frame on one log axis: geometry compute per condition, then the VLM answer time."""
    labels, values, colors = [], [], []
    for condition in conditions:
        labels.append(f"{display_name(condition)}: geometry")
        values.append(mean_of(select(rows, condition), "compute_s") or np.nan)
        colors.append(condition_color(condition))
    for condition in conditions:
        vlm = mean_of(select(rows, condition), "vlm_s")
        if vlm is not None:
            labels.append(f"{display_name(condition)}: VLM answer")
            values.append(vlm)
            colors.append(condition_color(condition))
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(labels) + 1.2), facecolor=SURFACE)
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.55, color=colors)
    for yi, v in zip(y, values):
        ax.text(v * 1.15, yi, f"{v:.2f} s" if v < 10 else f"{v:.1f} s", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(min(values) / 2, max(values) * 4)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g} s"))  # plain text, not 10^x
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("seconds per frame (log scale)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Compute time per frame", fontsize=11, color=INK, loc="left")
    style_axes(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.tight_layout()
    return fig


def example_figure(
    image: np.ndarray,
    calib: CameraCalibration,
    results: dict[str, ConditionResult],
    id_to_name: dict[int, str],
    id_to_color: dict[int, tuple[int, int, int]],
    title: str,
) -> plt.Figure:
    """One frame: camera image with every condition's path, and the top-down ground-truth classes with the paths."""
    names = list(results)
    # Widths match the panels' shapes (camera 1.6 : 1, top-down map about 0.85 : 1), so little space is wasted.
    # Save with bbox_inches="tight" so the multi-line title above the camera image is never clipped.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.6, 0.85]}, facecolor=SURFACE)
    camera = image
    radii = np.linspace(9, 4, len(names)).round().astype(int)  # nested dots, so overlapping paths stay visible
    for name, radius in zip(names, radii):
        plan = results[name].plan
        if plan.found:
            rgb = tuple(int(condition_color(name)[i:i + 2], 16) for i in (1, 3, 5))
            camera = draw_path_on_camera(camera, calib, plan, rgb, radius=int(radius))
    axes[0].imshow(camera)
    axes[0].axis("off")
    lines = [title] + [
        f"{display_name(n)}: {'success' if r.metrics.success else 'fail'}, "
        f"water {r.metrics.share_water or 0:.0%}, mud {r.metrics.share_mud or 0:.0%}, "
        f"short {r.metrics.shortfall_m or 0:.1f} m" for n, r in results.items()]
    axes[0].set_title("\n".join(lines), fontsize=10.5, color=INK, loc="left")

    base = results[names[0]].plan
    show_top_down(axes[1], label_colors(base.labels, id_to_color), base.spec, "ground-truth class (LiDAR) and paths")
    draw_camera_fov(axes[1], calib.K, image.shape[1], base.spec)
    tx, ty = base.target_xy
    axes[1].plot(ty, tx, "X", color="white", markersize=13, markeredgecolor=INK)
    for name in names:
        plan = results[name].plan
        if plan.found:
            xy = plan.path_xy()
            axes[1].plot(xy[:, 1], xy[:, 0], "-", color=condition_color(name), linewidth=2.5, label=display_name(name))
    axes[1].legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    return fig
