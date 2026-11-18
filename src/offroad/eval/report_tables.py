"""Turn evaluate.py's per-frame results into report tables, as Markdown and LaTeX (pure, no file I/O).

Input rows are the dicts in outputs/eval/per_frame.csv (all values are strings there);
`parse_row` turns them back into bools, floats, and None.
"""

import numpy as np

from offroad.eval.compare import BASELINE, paired_comparison

BOOL_FIELDS = ("success", "reached", "goal_blocked")
TEXT_FIELDS = ("category", "sequence", "condition", "answer")

# (column title, metric, format) for the overall table. "count" = how many frames are True.
OVERALL_COLUMNS = [
    ("Success", "success", "count"),
    ("Reached", "reached", "count"),
    ("Short (m)", "shortfall_m", "num"),
    ("Detour", "detour_ratio", "num"),
    ("Water", "share_water", "pct"),
    ("Mud", "share_mud", "pct"),
    ("Veg. ok", "share_compliant", "pct"),
    ("Veg. bad", "share_noncompliant", "pct"),
    ("Hard min (m)", "hard_dist_min_m", "num"),
    ("Slope (cm/m)", "slope_cm_per_m", "num"),
    ("Plan (s)", "compute_s", "num"),
    ("VLM (s)", "vlm_s", "num"),
]
# (column title, metric, lower is better, tie tolerance) for the paired table.
PAIRED_COLUMNS = [
    ("Water", "share_water", True, 0.01),
    ("Mud", "share_mud", True, 0.01),
    ("Shortfall", "shortfall_m", True, 0.25),
    ("Hard min", "hard_dist_min_m", False, 0.1),
    ("Detour", "detour_ratio", True, 0.02),
]


def parse_row(row: dict) -> dict:
    """A per_frame.csv row with typed values: bools for success/reached/goal_blocked, floats, None for blanks."""
    parsed = {}
    for key, value in row.items():
        if key in TEXT_FIELDS:
            parsed[key] = value
        elif key in BOOL_FIELDS:
            parsed[key] = value == "True"
        elif key == "frame":
            parsed[key] = int(value)
        else:
            parsed[key] = float(value) if value not in ("", "None", "-") else None
    return parsed


def select(rows: list[dict], condition: str, category: str | None = None) -> list[dict]:
    """Rows of one condition, optionally of one category (None = all)."""
    return [r for r in rows if r["condition"] == condition and (category is None or r["category"] == category)]


def mean_of(rows: list[dict], metric: str) -> float | None:
    values = [r[metric] for r in rows if r[metric] is not None]
    return float(np.mean(values)) if values else None


def cell(rows: list[dict], metric: str, kind: str) -> str:
    """One table cell: "k/n" for counts, a percentage, or a number with 2 decimals; "–" if undefined."""
    if kind == "count":
        return f"{sum(bool(r[metric]) for r in rows)}/{len(rows)}"
    value = mean_of(rows, metric)
    if value is None:
        return "–"
    return f"{value:.0%}" if kind == "pct" else f"{value:.2f}"


def overall_table(rows: list[dict], conditions: list[str]) -> tuple[list[str], list[list[str]]]:
    """(header, body) with one row per condition, means over all frames."""
    header = ["Condition", *[title for title, _, _ in OVERALL_COLUMNS]]
    body = [[c, *[cell(select(rows, c), metric, kind) for _, metric, kind in OVERALL_COLUMNS]] for c in conditions]
    return header, body


def category_table(rows: list[dict], conditions: list[str], metric: str, kind: str) -> tuple[list[str], list[list[str]]]:
    """(header, body) with one row per category (plus "All") and one column per condition, for one metric."""
    categories = list(dict.fromkeys(r["category"] for r in rows))
    header = ["Category", *conditions]
    body = [[cat, *[cell(select(rows, c, cat), metric, kind) for c in conditions]] for cat in categories]
    body.append(["All", *[cell(select(rows, c), metric, kind) for c in conditions]])
    return header, body


def paired_table(rows: list[dict], conditions: list[str], baseline: str = BASELINE) -> tuple[list[str], list[list[str]]]:
    """(header, body): per hybrid condition, frames gained/lost in success and wins/losses/ties per metric.

    baseline: the condition name the others are compared with (as it appears in rows).
    """
    frames = list(dict.fromkeys((r["sequence"], r["frame"]) for r in rows))
    keyed = {(r["sequence"], r["frame"], r["condition"]): r for r in rows}
    header = ["Condition", "Success +/−", *[f"{title} W/L/T" for title, _, _, _ in PAIRED_COLUMNS]]
    body = []
    for condition in conditions:
        if condition == baseline:
            continue
        base = [keyed[(s, f, baseline)] for s, f in frames]
        other = [keyed[(s, f, condition)] for s, f in frames]
        gained = sum(not b["success"] and o["success"] for b, o in zip(base, other))
        lost = sum(b["success"] and not o["success"] for b, o in zip(base, other))
        cells = []
        for _, metric, lower, tolerance in PAIRED_COLUMNS:
            c = paired_comparison([b[metric] for b in base], [o[metric] for o in other], lower, tolerance)
            cells.append(f"{c['wins']}/{c['losses']}/{c['ties']}")
        body.append([condition, f"+{gained}/−{lost}", *cells])
    return header, body


def markdown_table(header: list[str], body: list[list[str]]) -> str:
    """A GitHub-style Markdown table; every column but the first is right-aligned."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join([":---", *["---:"] * (len(header) - 1)]) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def latex_escape(text: str) -> str:
    return text.replace("\\", r"\textbackslash{}").replace("%", r"\%").replace("_", r"\_").replace("&", r"\&") \
        .replace("#", r"\#").replace("−", "$-$").replace("–", "--")


def latex_table(header: list[str], body: list[list[str]], caption: str, label: str) -> str:
    """A booktabs LaTeX table (needs \\usepackage{booktabs}); first column left, the rest right."""
    spec = "l" + "r" * (len(header) - 1)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
        " & ".join(latex_escape(h) for h in header) + r" \\", r"\midrule",
        *[" & ".join(latex_escape(c) for c in row) + r" \\" for row in body],
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{{latex_escape(caption)}}}", rf"\label{{{label}}}", r"\end{table}",
    ]
    return "\n".join(lines)
