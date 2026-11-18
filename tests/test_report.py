"""Tests for the report tables and charts (made-up evaluation rows; no dataset needed)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from offroad.eval.report_tables import (  # noqa: E402
    category_table,
    latex_escape,
    latex_table,
    markdown_table,
    overall_table,
    paired_table,
    parse_row,
)
from offroad.viz.report_charts import (  # noqa: E402
    by_category_figure,
    condition_color,
    display_name,
    latency_figure,
    paired_figure,
)

CONDITIONS = ["baseline", "VLM-8B_4bit"]


def row(frame, category, condition, success, water, vlm=None):
    return {"category": category, "sequence": "00000", "frame": frame, "condition": condition, "answer": "-",
            "success": success, "reached": success, "goal_blocked": False, "shortfall_m": 0.0 if success else 3.0,
            "length_m": 20.0, "detour_ratio": 1.05, "share_water": water, "share_mud": 0.1, "share_compliant": 0.6,
            "share_noncompliant": 0.0, "share_unseen": 0.2, "hard_dist_mean_m": 4.0, "hard_dist_min_m": 1.5,
            "slope_cm_per_m": 6.0, "bump_height_m": 0.05, "compute_s": 0.06, "vlm_s": vlm,
            "apart_mean_m": 0.0, "apart_max_m": 0.0}


ROWS = [
    row(1, "water", "baseline", True, 0.30), row(1, "water", "VLM-8B_4bit", True, 0.10, 13.0),
    row(2, "bush", "baseline", False, 0.00), row(2, "bush", "VLM-8B_4bit", True, 0.00, 14.0),
]


def test_parse_row_types():
    parsed = parse_row({"category": "mud", "sequence": "00004", "frame": "1292", "condition": "baseline",
                        "answer": "-", "success": "True", "reached": "False", "goal_blocked": "False",
                        "shortfall_m": "0.18", "vlm_s": ""})
    assert parsed["success"] is True and parsed["reached"] is False
    assert parsed["frame"] == 1292 and parsed["sequence"] == "00004"
    assert parsed["shortfall_m"] == 0.18 and parsed["vlm_s"] is None


def test_overall_table():
    header, body = overall_table(ROWS, CONDITIONS)
    assert header[:3] == ["Condition", "Success", "Reached"]
    assert body[0][:2] == ["baseline", "1/2"] and body[1][:2] == ["VLM-8B_4bit", "2/2"]
    water = header.index("Water")
    assert body[0][water] == "15%" and body[1][water] == "5%"
    assert body[0][header.index("VLM (s)")] == "–"  # the baseline has no VLM time


def test_category_table_has_an_all_row():
    header, body = category_table(ROWS, CONDITIONS, "share_water", "pct")
    assert header == ["Category", *CONDITIONS]
    assert [r[0] for r in body] == ["water", "bush", "All"]
    assert body[0] == ["water", "30%", "10%"]


def test_paired_table():
    header, body = paired_table(ROWS, CONDITIONS)
    assert len(body) == 1 and body[0][0] == "VLM-8B_4bit"
    assert body[0][1] == "+1/−0"  # the bush frame becomes a success
    assert body[0][header.index("Water W/L/T")] == "1/0/1"


def test_paired_table_with_renamed_baseline():
    renamed = [{**r, "condition": display_name(r["condition"])} for r in ROWS]
    _, body = paired_table(renamed, [display_name(c) for c in CONDITIONS], baseline="Baseline")
    assert body[0][0] == "Hybrid 8B"


def test_markdown_and_latex():
    md = markdown_table(["A", "B"], [["x", "1%"]])
    assert md.splitlines() == ["| A | B |", "|:---|---:|", "| x | 1% |"]
    assert latex_escape("12% of a_b −1") == r"12\% of a\_b $-$1"
    tex = latex_table(["A", "B"], [["x", "1%"]], caption="A caption", label="tab:x")
    assert r"\begin{tabular}{lr}" in tex and r"x & 1\% \\" in tex and r"\label{tab:x}" in tex


def test_names_and_colors():
    assert display_name("baseline") == "Baseline"
    assert display_name("VLM-4B_4bit") == "Hybrid 4B"
    assert display_name("oracle") == "Hybrid oracle"
    assert len({condition_color(c) for c in ["baseline", "VLM-4B_4bit", "VLM-8B_4bit", "oracle"]}) == 4


def test_charts_draw_without_errors():
    for make in (by_category_figure, paired_figure, latency_figure):
        fig = make(ROWS, CONDITIONS)
        assert fig.axes
        plt.close(fig)
