"""Make the report's tables and figures from the evaluation results (phase 6, step 4).

Reads outputs/eval/per_frame.csv (run scripts/evaluate.py first) and writes outputs/report/:
    tables.md / tables.tex      overall, per-category, and frame-by-frame tables (Markdown and LaTeX)
    fig_by_category.png         six key metrics per category and condition
    fig_paired.png              frames better / worse than the baseline, per hybrid and metric
    fig_latency.png             geometry compute time vs VLM answer time
    examples/<frame>.png        a few frames with every condition's path (re-planned here)

The LaTeX tables use booktabs (\\usepackage{booktabs}).

Usage (from the repo root):
    python scripts/make_report.py
    python scripts/make_report.py --no-examples
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file only; also works without a display (e.g. on the cluster)
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import offroad` work without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from offroad.data.rellis import RellisSequence, load_ontology  # noqa: E402
from offroad.eval.compare import BASELINE, evaluate_frame  # noqa: E402
from offroad.eval.report_tables import (  # noqa: E402
    category_table,
    latex_table,
    markdown_table,
    overall_table,
    paired_table,
    parse_row,
)
from offroad.viz.report_charts import (  # noqa: E402
    by_category_figure,
    display_name,
    example_figure,
    latency_figure,
    paired_figure,
)
from offroad.vlm.grid import ImageGrid  # noqa: E402
from offroad.vlm.oracle import oracle_answer  # noqa: E402

# Frames that show the main findings (water avoided, grass opened, blind zone, over-blocking).
EXAMPLES = ["water_00003_000158", "water_00000_000120", "bush_00002_000935", "bush_00001_002132", "easy_00003_001060"]
EXAMPLE_RUN = "Qwen3-VL-8B-Instruct_4bit"  # the main VLM, shown next to the baseline and the oracle

CATEGORY_TABLES = [
    ("share_water", "pct", "Share of the path through water or puddles (lower is better)", "tab:water"),
    ("share_mud", "pct", "Share of the path through mud (RELLIS-3D labels trail ruts as mud)", "tab:mud"),
    ("success", "count", "Frames where the path reached the target, or stopped within 2.5 m of a blocked target",
     "tab:success"),
    ("hard_dist_min_m", "num", "Closest approach to a hard obstacle along the path, m (mean over frames)",
     "tab:clearance"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "outputs" / "eval" / "per_frame.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "report")
    parser.add_argument("--no-examples", action="store_true", help="skip the example-frame pictures (no dataset needed)")
    parser.add_argument("--root", type=Path, default=None, help="Rellis-3D folder (default: $RELLIS_ROOT or data/Rellis-3D)")
    return parser.parse_args()


def write_tables(rows: list[dict], conditions: list[str], out_dir: Path):
    """tables.md and tables.tex with the same tables; condition names are the display names."""
    pretty = [{**r, "condition": display_name(r["condition"])} for r in rows]
    names = [display_name(c) for c in conditions]
    tables = [("Overall results (means over all frames)", "tab:overall", overall_table(pretty, names))]
    tables += [(caption, label, category_table(pretty, names, metric, kind))
               for metric, kind, caption, label in CATEGORY_TABLES]
    tables.append(("Frame by frame against the baseline: success frames gained/lost, and wins/losses/ties "
                   "per metric (ties within 0.01 share, 0.25 m shortfall, 0.1 m clearance, 0.02 detour)",
                   "tab:paired", paired_table(pretty, names, baseline=display_name(BASELINE))))
    md = [f"### {caption}\n\n{markdown_table(*table)}\n" for caption, _, table in tables]
    tex = [latex_table(*table, caption=caption, label=label) for caption, label, table in tables]
    (out_dir / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "tables.tex").write_text("\n\n".join(tex) + "\n", encoding="utf-8")


def save(fig: plt.Figure, path: Path):
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def make_examples(out_dir: Path, root: Path | None):
    """Re-plan a few frames with the baseline, the main VLM, and the oracle, and draw them."""
    id_to_name, id_to_color = load_ontology(root)
    frames_csv = {f"{r['category']}_{r['sequence']}_{int(r['frame']):06d}": r
                  for r in csv.DictReader(open(REPO_ROOT / "configs" / "eval_frames.csv", newline=""))}
    run_dir = REPO_ROOT / "outputs" / "vlm_runs" / EXAMPLE_RUN
    (out_dir / "examples").mkdir(exist_ok=True)
    for stem in EXAMPLES:
        if stem not in frames_csv:
            print(f"  {stem}: not in eval_frames.csv, skipped")
            continue
        entry = frames_csv[stem]
        frame = RellisSequence(entry["sequence"], root=root)[int(entry["frame"])]
        grid = ImageGrid(height=frame.image.shape[0], width=frame.image.shape[1])
        answers = {}
        answer_path = run_dir / f"{stem}.json"
        if answer_path.exists():
            answers["VLM-8B_4bit"] = (json.loads(answer_path.read_text(encoding="utf-8"))["answer"], None)
        if frame.label_image is not None:
            answers["oracle"] = (oracle_answer(frame.label_image, grid, id_to_name), None)
        results = evaluate_frame(frame.points, frame.point_labels, frame.calib.T_cam_from_lidar, frame.calib.K,
                                 grid, frame.label_image, answers, id_to_name)
        title = f"{entry['category']}: sequence {entry['sequence']} frame {entry['frame']}"
        save(example_figure(frame.image, frame.calib, results, id_to_name, id_to_color, title),
             out_dir / "examples" / f"{stem}.png")
        print(f"  saved examples/{stem}.png")


def main() -> None:
    args = parse_args()
    if not args.results.exists():
        sys.exit(f"{args.results} not found. Run scripts/evaluate.py first.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [parse_row(r) for r in csv.DictReader(open(args.results, newline=""))]
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    if conditions[0] != BASELINE:
        sys.exit("per_frame.csv has no baseline rows")

    write_tables(rows, conditions, args.out_dir)
    save(by_category_figure(rows, conditions), args.out_dir / "fig_by_category.png")
    save(paired_figure(rows, conditions), args.out_dir / "fig_paired.png")
    save(latency_figure(rows, conditions), args.out_dir / "fig_latency.png")
    print(f"Saved tables.md, tables.tex, fig_by_category.png, fig_paired.png, fig_latency.png to {args.out_dir}")
    if not args.no_examples:
        print("Example frames:")
        make_examples(args.out_dir, args.root)


if __name__ == "__main__":
    main()
