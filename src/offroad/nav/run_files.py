"""Find a selector run's answer files under outputs/selector, which is sorted into subfolders.

A run is the set of files run_selector.py writes for one tag: <tag>_modeA.jsonl and/or
<tag>_modeB.jsonl (plus a summary and a few pictures). Finished runs are moved into one
subfolder per collection (e.g. outputs/selector/phase4_test_zeroshot/), so a tag is looked up
in the whole tree. A tag must be in one folder only.
"""

from pathlib import Path

MODES = ("A", "B")


def answer_file_name(tag: str, mode: str) -> str:
    return f"{tag}_mode{mode}.jsonl"


def find_run_dir(root: Path, tag: str) -> Path | None:
    """The folder under `root` (root itself or any subfolder) that holds the tag's answer files.

    None if there are none. Raises ValueError if the tag's answer files are in more than one folder.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    folders = sorted({p.parent for m in MODES for p in root.rglob(answer_file_name(tag, m))})
    if len(folders) > 1:
        listed = ", ".join(str(f) for f in folders)
        raise ValueError(f"run {tag!r} has answer files in more than one folder: {listed}")
    return folders[0] if folders else None
