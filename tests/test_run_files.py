"""Tests for finding selector runs in the sorted outputs/selector tree."""

import pytest

from offroad.nav.run_files import answer_file_name, find_run_dir


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
    return path


def test_finds_a_run_in_the_root_or_a_subfolder(tmp_path):
    touch(tmp_path / answer_file_name("val_a", "A"))
    touch(tmp_path / "phase5" / answer_file_name("test_b", "B"))
    assert find_run_dir(tmp_path, "val_a") == tmp_path
    assert find_run_dir(tmp_path, "test_b") == tmp_path / "phase5"


def test_a_tag_is_not_confused_with_a_longer_tag(tmp_path):
    touch(tmp_path / "p4" / answer_file_name("test_4B_nohist", "A"))
    assert find_run_dir(tmp_path, "test_4B") is None
    assert find_run_dir(tmp_path / "missing", "test_4B") is None


def test_a_run_in_two_folders_is_an_error(tmp_path):
    touch(tmp_path / "p4" / answer_file_name("test_4B", "A"))
    touch(tmp_path / "p5" / answer_file_name("test_4B", "B"))
    with pytest.raises(ValueError, match="more than one folder"):
        find_run_dir(tmp_path, "test_4B")
