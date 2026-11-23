"""Unit tests for the DriveVLN-style VLM selector prompt and parser."""

from offroad.vlm.selector import build_selector_prompt, parse_selector_answer


def test_build_selector_prompt_lists_colors_and_briefing() -> None:
    prompt = build_selector_prompt(["red", "blue", "yellow"], goal_distance_m=20.0)
    assert "red, blue, yellow" in prompt
    assert "20 meters" in prompt
    assert "selected_color" in prompt

    no_brief = build_selector_prompt(["red", "blue"], vehicle_briefing=None)
    assert len(no_brief) < len(prompt)


def test_parse_selector_answer_valid_json() -> None:
    raw = '```json\n{"reasoning": "Blue avoids the puddle.", "selected_color": "blue"}\n```'
    ans = parse_selector_answer(raw, ["red", "blue", "yellow"])
    assert ans.valid
    assert ans.selected_color == "blue"
    assert ans.reasoning == "Blue avoids the puddle."
    assert ans.issues == []


def test_parse_selector_answer_rejects_unavailable_color() -> None:
    raw = '{"reasoning": "Green looks best.", "selected_color": "green"}'
    ans = parse_selector_answer(raw, ["red", "blue"])
    assert not ans.valid
    assert ans.selected_color is None
    assert any("invalid selected_color" in issue for issue in ans.issues)


def test_parse_selector_answer_handles_drivevln_tag_fallback() -> None:
    raw = "<think>Red hits mud, yellow is dry.</think><answer>Not arrived, yellow.</answer>"
    ans = parse_selector_answer(raw, ["red", "blue", "yellow"])
    assert ans.valid
    assert ans.selected_color == "yellow"
    assert len(ans.issues) == 1  # notes text fallback


def test_parse_selector_answer_handles_empty_text() -> None:
    ans = parse_selector_answer("   ", ["red", "blue"])
    assert not ans.valid
    assert ans.selected_color is None
    assert "the answer is empty" in ans.issues