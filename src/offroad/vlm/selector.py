"""DriveVLN-style VLM trajectory selector prompt and answer parser (pure, no file I/O).

Given a camera image with colored candidate trajectories drawn on it, asks the VLM to
choose the safest, most traversable path toward the goal and parses the chosen color.
"""

from dataclasses import dataclass, field
import json
import re

from offroad.vlm.answer import extract_json
from offroad.vlm.prompt import VEHICLE_BRIEFING


@dataclass
class SelectorAnswer:
    """Parsed VLM choice among colored candidate trajectories."""

    selected_color: str | None
    reasoning: str = ""
    issues: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.selected_color is not None


def build_selector_prompt(
    available_colors: list[str],
    goal_distance_m: float = 20.0,
    vehicle_briefing: str | None = VEHICLE_BRIEFING,
) -> str:
    """Build the DriveVLN-style trajectory selection prompt.

    available_colors: color names of the candidate paths drawn on the camera image
        (e.g. ["red", "blue", "yellow", "green"]).
    """
    colors_str = ", ".join(available_colors)
    lines = [
        "You are the high-level navigation selector for an autonomous off-road ground vehicle.",
        f"Your goal is to drive safely toward a point about {goal_distance_m:.0f} meters straight ahead.",
        f"The image shows {len(available_colors)} candidate trajectories drawn in distinct colors: {colors_str}.",
    ]
    if vehicle_briefing:
        lines.append(vehicle_briefing)
    lines.extend(
        [
            "Compare the colored paths against the terrain in the image. Prefer dry dirt, gravel, or short grass; "
            "avoid puddles, water, deep mud, people, vehicles, trees, and hard obstacles.",
            "Choose the single best colored path from the available colors.",
            "Respond with ONLY a JSON object in this exact format:",
            '{"reasoning": "<brief comparison of the colored paths>", "selected_color": "<one of: '
            + colors_str
            + '>"}',
        ]
    )
    return "\n\n".join(lines)


def _match_color(raw: str, valid_colors: dict[str, str]) -> str | None:
    """Match a color string against {lowercase: canonical} valid_colors."""
    cleaned = raw.strip().lower()
    if cleaned in valid_colors:
        return valid_colors[cleaned]
    # Handle answers like "blue path" or "Not arrived, yellow."
    for low, canonical in valid_colors.items():
        if re.search(rf"\b{re.escape(low)}\b", cleaned):
            return canonical
    return None

def parse_selector_answer(text: str, available_colors: list[str]) -> SelectorAnswer:
    """Parse a VLM response into a SelectorAnswer. Never raises."""
    issues: list[str] = []
    valid_colors = {c.strip().lower(): c for c in available_colors}

    if not text or not text.strip():
        return SelectorAnswer(selected_color=None, reasoning="", issues=["the answer is empty"])

    if not valid_colors:
        return SelectorAnswer(selected_color=None, reasoning="", issues=["no available colors provided"])

    # 1. Try JSON parsing via extract_json -> (dict | None, list[str]).
    data, json_issues = extract_json(text)
    issues.extend(json_issues)
    if isinstance(data, dict):
        reasoning = str(data.get("reasoning", "")).strip()
        raw_choice = data.get("selected_color") or data.get("color") or data.get("choice")
        if raw_choice is None:
            issues.append("JSON missing 'selected_color' field")
        else:
            matched = _match_color(str(raw_choice), valid_colors)
            if matched is not None:
                return SelectorAnswer(selected_color=matched, reasoning=reasoning, issues=issues)
            issues.append(f"invalid selected_color {raw_choice!r}; expected one of {available_colors}")
            return SelectorAnswer(selected_color=None, reasoning=reasoning, issues=issues)

    # 2. Fallback: check <answer>...</answer> tag (DriveVLN format) or full text.
    tag_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    search_text = tag_match.group(1) if tag_match else text
    found = [
        valid_colors[m.group(0).lower()]
        for m in re.finditer(
            r"\b(" + "|".join(re.escape(c) for c in valid_colors) + r")\b",
            search_text,
            flags=re.IGNORECASE,
        )
    ]
    if found:
        return SelectorAnswer(selected_color=found[-1], reasoning=text.strip(), issues=issues)

    issues.append(f"could not find any available color {available_colors} in response")
    return SelectorAnswer(selected_color=None, reasoning=text.strip(), issues=issues)