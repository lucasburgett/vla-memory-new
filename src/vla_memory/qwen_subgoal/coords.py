"""Coordinate-space conversion between our oracle grounding format and Qwen-VL's
native grounding space.

Two spaces are in play:
  - **Oracle / GroundSG format:** ``<y, x>`` in 0–``img_size`` PIXELS — what the
    env's ``grounded_subgoal_online`` emits and what the frozen GroundSG π0.5 was
    trained to consume.
  - **Qwen-VL native format:** ``<x, y>`` normalized to 0–1000 — what the base VLM
    localizes in. SFT targets are rescaled to this so the model reuses its own
    grounding instead of relearning a foreign space (see
    ``project_qwen_coordinate_space``); the input-blind constant-coordinate failure
    was this mismatch.

``to_qwen_xy`` builds the SFT target (pixels → Qwen) at dataset-build time.
``from_qwen_xy`` is its inverse, applied at INFERENCE to the VLM's predicted
subtask before it reaches GroundSG. The pair round-trips to ±1px (integer rounding
at each scale). Both are no-ops on strings without a ``<a, b>`` coordinate.
"""

from __future__ import annotations

import re

# Both formats are literally "<a, b>"; only axis order + scale differ (per fn).
_COORD = re.compile(r"<\s*(\d+)\s*,\s*(\d+)\s*>")


def to_qwen_xy(text: str, img_size: int = 256) -> str:
    """``<y, x>`` 0–img_size pixels → Qwen-native ``<x, y>`` 0–1000."""
    m = _COORD.search(text)
    if not m:
        return text
    y, x = int(m.group(1)), int(m.group(2))   # oracle format is <y, x>
    xn = round(x / img_size * 1000)
    yn = round(y / img_size * 1000)
    return text[: m.start()] + f"<{xn}, {yn}>" + text[m.end():]


def from_qwen_xy(text: str, img_size: int = 256) -> str:
    """Qwen-native ``<x, y>`` 0–1000 → ``<y, x>`` 0–img_size pixels (GroundSG format).

    Inverse of ``to_qwen_xy``. Applied to the VLM's predicted subtask at inference
    so the frozen GroundSG π0.5 receives the pixel ``<y, x>`` it was trained on.
    Without this, GroundSG reads the 0–1000 value as a pixel and the GRPO reward is
    silently wrong → a fake flatline that looks algorithmic.
    """
    m = _COORD.search(text)
    if not m:
        return text
    xn, yn = int(m.group(1)), int(m.group(2))   # Qwen format is <x, y>
    x = round(xn / 1000 * img_size)
    y = round(yn / 1000 * img_size)
    return text[: m.start()] + f"<{y}, {x}>" + text[m.end():]


__all__ = ["to_qwen_xy", "from_qwen_xy"]
