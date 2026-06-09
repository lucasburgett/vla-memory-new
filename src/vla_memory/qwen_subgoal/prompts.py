"""Prompt templates for the MemER-style Qwen3-VL subgoal predictor.

This is the **memory** prompt format: the VLM conditions on a list of *past
keyframes* (selected frames it must remember) plus the *recent execution frames*
(current context), and emits a JSON object naming the current subtask and which
recent frames are keyframes. It mirrors MemER (Pan et al., arXiv:2510.20328).

Must match the SFT dataset builder
``robomme_policy_learning/src/mme_vla_suite/dataset_builder/build_vlm_subgoal_dataset_memer.py``.
Any drift between train- and inference-time prompts silently degrades the policy
(the "prompt-parity invariant"). The system prompt and user template below are
copied verbatim from that builder; the *only* difference between simple and
grounded modes is the target output (grounded embeds a ``<box>`` in
``current_subtask``), so the prompt itself does not branch on ``subgoal_type``.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

# Verbatim from build_vlm_subgoal_dataset_memer.py:SUBGOAL_SYSTEM_PROMPT.
SUBGOAL_SYSTEM_PROMPT = (
    "You are a robot program that predicts actions. The current input images from the front-view camera shows the most recent actions the robot has executed. "
    "The past keyframes are selected frames of particular importance from all the actions the robot has executed so far. Based on these, output the current subtask the robot should execute and nothing else. "
    "Some tasks may have a video input for initial setup, some may not.\n\n"
    "Return a JSON with:\n"
    "- current_subtask: the action that should be executed at the current timestep\n"
    "- keyframe_positions: list of frame positions (1-indexed) from the current input images where actions change"
)

# System prompt for the SELECT call of the joint select-then-use pipeline
# (JOINT_MEMORY_DESIGN.md). DELIBERATELY DISTINCT from SUBGOAL_SYSTEM_PROMPT: the
# two calls share near-identical IMAGE inputs, so if the prompts were also alike the
# model collapses `keyframe_positions` to the majority target (`[]`, from USE rows)
# and the SELECT head learns nothing. A separate system prompt + the mode="select"
# user prompt + a SELECT target that carries ONLY keyframe_positions make the two
# tasks unambiguous. See project_sft_plan_adjustments (the SELECT/USE schema fix).
SELECT_SYSTEM_PROMPT = (
    "You are a robot program with visual memory. You are shown the most recent frames the robot has executed. "
    "Your job is to decide which of these frames are worth REMEMBERING as keyframes — frames that capture "
    "information needed for later subtasks (for example, where an object is before it becomes hidden). "
    "Output the keyframe positions and nothing else.\n\n"
    "Return a JSON with:\n"
    "- keyframe_positions: list of frame positions (1-indexed) from the current input images to remember"
)


def _wrap_images(n: int) -> str:
    """Render ``n`` ``<image>`` placeholders as MemER's bracketed list.

    Matches ``build_vlm_subgoal_dataset_memer.py:_wrap_images`` exactly: ``[]``
    when empty, otherwise ``[<image>, <image>, ...]``. The Qwen processor later
    replaces each ``<image>`` with that image's vision-token span, in order, so
    the count here must equal the number of frames handed to the processor.
    """
    if n <= 0:
        return "[]"
    return "[" + ", ".join("<image>" for _ in range(n)) + "]"


def _wrap_history(history_subgoals: List[str]) -> str:
    return "; ".join(f"{i + 1}. {sg}" for i, sg in enumerate(history_subgoals))


def build_user_prompt(
    task_goal: str,
    n_key_frames: int,
    n_recent_frames: int,
    history_subgoals: Optional[List[str]] = None,
    has_video_demo: bool = False,
    mode: str = "use",
) -> str:
    """Build the MemER user-turn prompt (keyframes for memory + history for timing).

    Args:
        task_goal: natural-language task description from ``env.reset().info``.
        n_key_frames: number of past keyframes (memory) prepended to the images.
        n_recent_frames: number of recent execution frames (current context).
        history_subgoals: subtasks already completed this episode (e.g.
            ``["press the button"]``). Without this the model can't tell the
            press phase is DONE — the decision-point frames look like "still at
            the button" (bins drop at t=64, so nothing visually says "now pick"),
            and the SFT collapsed to a constant "press the button" output. The
            original RoboMME recipe carries this signal; the keyframes-only MemER
            prompt dropped it.
        has_video_demo: True when the task ships an initial-setup demo video; a
            ``<video>`` placeholder is prepended (the processor expands it).
        mode: ``"use"`` (default) — predict the next subtask from past keyframes +
            recent frames (the one-shot path AND the joint USE call); the returned
            string is byte-identical to the original prompt. ``"select"`` — the
            joint SELECT call: choose which of the recent frames to remember as
            keyframes. The two modes MUST be distinct (see ``SELECT_SYSTEM_PROMPT``).

    Returns:
        The user-turn string. ``<image>`` / ``<video>`` placeholders are replaced
        by the processor; the order of images passed to the processor MUST be
        ``key_frames + recent_frames`` to line up with the placeholder order here.
    """
    video_prefix = (
        "The task has a video input for initial setup: <video>\n" if has_video_demo else ""
    )
    history_line = (
        f"The subtasks already completed are: {_wrap_history(history_subgoals)}\n"
        if history_subgoals
        else ""
    )
    if mode == "select":
        # SELECT call: no past keyframes; pick which RECENT frames to remember.
        return (
            f"{video_prefix}The task goal is: {task_goal}\n"
            f"{history_line}"
            f"Here is current input image list from the front-view camera: {_wrap_images(n_recent_frames)}\n\n"
            "Which of the current input frames should be remembered as keyframes for "
            "later subtasks? Output only keyframe_positions."
        )
    if mode != "use":
        raise ValueError(f"build_user_prompt: mode must be 'use' or 'select', got {mode!r}")
    return (
        f"{video_prefix}The task goal is: {task_goal}\n"
        f"{history_line}"
        f"Here are the selected frames from the entirety of the full execution that are of particular importance:{_wrap_images(n_key_frames)}\n"
        f"Here is current input image list from the front-view camera: {_wrap_images(n_recent_frames)}\n\n"
        "What subtask should the robot execute and what is the keyframe position?"
    )


def build_messages(
    task_goal: str,
    n_key_frames: int,
    n_recent_frames: int,
    history_subgoals: Optional[List[str]] = None,
    has_video_demo: bool = False,
    mode: str = "use",
) -> List[dict]:
    """Build the OpenAI-style messages array (system + user turn).

    ``mode`` selects the system prompt (``SELECT_SYSTEM_PROMPT`` for ``"select"``,
    ``SUBGOAL_SYSTEM_PROMPT`` for ``"use"``) and the user-prompt variant — both must
    match between the SFT builder and the rollout/trainer (the prompt-parity invariant).
    """
    system = SELECT_SYSTEM_PROMPT if mode == "select" else SUBGOAL_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": build_user_prompt(
                task_goal=task_goal,
                n_key_frames=n_key_frames,
                n_recent_frames=n_recent_frames,
                history_subgoals=history_subgoals,
                has_video_demo=has_video_demo,
                mode=mode,
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# Matches the JSON the model is trained to emit, e.g.
#   {"current_subtask": "pick up the container at <box>...</box>", "keyframe_positions": [3]}
# We parse defensively: the generated string may carry trailing tokens, single
# quotes, or a truncated tail, and a hard JSON error must not crash a rollout.
_SUBTASK_RE = re.compile(r'"current_subtask"\s*:\s*"((?:[^"\\]|\\.)*)"')
# Closing ``]`` is optional so a truncated generation ("...[2") still recovers
# the positions seen so far rather than dropping them silently.
_KEYFRAMES_RE = re.compile(r'"keyframe_positions"\s*:\s*\[([^\]]*)\]?')


def parse_subgoal_output(text: str) -> Tuple[str, List[int]]:
    """Parse the MemER JSON response into ``(current_subtask, keyframe_positions)``.

    The ``current_subtask`` string is what we feed to the GroundSG π0.5 as the
    grounded subgoal. ``keyframe_positions`` are 1-indexed positions into the
    recent-frame list that the model nominates as future keyframes (used by the
    learned-selection path; ignored in heuristic mode).

    Tries strict ``json.loads`` first, then falls back to regex extraction so a
    malformed or truncated generation still yields a usable subtask rather than
    raising mid-rollout. Returns ``("", [])`` only if no subtask can be recovered
    — the caller treats an empty subtask as a degenerate candidate.
    """
    text = text.strip()
    # Strip a leading ```json fence if the model emits one.
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]

    # Strict parse first.
    try:
        obj = json.loads(text)
        subtask = str(obj.get("current_subtask", "")).strip()
        kf = obj.get("keyframe_positions", []) or []
        positions = [int(x) for x in kf if isinstance(x, (int, float)) or str(x).strip().lstrip("-").isdigit()]
        if subtask:
            return subtask, positions
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: pull fields out with regex.
    subtask = ""
    m = _SUBTASK_RE.search(text)
    if m:
        # Unescape the captured JSON string body.
        subtask = m.group(1).encode().decode("unicode_escape").strip()
    positions: List[int] = []
    km = _KEYFRAMES_RE.search(text)
    if km:
        positions = [int(tok) for tok in re.findall(r"-?\d+", km.group(1))]
    return subtask, positions


__all__ = [
    "SUBGOAL_SYSTEM_PROMPT",
    "SELECT_SYSTEM_PROMPT",
    "build_user_prompt",
    "build_messages",
    "parse_subgoal_output",
]
