"""Prompt templates for the Qwen3-VL subgoal predictor.

Must match the SFT dataset builder in
``robomme_policy_learning/src/mme_vla_suite/dataset_builder/build_vlm_subgoal_dataset_qwenvl.py``.
Any drift between train- and inference-time prompts silently degrades the policy.
"""

from __future__ import annotations

from typing import List, Optional

SYSTEM_PROMPTS = {
    "simple_subgoal": (
        "You are a helpful assistant to help guide the robot to complete the task "
        "by predicting a sequence of language subgoals"
    ),
    "grounded_subgoal": (
        "You are a helpful assistant to help guide the robot to complete the task "
        "by predicting a sequence of grounded language subgoals"
    ),
}


def _wrap_history(history: List[str]) -> str:
    return "; ".join(f"{i + 1}. {sg}" for i, sg in enumerate(history))


def build_user_prompt(
    task_goal: str,
    history_subgoals: List[str],
    subgoal_type: str = "simple_subgoal",
    has_video_demo: bool = False,
) -> str:
    """Build the user-turn prompt fed to Qwen3-VL.

    Args:
        task_goal: natural-language task description from ``env.reset().info``.
        history_subgoals: subgoals already issued this episode (deduped).
        subgoal_type: ``"simple_subgoal"`` or ``"grounded_subgoal"``.
        has_video_demo: True when the task ships a demonstration video prefix.

    Returns:
        The full user-turn string. Note that the trailing ``<image>`` token is
        a placeholder the Qwen processor replaces with the current observation.
    """
    if subgoal_type not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown subgoal_type: {subgoal_type}")

    video_prefix = "<video>" if has_video_demo else ""
    kind = "grounded language" if subgoal_type == "grounded_subgoal" else "language"

    if not history_subgoals:
        return (
            f"{video_prefix}The task goal is: {task_goal}\n"
            "This is the initial turn for prediction\n"
            f"<image>What's the next {kind} subgoal based on current observation?"
        )

    return (
        f"{video_prefix}The task goal is: {task_goal}\n"
        f"The history of previous predicted {kind} subgoals are: "
        f"{_wrap_history(history_subgoals)}\n"
        f"<image>What's the next {kind} subgoal based on current observation?"
    )


def build_messages(
    task_goal: str,
    history_subgoals: List[str],
    subgoal_type: str = "simple_subgoal",
    has_video_demo: bool = False,
) -> List[dict]:
    """Build the OpenAI-style messages array (system + user turn)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[subgoal_type]},
        {
            "role": "user",
            "content": build_user_prompt(
                task_goal=task_goal,
                history_subgoals=history_subgoals,
                subgoal_type=subgoal_type,
                has_video_demo=has_video_demo,
            ),
        },
    ]


__all__ = ["SYSTEM_PROMPTS", "build_user_prompt", "build_messages"]
