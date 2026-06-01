"""Qwen3-VL subgoal predictor — model + prompts.

The model class is intentionally NOT re-exported here: importing it would pull
in torch + transformers, which we don't want on CPU dev machines. Import it
directly when you need it::

    from vla_memory.qwen_subgoal.model import QwenSubgoalPolicy
"""

from .prompts import (
    SUBGOAL_SYSTEM_PROMPT,
    build_messages,
    build_user_prompt,
    parse_subgoal_output,
)

__all__ = [
    "SUBGOAL_SYSTEM_PROMPT",
    "build_messages",
    "build_user_prompt",
    "parse_subgoal_output",
]
