"""Unit tests for the SFT-collapse detector.

Guards the heuristic that flags a degenerate subgoal adapter (the v4 failure:
greedy decode collapses into 'Waitwaitwait...'). See
``project_sft_v4_adapter_degenerate``.
"""

import torch

from vla_memory.qwen_subgoal.model import _looks_degenerate

MAXNT = 64


def test_healthy_short_subgoal_is_not_degenerate():
    # "Pick up the green cube." — short, distinct tokens, terminated.
    ids = torch.tensor([55, 12, 7, 901, 1234, 13])
    assert _looks_degenerate(ids, MAXNT) is False


def test_single_token_repeat_is_degenerate():
    # 'Waitwaitwait...' — one token, full budget.
    ids = torch.full((MAXNT,), 9999, dtype=torch.long)
    assert _looks_degenerate(ids, MAXNT) is True


def test_dominant_token_below_budget_is_degenerate():
    # Terminated early but one token is >half the output → still collapsed.
    ids = torch.tensor([9999, 9999, 9999, 9999, 9999, 7])  # 5/6 same
    assert _looks_degenerate(ids, MAXNT) is True


def test_hitting_max_budget_is_degenerate_even_if_varied():
    # Never emitted EOS (ran the whole budget) → degenerate regardless of variety.
    ids = torch.arange(MAXNT)  # all distinct, but length == budget
    assert _looks_degenerate(ids, MAXNT) is True


def test_empty_generation_is_degenerate():
    assert _looks_degenerate(torch.tensor([], dtype=torch.long), MAXNT) is True


def test_short_varied_subgoal_with_one_repeat_is_ok():
    # A normal subgoal can repeat a common token ("the") without collapsing.
    ids = torch.tensor([55, 12, 7, 901, 7, 1234, 13])  # '7' twice of 7 → 2/7 < 0.5
    assert _looks_degenerate(ids, MAXNT) is False
