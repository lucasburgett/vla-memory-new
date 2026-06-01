"""Regression guard for the v1 "generations never terminate" bug.

The SFT'd Qwen3-VL ends its turn with ``<|im_end|>`` (151645), which is not
necessarily the tokenizer's nominal ``eos_token_id`` (``<|endoftext|>``, 151643).
``sample_subgoals`` must trim on the FULL stop set; trimming on a single id let
``<|im_end|>`` plus trailing rambling survive, producing 64-token subgoals that
decorrelated the GRPO gradient. See project memory
``project_grpo_v1_generation_no_eos``.
"""

import torch

from vla_memory.qwen_subgoal.model import _first_stop_index

IM_END = 151645
ENDOFTEXT = 151643


def test_trims_at_im_end_even_when_not_first_in_set():
    # The model stopped on <|im_end|>; the nominal eos is <|endoftext|>.
    ids = torch.tensor([10, 11, IM_END, 12, 13])
    assert _first_stop_index(ids, [ENDOFTEXT, IM_END]) == 2


def test_returns_first_stop_when_multiple_present():
    ids = torch.tensor([10, ENDOFTEXT, IM_END, 11])
    assert _first_stop_index(ids, [IM_END, ENDOFTEXT]) == 1


def test_returns_len_when_no_stop_token():
    # The exact failure window of the bug: no stop in the span → keep all of it.
    # The upstream fix (passing eos to generate) is what keeps this off the
    # common path; the trim must still degrade to len() rather than error.
    ids = torch.tensor([10, 11, 12, 13])
    assert _first_stop_index(ids, [IM_END, ENDOFTEXT]) == 4


def test_empty_stop_set_returns_len():
    ids = torch.tensor([10, 11])
    assert _first_stop_index(ids, []) == 2


def test_stop_at_position_zero():
    # Empty generation (immediate stop) → cut at 0, yielding a 0-length subgoal
    # that the trainer's empty-candidate guard then skips.
    ids = torch.tensor([IM_END, 10, 11])
    assert _first_stop_index(ids, [IM_END]) == 0
