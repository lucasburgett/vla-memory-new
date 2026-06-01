"""Tests for the MemER JSON output parser + multi-image prompt construction.

The VLM emits ``{"current_subtask": "...", "keyframe_positions": [...]}``; the
``current_subtask`` is what we hand to the GroundSG π0.5. Parsing must survive
trailing tokens, truncation, and ``json`` fences without crashing a rollout, and
the prompt's ``<image>`` count must equal the number of frames the processor is
given (the prompt-parity / vision-token-alignment invariant). See
``vla_memory.qwen_subgoal.prompts``.
"""

from vla_memory.qwen_subgoal.prompts import (
    build_user_prompt,
    parse_subgoal_output,
    _wrap_images,
)


def test_parses_clean_grounded_json():
    t = '{"current_subtask": "pick up the container at <box>120,40,200,180</box> that hides the red cube", "keyframe_positions": [3]}'
    sub, kf = parse_subgoal_output(t)
    assert "red cube" in sub and "<box>" in sub
    assert kf == [3]


def test_ignores_trailing_tokens_after_json():
    # Model rambled past <|im_end|>; the strict object still parses.
    t = '{"current_subtask": "press the button", "keyframe_positions": []} and then some'
    sub, kf = parse_subgoal_output(t)
    assert sub == "press the button"
    assert kf == []


def test_recovers_from_truncated_json_via_regex():
    # Generation cut off mid keyframe list — recover the subtask AND the positions seen.
    t = '{"current_subtask": "put down the container", "keyframe_positions": [2'
    sub, kf = parse_subgoal_output(t)
    assert sub == "put down the container"
    assert kf == [2]


def test_strips_json_code_fence():
    t = '```json\n{"current_subtask": "go left", "keyframe_positions": [1, 4]}\n```'
    sub, kf = parse_subgoal_output(t)
    assert sub == "go left"
    assert kf == [1, 4]


def test_degenerate_output_yields_empty_subtask():
    # The v4 collapse mode ('Waitwaitwait...') has no recoverable subtask; the
    # trainer scores an empty subtask 0 without rolling out.
    sub, kf = parse_subgoal_output("Waitwaitwaitwait")
    assert sub == ""
    assert kf == []


def test_prompt_image_count_matches_frames():
    # The processor pairs each <image> placeholder positionally with one frame, so
    # the count must equal len(key_frames) + len(recent_frames).
    up = build_user_prompt("press then pick the red container", n_key_frames=3, n_recent_frames=2)
    assert up.count("<image>") == 5
    assert "<video>" not in up


def test_prompt_video_prefix_when_demo_present():
    up = build_user_prompt("g", n_key_frames=1, n_recent_frames=1, has_video_demo=True)
    assert up.count("<image>") == 2
    assert up.count("<video>") == 1


def test_wrap_images_formatting():
    assert _wrap_images(0) == "[]"
    assert _wrap_images(1) == "[<image>]"
    assert _wrap_images(3) == "[<image>, <image>, <image>]"
