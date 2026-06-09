"""Schema separation for the joint select-then-use pipeline.

The SELECT and USE calls share near-identical IMAGE inputs, so the only thing that
keeps the model from collapsing `keyframe_positions` to the USE-majority `[]` is a
DISTINCT prompt + system prompt + target schema. These tests pin:

1. the USE prompt is BYTE-IDENTICAL to what the deployed SFT ckpt was trained on
   (prompt-parity invariant — drift here silently degrades the policy);
2. SELECT differs from USE in system prompt, user prompt, and target field set;
3. `parse_subgoal_output` handles a SELECT target (keyframe_positions only).
"""

import json

from vla_memory.qwen_subgoal.prompts import (
    SELECT_SYSTEM_PROMPT,
    SUBGOAL_SYSTEM_PROMPT,
    build_messages,
    build_user_prompt,
    parse_subgoal_output,
)

_TASK = "put the red cube in the bin"


def test_use_prompt_is_byte_identical_parity_anchor():
    # The exact string the one-shot SFT (ckpt-150) trains/infers on. If this ever
    # changes, every prior checkpoint is off-distribution — update deliberately.
    expected = (
        "The task goal is: put the red cube in the bin\n"
        "The subtasks already completed are: 1. press the button\n"
        "Here are the selected frames from the entirety of the full execution that are of particular importance:"
        "[<image>, <image>, <image>, <image>]\n"
        "Here is current input image list from the front-view camera: [<image>, <image>]\n\n"
        "What subtask should the robot execute and what is the keyframe position?"
    )
    got = build_user_prompt(
        task_goal=_TASK, n_key_frames=4, n_recent_frames=2,
        history_subgoals=["press the button"],
    )
    assert got == expected
    # default mode is "use"
    assert build_user_prompt(_TASK, 4, 2, ["press the button"], mode="use") == expected


def test_select_prompt_and_system_differ_from_use():
    use_u = build_user_prompt(_TASK, 4, 2, ["press the button"], mode="use")
    sel_u = build_user_prompt(_TASK, 0, 12, [], mode="select")
    assert sel_u != use_u
    assert "keyframe_positions" in sel_u
    assert "What subtask should the robot execute" not in sel_u
    # SELECT shows the 12 candidate frames as the recent-image list, no past keyframes.
    assert sel_u.count("<image>") == 12
    assert "particular importance" not in sel_u
    assert SELECT_SYSTEM_PROMPT != SUBGOAL_SYSTEM_PROMPT
    assert build_messages(_TASK, 0, 12, [], mode="select")[0]["content"] == SELECT_SYSTEM_PROMPT
    assert build_messages(_TASK, 4, 2, ["x"])[0]["content"] == SUBGOAL_SYSTEM_PROMPT


def test_target_schemas_are_disjoint_on_the_contested_field():
    # USE target carries current_subtask (+ empty keyframe_positions); SELECT target
    # carries ONLY keyframe_positions. The contested field never gets two non-trivial
    # targets from the same prompt → no collapse.
    use_t = json.loads('{"current_subtask": "pick up the container at <605, 332>", "keyframe_positions": []}')
    sel_t = json.loads('{"keyframe_positions": [1, 2, 5]}')
    assert use_t["current_subtask"] and use_t["keyframe_positions"] == []
    assert "current_subtask" not in sel_t and sel_t["keyframe_positions"]


def test_parse_handles_select_target():
    sub, pos = parse_subgoal_output('{"keyframe_positions": [1, 3]}')
    assert sub == "" and pos == [1, 3]
    sub, pos = parse_subgoal_output('{"current_subtask": "pick up the container at <605, 332>", "keyframe_positions": []}')
    assert sub == "pick up the container at <605, 332>" and pos == []


def test_invalid_mode_raises():
    try:
        build_user_prompt(_TASK, 0, 1, mode="bogus")
    except ValueError:
        return
    raise AssertionError("build_user_prompt did not reject an invalid mode")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all select/use prompt tests passed")
