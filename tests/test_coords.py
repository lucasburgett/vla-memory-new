"""Round-trip tests for the Qwen ↔ GroundSG coordinate conversion.

A wrong conversion here is invisible: GroundSG would silently read a 0–1000 value
as a pixel, the GRPO reward would score the wrong container, and the run would look
like an algorithmic flatline. So both directions and the round-trip are pinned.
"""

from vla_memory.qwen_subgoal.coords import from_qwen_xy, to_qwen_xy


def test_known_example_matches_probe():
    # The cube-visibility probe: target container <y=85, x=155> px ≈ Qwen <605, 332>.
    assert to_qwen_xy("<85, 155>") == "<605, 332>"
    # Inverse lands back on the original pixel <y, x> (±1px rounding).
    assert from_qwen_xy("<605, 332>") == "<85, 155>"


def test_axis_order_swaps():
    # to_qwen: input is <y, x>; output must be <x, y> (x first). y=10,x=200 px.
    # xn = round(200/256*1000)=781, yn = round(10/256*1000)=39  → "<781, 39>".
    assert to_qwen_xy("<10, 200>") == "<781, 39>"
    # from_qwen: input <x, y>=<781, 39>; output <y, x> px → "<10, 200>".
    assert from_qwen_xy("<781, 39>") == "<10, 200>"


def test_round_trip_within_one_pixel():
    # from_qwen_xy(to_qwen_xy(<y,x>)) is identity to ±1px across the grid.
    worst = 0
    for y in range(0, 256, 7):
        for x in range(0, 256, 7):
            out = from_qwen_xy(to_qwen_xy(f"<{y}, {x}>"))
            yy, xx = out.strip("<>").split(",")
            worst = max(worst, abs(int(yy) - y), abs(int(xx) - x))
    assert worst <= 1, f"round-trip drifted {worst}px (>1)"


def test_full_subtask_string_preserves_prefix():
    s = "pick up the container at <607, 338>"
    assert from_qwen_xy(s) == "pick up the container at <87, 155>"


def test_noop_without_coordinate():
    assert from_qwen_xy("press the button") == "press the button"
    assert to_qwen_xy("press the button") == "press the button"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all coords tests passed")
