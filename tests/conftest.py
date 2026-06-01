"""Make the ``src/`` layout importable under pytest.

Mirrors ``src/vla_memory/grpo/main.py``'s ``sys.path.insert(0, ".../src")`` so
``from vla_memory...`` resolves without an editable install. Importing
``vla_memory.qwen_subgoal.model`` is light: ``transformers`` / ``peft`` are
imported lazily inside ``QwenSubgoalPolicy.__init__``, not at module load.
"""

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
