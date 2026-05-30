"""Thin wrapper that invokes the submodule's QwenVL dataset builder.

The actual builder lives at
``robomme_policy_learning/src/mme_vla_suite/dataset_builder/build_vlm_subgoal_dataset_qwenvl.py``.
We keep it in the submodule because regenerating the JSONL is closely tied to
how RoboMME records HDF5 episodes — duplicating that logic up here would
silently drift the moment the benchmark changes.

This wrapper just imports the builder, runs it from the project root with the
right paths, and writes its output into the Modal volume.
"""

from __future__ import annotations

import sys
from pathlib import Path


def build_sft_dataset(
    raw_data_path: str,
    preprocessed_data_path: str,
    max_episodes: int | None = None,
    visualize: bool = False,
) -> dict:
    """Run the submodule's QwenVL dataset builder.

    Returns a dict with output paths so the Modal entrypoint can attach them
    as run artifacts. Must be called from a Python process that has the
    submodule on its ``sys.path`` (Modal handles that in ``modal_train.py``).
    """
    # The submodule pip-installs ``mme_vla_suite`` into its venv, so once the
    # container is built this import is just a regular import.
    from mme_vla_suite.dataset_builder.build_vlm_subgoal_dataset_qwenvl import (
        DatasetBuilder,
    )

    builder = DatasetBuilder(
        raw_data_path=raw_data_path,
        preprocessed_data_path=preprocessed_data_path,
        max_episodes=max_episodes,
        visualize=visualize,
    )
    builder.run()

    # BaseVLMSubgoalDatasetBuilder writes JSONL + images under a
    # ``<preprocessed_data_path>/<vlm_dir_name>/`` subdirectory (default
    # ``vlm_subgoal``). Return the *actual* paths so callers don't have to
    # know about that detail.
    out_root = Path(preprocessed_data_path) / "vlm_subgoal"
    return {
        "simple_subgoal_train": str(out_root / "simple_subgoal_train.jsonl"),
        "grounded_subgoal_train": str(out_root / "grounded_subgoal_train.jsonl"),
        "images_dir": str(out_root / "images"),
    }


__all__ = ["build_sft_dataset"]
