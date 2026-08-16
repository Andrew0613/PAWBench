"""Load, validate, and build video submissions.

A submission is the fixed denominator made explicit: one item per
(scene, repeat) slot of the declared scope, where a slot is either a
produced video or a declared failure.
"""

from __future__ import annotations

import json
from pathlib import Path

from pawbench.benchmark import Benchmark
from pawbench.submission._build import build as _build
from pawbench.submission._load import read_envelope
from pawbench.submission._model import (
    DEFAULT_NAMING,
    INFRASTRUCTURE_FAILURE_CODES,
    MODEL_FAILURE_CODES,
    VIDEO_EXTENSIONS,
    Item,
    ItemStatus,
    Submission,
)
from pawbench.submission._validate import validate

__all__ = [
    "DEFAULT_NAMING",
    "INFRASTRUCTURE_FAILURE_CODES",
    "Item",
    "ItemStatus",
    "MODEL_FAILURE_CODES",
    "Submission",
    "VIDEO_EXTENSIONS",
    "build",
    "load",
    "write",
]


def load(path: str | Path, *, benchmark: Benchmark) -> Submission:
    """Load and validate the submission manifest at file ``path``.

    Raises :class:`~pawbench.errors.SubmissionError` listing every problem
    found, including any missing grid slots.
    """
    envelope, problems = read_envelope(path)
    if envelope is None:
        from pawbench.errors import SubmissionError

        raise SubmissionError(problems)
    return validate(envelope, benchmark, root=str(path))


def build(
    benchmark: Benchmark,
    videos_dir: str | Path,
    *,
    system: str,
    prompt_set: str = "gt_guided",
    splits=("calibration", "coverage"),
    scene_ids: list[str] | None = None,
    repeats_per_scene: int | None = None,
    submission_id: str | None = None,
    naming_pattern: str = DEFAULT_NAMING,
    missing_policy: str = "error",
) -> Submission:
    """Build a Submission from a directory of convention-named videos.

    See :func:`pawbench.submission._build.build` (same signature) for the
    exact scanning and missing-slot policy.
    """
    return _build(
        benchmark,
        videos_dir,
        system=system,
        prompt_set=prompt_set,
        splits=splits,
        scene_ids=scene_ids,
        repeats_per_scene=repeats_per_scene,
        submission_id=submission_id,
        naming_pattern=naming_pattern,
        missing_policy=missing_policy,
    )


def write(sub: Submission, path: str | Path) -> None:
    """Write the submission manifest as JSON to file ``path``.

    Output is deterministic: identical submissions serialize to identical
    bytes. Paths are recorded relative to the manifest's directory.
    """
    Path(path).write_text(
        json.dumps(sub.to_envelope(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
