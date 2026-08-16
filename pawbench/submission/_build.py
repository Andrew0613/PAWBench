"""Build a submission from a directory of generated videos.

This is the adapter between a submitter's real world — a folder of files
with a naming convention and some missing slots — and the grid discipline
of the submission contract. All the messiness (naming, extensions, missing
files, stale files) is absorbed here so callers do not have to deal with it.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

from pawbench.benchmark import Benchmark, Split
from pawbench.errors import SubmissionError, ValidationProblem
from pawbench.submission._model import (
    DEFAULT_NAMING,
    SCHEMA_VERSION,
    VIDEO_EXTENSIONS,
    Submission,
)
from pawbench.submission._validate import validate

_STALE_NAME_RE = re.compile(
    r"^(?P<scene_id>.+)__r(?P<repeat>\d{3,})(?P<ext>\.mp4|\.webm|\.mov)$"
)


def _slugify(text: str) -> str:
    """Turn a free-form system name into a submission_id-shaped slug."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-.")
    return slug or "submission"


def _naming_pattern_detail(pattern: str) -> str | None:
    """Return a problem detail unless the pattern has exactly the two slots."""
    try:
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(pattern)
            if field_name is not None
        }
    except ValueError as exc:
        return f"invalid format pattern: {exc}"
    if fields != {"scene_id", "repeat_index"}:
        return "pattern must reference exactly {scene_id} and {repeat_index}"
    return None


def build(
    benchmark: Benchmark,
    videos_dir: str | Path,
    *,
    system: str,
    prompt_set: str = "gt_guided",
    splits: tuple[Split, ...] = ("calibration", "coverage"),
    scene_ids: list[str] | None = None,
    repeats_per_scene: int | None = None,
    submission_id: str | None = None,
    naming_pattern: str = DEFAULT_NAMING,
    missing_policy: str = "error",
) -> Submission:
    """Scan ``videos_dir`` for convention-named videos and build a Submission.

    ``videos_dir`` is scanned non-recursively; each expected slot looks for
    ``naming_pattern`` + a known video extension (in a fixed order). Write the
    manifest into the same directory (``submission.write(sub, videos_dir /
    "submission.json")``) so the recorded relative paths resolve.

    Missing slots follow ``missing_policy``: ``"error"`` (default) refuses to
    guess and raises with every missing slot listed; ``"model_failure"``
    registers each missing slot as a ``generation_failed`` item so the
    denominator stays complete.
    """
    videos_dir = Path(videos_dir)
    problems: list[ValidationProblem] = []

    if missing_policy not in ("error", "model_failure"):
        problems.append(
            ValidationProblem(
                "build: missing_policy",
                f"must be 'error' or 'model_failure', got {missing_policy!r}",
            )
        )

    detail = _naming_pattern_detail(naming_pattern)
    if detail is not None:
        problems.append(ValidationProblem("build: naming_pattern", detail))

    if repeats_per_scene is None:
        repeats_per_scene = benchmark.formal_repeats
    if repeats_per_scene is None:
        problems.append(
            ValidationProblem(
                "build: repeats_per_scene",
                "no value given and the benchmark declares no formal_repeats",
            )
        )

    if not videos_dir.is_dir():
        problems.append(ValidationProblem(str(videos_dir), "videos_dir is not a directory"))

    if problems:
        raise SubmissionError(problems)

    scope_scene_ids = [
        scene.scene_id
        for scene in benchmark.scenes
        if scene.split in set(splits)
        and (scene_ids is None or scene.scene_id in set(scene_ids))
    ]

    items: list[dict] = []
    expected_names: dict[str, tuple[str, int]] = {}
    for scene_id in scope_scene_ids:
        for repeat in range(repeats_per_scene):
            for extension in VIDEO_EXTENSIONS:
                name = naming_pattern.format(scene_id=scene_id, repeat_index=repeat) + extension
                expected_names[name] = (scene_id, repeat)

    present = {entry.name for entry in videos_dir.iterdir()}
    found_slots: set[tuple[str, int]] = set()
    for name, slot in expected_names.items():
        if name in present:
            found_slots.add(slot)
            items.append(
                {
                    "scene_id": slot[0],
                    "repeat_index": slot[1],
                    "status": "produced",
                    "video_path": name,
                }
            )

    # Convention-shaped files naming a scene that is not in the benchmark at
    # all are almost always typos; they must not vanish silently. Files for
    # real scenes outside the selected scope are the user's own business.
    benchmark_scene_ids = {scene.scene_id for scene in benchmark.scenes}
    for entry_name in sorted(present):
        if entry_name in expected_names:
            continue
        match = _STALE_NAME_RE.fullmatch(entry_name)
        if match and match.group("scene_id") not in benchmark_scene_ids:
            problems.append(
                ValidationProblem(
                    f"{videos_dir}: {entry_name}",
                    "matches the naming convention but names a scene that is "
                    "not in the benchmark (typo or stale run)",
                )
            )

    for scene_id in scope_scene_ids:
        for repeat in range(repeats_per_scene):
            if (scene_id, repeat) in found_slots:
                continue
            if missing_policy == "model_failure":
                items.append(
                    {
                        "scene_id": scene_id,
                        "repeat_index": repeat,
                        "status": "model_failure",
                        "failure_code": "generation_failed",
                    }
                )
            else:
                problems.append(
                    ValidationProblem(
                        "build",
                        f"missing video for slot {scene_id} r{repeat} "
                        f"({naming_pattern.format(scene_id=scene_id, repeat_index=repeat)}"
                        "+ .mp4/.webm/.mov); pass missing_policy='model_failure' to "
                        "register missing slots as generation failures",
                    )
                )

    if problems:
        raise SubmissionError(problems)

    scope_payload: dict = {"splits": list(splits)}
    if scene_ids is not None:
        scope_payload["scene_ids"] = list(scene_ids)

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "submission_id": submission_id if submission_id is not None else _slugify(system),
        "benchmark": {
            "dataset_id": benchmark.dataset_id,
            "benchmark_revision": benchmark.benchmark_revision,
        },
        "system": system,
        "prompt_set": prompt_set,
        "scope": scope_payload,
        "repeats_per_scene": repeats_per_scene,
        "items": items,
    }
    # Self-check: a built submission always satisfies the load-time contract.
    return validate(envelope, benchmark, root="build")
