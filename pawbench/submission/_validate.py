"""Invariants of a submission manifest.

The load-bearing invariant is the grid: items must cover scope x repeats
exactly, with no duplicates and no missing slots. Missing generation results
are declared as failure items, never omitted.
"""

from __future__ import annotations

import re

from pawbench._paths import safe_relative_path_detail
from pawbench.benchmark import Benchmark
from pawbench.errors import SubmissionError, ValidationProblem
from pawbench.submission._model import (
    INFRASTRUCTURE_FAILURE_CODES,
    MODEL_FAILURE_CODES,
    SCHEMA_VERSION,
    Item,
    Submission,
)

SUBMISSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

ENVELOPE_FIELDS = {
    "schema_version",
    "submission_id",
    "benchmark",
    "system",
    "prompt_set",
    "scope",
    "repeats_per_scene",
    "items",
}
BENCHMARK_BINDING_FIELDS = {"dataset_id", "benchmark_revision"}
SCOPE_FIELDS = {"splits", "scene_ids"}
PRODUCED_FIELDS = {"scene_id", "repeat_index", "status", "video_path"}
FAILURE_FIELDS = {"scene_id", "repeat_index", "status", "failure_code"}

MAX_SLOTS_IN_DETAIL = 10


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_str(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _unexpected(fields: set, allowed: set) -> list[str]:
    return sorted(str(field) for field in fields - allowed)


def _slots_detail(slots: list[tuple[str, int]]) -> str:
    names = [f"{scene_id} r{repeat}" for scene_id, repeat in slots]
    shown = ", ".join(names[:MAX_SLOTS_IN_DETAIL])
    if len(names) > MAX_SLOTS_IN_DETAIL:
        shown += f", ...and {len(names) - MAX_SLOTS_IN_DETAIL} more"
    return shown


def validate(envelope: dict, benchmark: Benchmark, root: str = "submission.json") -> Submission:
    """Validate an envelope against ``benchmark``; raise SubmissionError with
    every problem found, or return the typed model."""
    problems: list[ValidationProblem] = []

    unexpected = _unexpected(set(envelope), ENVELOPE_FIELDS)
    if unexpected:
        problems.append(
            ValidationProblem(root, f"unknown envelope field(s): {', '.join(unexpected)}")
        )

    if envelope.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            ValidationProblem(
                f"{root}: schema_version",
                f"expected {SCHEMA_VERSION!r}, got {envelope.get('schema_version')!r}",
            )
        )

    submission_id = envelope.get("submission_id")
    if not _nonempty_str(submission_id) or not SUBMISSION_ID_RE.fullmatch(submission_id):
        problems.append(
            ValidationProblem(
                f"{root}: submission_id", "must match ^[A-Za-z0-9][A-Za-z0-9_.-]*$"
            )
        )

    binding = envelope.get("benchmark")
    if not isinstance(binding, dict):
        problems.append(
            ValidationProblem(f"{root}: benchmark", "must be an object with the benchmark binding")
        )
    else:
        unexpected = _unexpected(set(binding), BENCHMARK_BINDING_FIELDS)
        if unexpected:
            problems.append(
                ValidationProblem(
                    f"{root}: benchmark", f"unknown field(s): {', '.join(unexpected)}"
                )
            )
        if binding.get("dataset_id") != benchmark.dataset_id:
            problems.append(
                ValidationProblem(
                    f"{root}: benchmark.dataset_id",
                    f"submission binds {binding.get('dataset_id')!r}, "
                    f"benchmark declares {benchmark.dataset_id!r}",
                )
            )
        declared_revision = binding.get("benchmark_revision")
        if not _is_int(declared_revision) or declared_revision != benchmark.benchmark_revision:
            problems.append(
                ValidationProblem(
                    f"{root}: benchmark.benchmark_revision",
                    f"submission binds {declared_revision!r}, "
                    f"benchmark declares {benchmark.benchmark_revision!r}",
                )
            )

    system = envelope.get("system")
    if not _nonempty_str(system):
        problems.append(ValidationProblem(f"{root}: system", "must be a non-empty string"))

    prompt_set = envelope.get("prompt_set")
    if not _nonempty_str(prompt_set):
        problems.append(ValidationProblem(f"{root}: prompt_set", "must be a non-empty string"))
    elif prompt_set not in benchmark.prompt_sets:
        problems.append(
            ValidationProblem(
                f"{root}: prompt_set",
                f"unknown prompt set {prompt_set!r} "
                f"(benchmark declares: {', '.join(sorted(benchmark.prompt_sets))})",
            )
        )

    scope = envelope.get("scope")
    scope_scenes: list[str] = []
    splits: tuple[str, ...] = ()
    if not isinstance(scope, dict):
        problems.append(ValidationProblem(f"{root}: scope", "must be an object"))
    else:
        unexpected = _unexpected(set(scope), SCOPE_FIELDS)
        if unexpected:
            problems.append(
                ValidationProblem(f"{root}: scope", f"unknown field(s): {', '.join(unexpected)}")
            )
        raw_splits = scope.get("splits")
        if (
            not isinstance(raw_splits, list)
            or not raw_splits
            or len(set(raw_splits)) != len(raw_splits)
        ):
            problems.append(
                ValidationProblem(
                    f"{root}: scope.splits", "must be a non-empty list without duplicates"
                )
            )
        else:
            known = {scene.split for scene in benchmark.scenes}
            for split in raw_splits:
                if split not in known:
                    problems.append(
                        ValidationProblem(
                            f"{root}: scope.splits", f"unknown split {split!r}"
                        )
                    )
            if not any(s in known for s in raw_splits if isinstance(s, str)):
                problems.append(
                    ValidationProblem(
                        f"{root}: scope.splits",
                        "at least one split must select existing scenes",
                    )
                )
            splits = tuple(split for split in raw_splits if split in known)

        selected = set(splits)
        raw_scene_ids = scope.get("scene_ids")
        if raw_scene_ids is None:
            scope_scenes = [
                scene.scene_id
                for scene in benchmark.scenes
                if scene.split in selected
            ]
        elif isinstance(raw_scene_ids, list) and raw_scene_ids:
            if len(set(raw_scene_ids)) != len(raw_scene_ids):
                problems.append(
                    ValidationProblem(
                        f"{root}: scope.scene_ids", "must not contain duplicates"
                    )
                )
            for scene_id in raw_scene_ids:
                try:
                    scene = benchmark.scene_by_id(scene_id)
                except KeyError:
                    problems.append(
                        ValidationProblem(
                            f"{root}: scope.scene_ids",
                            f"unknown scene {scene_id!r} (not in benchmark)",
                        )
                    )
                    continue
                if scene.split not in selected:
                    problems.append(
                        ValidationProblem(
                            f"{root}: scope.scene_ids",
                            f"scene {scene_id!r} belongs to split {scene.split!r}, "
                            f"which is not selected in scope.splits",
                        )
                    )
                else:
                    scope_scenes.append(scene_id)
            # Preserve benchmark order regardless of the submitted order.
            benchmark_order = [s.scene_id for s in benchmark.scenes if s.split in selected]
            allowed = set(scope_scenes)
            scope_scenes = [scene_id for scene_id in benchmark_order if scene_id in allowed]
        else:
            problems.append(
                ValidationProblem(
                    f"{root}: scope.scene_ids",
                    "must be absent/null (all scenes of the selected splits) "
                    "or a non-empty list",
                )
            )

    repeats = envelope.get("repeats_per_scene")
    if not _is_int(repeats) or repeats < 1:
        problems.append(
            ValidationProblem(f"{root}: repeats_per_scene", "must be an integer >= 1")
        )

    items = _check_items(envelope.get("items"), root, problems)

    # Items must belong to the declared scope; out-of-scope or unknown scenes
    # must not ride along silently.
    if scope_scenes:
        scope_set = set(scope_scenes)
        reported: set[str] = set()
        for item in items:
            if item.scene_id in reported:
                continue
            reported.add(item.scene_id)
            if item.scene_id not in scope_set:
                try:
                    scene = benchmark.scene_by_id(item.scene_id)
                    detail = (
                        f"item(s) for scene {item.scene_id!r} belong to split "
                        f"{scene.split!r}, which is not selected in scope.splits"
                    )
                except KeyError:
                    detail = f"unknown scene {item.scene_id!r} (not in benchmark)"
                problems.append(ValidationProblem(f"{root}: items", detail))

    # Grid: every expected slot present, nothing unexpected, no duplicates.
    if scope_scenes and _is_int(repeats) and repeats >= 1:
        expected = {
            (scene_id, repeat) for scene_id in scope_scenes for repeat in range(repeats)
        }
        declared = {item.key for item in items if _valid_item_key(item, scope_scenes)}
        missing = sorted(expected - declared)
        unexpected_slots = sorted(declared - expected)
        if missing:
            problems.append(
                ValidationProblem(
                    f"{root}: items",
                    f"{len(missing)} missing slot(s): {_slots_detail(missing)} — "
                    "declare them as failure items instead of omitting them",
                )
            )
        if unexpected_slots:
            problems.append(
                ValidationProblem(
                    f"{root}: items",
                    f"unexpected slot(s) outside scope x repeats: "
                    f"{_slots_detail(unexpected_slots)}",
                )
            )

    if problems:
        raise SubmissionError(problems)

    return Submission(
        submission_id=submission_id,  # type: ignore[arg-type]
        dataset_id=benchmark.dataset_id,
        benchmark_revision=benchmark.benchmark_revision,
        system=system,  # type: ignore[arg-type]
        prompt_set=prompt_set,  # type: ignore[arg-type]
        splits=splits,  # type: ignore[arg-type]
        scene_ids=tuple(scope_scenes),
        repeats_per_scene=repeats,  # type: ignore[arg-type]
        items=tuple(items),
    )


def _valid_item_key(item: Item, scope_scenes: list[str]) -> bool:
    return item.scene_id in scope_scenes


def _check_items(value, root: str, problems: list[ValidationProblem]) -> list[Item]:
    if not isinstance(value, list) or not value:
        problems.append(ValidationProblem(f"{root}: items", "must be a non-empty array"))
        return []

    items: list[Item] = []
    seen_keys: dict[tuple[str, int], int] = {}

    for index, record in enumerate(value):
        where = f"{root}: items[{index}]"
        if not isinstance(record, dict):
            problems.append(ValidationProblem(where, "must be an object"))
            continue

        scene_id = record.get("scene_id")
        scene_ok = _nonempty_str(scene_id)
        if not scene_ok:
            problems.append(ValidationProblem(f"{where}: scene_id", "must be a non-empty string"))

        repeat_index = record.get("repeat_index")
        if not _is_int(repeat_index) or repeat_index < 0:
            problems.append(
                ValidationProblem(f"{where}: repeat_index", "must be an integer >= 0")
            )
            repeat_index = None

        status = record.get("status")
        if status == "produced":
            unexpected = _unexpected(set(record), PRODUCED_FIELDS)
            if unexpected:
                problems.append(
                    ValidationProblem(where, f"unknown field(s): {', '.join(unexpected)}")
                )
            video_path = record.get("video_path")
            detail = safe_relative_path_detail(video_path)
            if detail is not None:
                problems.append(ValidationProblem(f"{where}: video_path", detail))
            if scene_ok and repeat_index is not None:
                items.append(
                    Item(
                        scene_id=scene_id,  # type: ignore[arg-type]
                        repeat_index=repeat_index,
                        status="produced",
                        video_path=video_path if isinstance(video_path, str) else None,
                    )
                )
        elif status in ("model_failure", "infrastructure_failure"):
            unexpected = _unexpected(set(record), FAILURE_FIELDS)
            if unexpected:
                problems.append(
                    ValidationProblem(where, f"unknown field(s): {', '.join(unexpected)}")
                )
            code = record.get("failure_code")
            allowed = (
                MODEL_FAILURE_CODES
                if status == "model_failure"
                else INFRASTRUCTURE_FAILURE_CODES
            )
            if code not in allowed:
                problems.append(
                    ValidationProblem(
                        f"{where}: failure_code",
                        f"for status {status!r} must be one of {sorted(allowed)}, got {code!r}",
                    )
                )
            if scene_ok and repeat_index is not None:
                items.append(
                    Item(
                        scene_id=scene_id,  # type: ignore[arg-type]
                        repeat_index=repeat_index,
                        status=status,
                        failure_code=code if isinstance(code, str) else None,
                    )
                )
        else:
            problems.append(
                ValidationProblem(
                    f"{where}: status",
                    f"must be 'produced', 'model_failure', or "
                    f"'infrastructure_failure', got {status!r}",
                )
            )

        if scene_ok and repeat_index is not None:
            key = (scene_id, repeat_index)
            if key in seen_keys:
                problems.append(
                    ValidationProblem(
                        where,
                        f"duplicate slot {scene_id} r{repeat_index} "
                        f"(first seen at items[{seen_keys[key]}])",
                    )
                )
            else:
                seen_keys[key] = index

    return items
