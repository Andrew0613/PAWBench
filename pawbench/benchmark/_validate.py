"""Invariants of a benchmark package.

Every check here is demanded by a downstream consumer (scoring needs exact
references and ordering, submission validation needs scene identity and
split membership, evaluation needs the vocabulary and prompts). Anything a
consumer does not need is deliberately not checked.

All problems are collected, then raised once as a single
:class:`~pawbench.errors.BenchmarkError`.
"""

from __future__ import annotations

import re
from fractions import Fraction

from pawbench.benchmark._load import RawPackage
from pawbench.benchmark._model import (
    Benchmark,
    Outcome,
    PromptSet,
    ReferenceCount,
    Scene,
    Split,
)
from pawbench.errors import BenchmarkError, ValidationProblem

SCHEMA_VERSION = "pawbench.benchmark_package/v1"
SPLITS: tuple[Split, ...] = ("calibration", "coverage")

MANIFEST_FIELDS = {
    "schema_version",
    "dataset_id",
    "benchmark_revision",
    "formal_repeats",
    "scene_table",
    "prompt_sets",
}
SCENE_FIELDS = {
    "scene_id",
    "split",
    "split_order",
    "action",
    "source_image_path",
    "base_prompt",
    "outcomes",
    "judge_notes",
}
OUTCOME_FIELDS = {"label", "judge_notes", "reference_count"}
PROMPT_SET_ENTRY_FIELDS = {"path"}
PROMPT_ROW_FIELDS = {"scene_id", "outcome_prompts"}
PROMPT_FIELDS = {"outcome_label", "prompt"}

DATASET_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
SCENE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*$")
PROMPT_SET_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
OUTCOME_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_ -]*$")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_str(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _safe_relative_path_detail(value: object) -> str | None:
    """Return a problem detail if ``value`` is not a safe package-relative path."""
    if not _nonempty_str(value):
        return "path must be a non-empty string"
    text: str = value  # type: ignore[assignment]
    if text.startswith("/"):
        return "path must be relative (found leading '/')"
    if "\\" in text:
        return "path must use POSIX separators (found '\\')"
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text):
        return "path must not contain control characters"
    first_segment = text.split("/")[0]
    if ":" in first_segment:
        return "path must not contain ':' in the first segment (URI scheme or drive letter)"
    for segment in text.split("/"):
        if segment == "":
            return "path must not contain empty segments ('//' or trailing '/')"
        if segment in (".", ".."):
            return f"path must not contain {segment!r} segments"
    return None


def _unexpected_fields(fields: set[str], allowed: set[str]) -> list[str]:
    return sorted(fields - allowed)


def validate(raw: RawPackage) -> Benchmark:
    """Validate a raw package and build the typed model.

    Raises :class:`BenchmarkError` listing every problem found.
    """
    problems: list[ValidationProblem] = list(raw.problems)
    manifest = raw.manifest
    if manifest is None:  # pragma: no cover - load() raises earlier
        raise BenchmarkError(problems)

    m = _check_manifest(manifest, problems)
    scene_table_name = m["scene_table"] or "scenes.jsonl"
    scene_records = _check_scenes(raw.scenes, scene_table_name, problems)
    prompt_data = _check_prompt_sets(raw.prompt_sets, scene_records, problems)

    if problems:
        raise BenchmarkError(problems)

    return _build_model(raw, m, scene_records, prompt_data)


def _check_manifest(manifest: dict, problems: list[ValidationProblem]) -> dict:
    where = "benchmark.json"

    unexpected = _unexpected_fields(set(manifest), MANIFEST_FIELDS)
    if unexpected:
        problems.append(
            ValidationProblem(where, f"unknown manifest field(s): {', '.join(unexpected)}")
        )

    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        problems.append(
            ValidationProblem(
                f"{where}: schema_version",
                f"expected {SCHEMA_VERSION!r}, got {version!r}",
            )
        )

    dataset_id = manifest.get("dataset_id")
    if not _nonempty_str(dataset_id) or not DATASET_ID_RE.fullmatch(dataset_id):
        problems.append(
            ValidationProblem(f"{where}: dataset_id", "must match ^[A-Za-z][A-Za-z0-9_-]*$")
        )

    revision = manifest.get("benchmark_revision")
    if not _is_int(revision) or revision < 1:
        problems.append(
            ValidationProblem(f"{where}: benchmark_revision", "must be an integer >= 1")
        )

    formal_repeats = manifest.get("formal_repeats")
    if formal_repeats is not None and (not _is_int(formal_repeats) or formal_repeats < 1):
        problems.append(
            ValidationProblem(f"{where}: formal_repeats", "must be an integer >= 1 when present")
        )

    scene_table = manifest.get("scene_table")
    scene_table_ok = False
    detail = _safe_relative_path_detail(scene_table)
    if detail is None:
        scene_table_ok = True
    else:
        problems.append(ValidationProblem(f"{where}: scene_table", detail))

    specs: dict[str, str] = {}
    declared = manifest.get("prompt_sets")
    if not isinstance(declared, dict) or not declared:
        problems.append(
            ValidationProblem(f"{where}: prompt_sets", "must be an object with at least one entry")
        )
    else:
        for name, entry in declared.items():
            entry_where = f"{where}: prompt_sets.{name}"
            if not isinstance(name, str) or not PROMPT_SET_NAME_RE.fullmatch(name):
                problems.append(
                    ValidationProblem(entry_where, "prompt set name must match ^[a-z][a-z0-9_]*$")
                )
            if not isinstance(entry, dict):
                problems.append(
                    ValidationProblem(entry_where, "prompt set entry must be an object")
                )
                continue
            unexpected = _unexpected_fields(set(entry), PROMPT_SET_ENTRY_FIELDS)
            if unexpected:
                problems.append(
                    ValidationProblem(entry_where, f"unknown field(s): {', '.join(unexpected)}")
                )
            path = entry.get("path")
            detail = _safe_relative_path_detail(path)
            if detail is None:
                specs[name] = path
            else:
                problems.append(ValidationProblem(f"{entry_where}.path", detail))

    return {
        "dataset_id": dataset_id if _nonempty_str(dataset_id) else "",
        "benchmark_revision": revision if _is_int(revision) else 0,
        "formal_repeats": formal_repeats if _is_int(formal_repeats) else None,
        "scene_table": scene_table if scene_table_ok else None,
        "prompt_set_specs": specs,
    }


def _check_scenes(rows, scene_table_name: str, problems) -> list[tuple[int, dict, list]]:
    """Per-line structural checks plus cross-scene uniqueness.

    Returns one record per valid row: ``(line, scene_dict, parsed_outcomes)``
    where ``parsed_outcomes`` is a list of ``(label, judge_notes, ReferenceCount | None)``.
    Rows that fail structurally are skipped for cross-checks but still counted
    in ``problems``.
    """
    records: list[tuple[int, dict, list]] = []
    seen_ids: dict[str, int] = {}
    seen_orders: dict[tuple[str, int], int] = {}

    if not rows:
        problems.append(
            ValidationProblem(scene_table_name, "scene table contains no scenes")
        )
        return records

    for line_number, row in rows:
        where = f"{scene_table_name} line {line_number}"
        unexpected = _unexpected_fields(set(row), SCENE_FIELDS)
        if unexpected:
            problems.append(
                ValidationProblem(where, f"unknown scene field(s): {', '.join(unexpected)}")
            )

        scene_id = row.get("scene_id")
        scene_id_ok = _nonempty_str(scene_id) and bool(SCENE_ID_RE.fullmatch(scene_id))
        if not scene_id_ok:
            problems.append(
                ValidationProblem(f"{where}: scene_id", "must match ^[A-Za-z][A-Za-z0-9-]*$")
            )
        elif scene_id in seen_ids:
            problems.append(
                ValidationProblem(
                    f"{where}: scene_id",
                    f"duplicate scene_id {scene_id!r} (first seen on line {seen_ids[scene_id]})",
                )
            )
        else:
            seen_ids[scene_id] = line_number

        split = row.get("split")
        split_ok = split in SPLITS
        if not split_ok:
            problems.append(
                ValidationProblem(f"{where}: split", f"must be one of {SPLITS}, got {split!r}")
            )

        split_order = row.get("split_order")
        order_ok = _is_int(split_order) and split_order >= 0
        if not order_ok:
            problems.append(
                ValidationProblem(f"{where}: split_order", "must be an integer >= 0")
            )
        elif split_ok and scene_id_ok:
            key = (split, split_order)
            if key in seen_orders:
                problems.append(
                    ValidationProblem(
                        f"{where}: split_order",
                        f"duplicate split_order {split_order} in split {split!r} "
                        f"(first seen on line {seen_orders[key]})",
                    )
                )
            else:
                seen_orders[key] = line_number

        for field in ("action", "base_prompt"):
            if not _nonempty_str(row.get(field)):
                problems.append(
                    ValidationProblem(f"{where}: {field}", "must be a non-empty string")
                )

        detail = _safe_relative_path_detail(row.get("source_image_path"))
        if detail is not None:
            problems.append(ValidationProblem(f"{where}: source_image_path", detail))

        judge_notes = row.get("judge_notes")
        if judge_notes is not None and not isinstance(judge_notes, str):
            problems.append(
                ValidationProblem(f"{where}: judge_notes", "must be a string when present")
            )

        parsed_outcomes = _check_outcomes(row.get("outcomes"), where, split, problems)

        if scene_id_ok and split_ok and order_ok:
            records.append((line_number, row, parsed_outcomes))

    return records


def _check_outcomes(value, where: str, split, problems) -> list:
    """Validate the outcome list of one scene; returns parsed outcomes."""
    where = f"{where}: outcomes"
    if not isinstance(value, list) or not value:
        problems.append(ValidationProblem(where, "must be a non-empty array"))
        return []

    seen_labels: set[str] = set()
    parsed: list[tuple[str, str | None, ReferenceCount | None]] = []
    counts_valid: list[bool] = []

    for index, outcome in enumerate(value):
        item_where = f"{where}[{index}]"
        if not isinstance(outcome, dict):
            problems.append(ValidationProblem(item_where, "must be an object"))
            continue
        unexpected = _unexpected_fields(set(outcome), OUTCOME_FIELDS)
        if unexpected:
            problems.append(
                ValidationProblem(item_where, f"unknown field(s): {', '.join(unexpected)}")
            )

        label = outcome.get("label")
        label_ok = _nonempty_str(label) and bool(OUTCOME_LABEL_RE.fullmatch(label))
        if not label_ok:
            problems.append(
                ValidationProblem(f"{item_where}: label", "must match ^[a-z0-9][a-z0-9_ -]*$")
            )
        elif label in seen_labels:
            problems.append(
                ValidationProblem(f"{item_where}: label", f"duplicate outcome label {label!r}")
            )
        else:
            seen_labels.add(label)

        judge_notes = outcome.get("judge_notes")
        if judge_notes is not None and not isinstance(judge_notes, str):
            problems.append(
                ValidationProblem(
                    f"{item_where}: judge_notes", "must be a string when present"
                )
            )

        raw_count = outcome.get("reference_count")
        if split == "calibration":
            if not isinstance(raw_count, dict):
                problems.append(
                    ValidationProblem(
                        f"{item_where}: reference_count",
                        "required for calibration outcomes",
                    )
                )
                parsed.append((label if label_ok else "", judge_notes, None))
                counts_valid.append(False)
            else:
                unexpected = _unexpected_fields(set(raw_count), {"numerator", "denominator"})
                if unexpected:
                    problems.append(
                        ValidationProblem(
                            f"{item_where}: reference_count",
                            f"unknown field(s): {', '.join(unexpected)}",
                        )
                    )
                numerator = raw_count.get("numerator")
                denominator = raw_count.get("denominator")
                if not (_is_int(numerator) and numerator >= 1):
                    problems.append(
                        ValidationProblem(
                            f"{item_where}: reference_count.numerator", "must be an integer >= 1"
                        )
                    )
                if not (_is_int(denominator) and denominator >= 1):
                    problems.append(
                        ValidationProblem(
                            f"{item_where}: reference_count.denominator", "must be an integer >= 1"
                        )
                    )
                ok = (
                    _is_int(numerator)
                    and _is_int(denominator)
                    and numerator >= 1
                    and denominator >= 1
                )
                parsed.append(
                    (
                        label if label_ok else "",
                        judge_notes,
                        ReferenceCount(numerator, denominator) if ok else None,
                    )
                )
                counts_valid.append(ok)
        else:  # coverage
            if raw_count is not None:
                problems.append(
                    ValidationProblem(
                        f"{item_where}: reference_count",
                        "forbidden for coverage outcomes (no published reference)",
                    )
                )
            parsed.append((label if label_ok else "", judge_notes, None))
            counts_valid.append(True)

    if split == "calibration" and counts_valid and all(counts_valid):
        total = Fraction(0)
        for _, _, count in parsed:
            if count is not None:
                total += count.fraction
        if total != 1:
            problems.append(
                ValidationProblem(
                    where,
                    f"calibration reference counts must sum to exactly 1 (got {total})",
                )
            )

    return parsed


def _check_prompt_sets(
    prompt_sets, scene_records: list[tuple[int, dict, list]], problems
) -> dict[str, dict[str, dict[str, str]]]:
    """Validate prompt-set files against the scene table; returns raw rows by set."""
    scene_labels: dict[str, set[str]] = {}
    for _, row, parsed in scene_records:
        scene_labels[row["scene_id"]] = {label for label, _, _ in parsed if label}

    data: dict[str, dict[str, dict[str, str]]] = {}
    seen_set_names: set[str] = set()

    for name, rows in prompt_sets:
        if name in seen_set_names:
            continue  # duplicate manifest keys are impossible; defensive only
        seen_set_names.add(name)
        rows_by_scene: dict[str, dict[str, str]] = {}
        seen_scene_rows: dict[str, int] = {}

        for line_number, row in rows:
            where = f"prompt set {name!r} line {line_number}"
            unexpected = _unexpected_fields(set(row), PROMPT_ROW_FIELDS)
            if unexpected:
                problems.append(
                    ValidationProblem(where, f"unknown field(s): {', '.join(unexpected)}")
                )

            scene_id = row.get("scene_id")
            if not _nonempty_str(scene_id):
                problems.append(
                    ValidationProblem(f"{where}: scene_id", "must be a non-empty string")
                )
                continue
            if scene_id in seen_scene_rows:
                problems.append(
                    ValidationProblem(
                        f"{where}: scene_id",
                        f"duplicate prompt row for scene {scene_id!r} "
                        f"(first seen on line {seen_scene_rows[scene_id]})",
                    )
                )
                continue
            seen_scene_rows[scene_id] = line_number

            prompts = row.get("outcome_prompts")
            if not isinstance(prompts, list) or not prompts:
                problems.append(
                    ValidationProblem(f"{where}: outcome_prompts", "must be a non-empty array")
                )
                continue

            labels: dict[str, str] = {}
            for index, prompt in enumerate(prompts):
                item_where = f"{where}: outcome_prompts[{index}]"
                if not isinstance(prompt, dict):
                    problems.append(ValidationProblem(item_where, "must be an object"))
                    continue
                unexpected = _unexpected_fields(set(prompt), PROMPT_FIELDS)
                if unexpected:
                    problems.append(
                        ValidationProblem(item_where, f"unknown field(s): {', '.join(unexpected)}")
                    )
                label = prompt.get("outcome_label")
                text = prompt.get("prompt")
                if not _nonempty_str(label):
                    problems.append(
                        ValidationProblem(
                            f"{item_where}: outcome_label", "must be a non-empty string"
                        )
                    )
                    continue
                if label in labels:
                    problems.append(
                        ValidationProblem(
                            f"{item_where}: outcome_label", f"duplicate prompt label {label!r}"
                        )
                    )
                    continue
                if not _nonempty_str(text):
                    problems.append(
                        ValidationProblem(f"{item_where}: prompt", "must be a non-empty string")
                    )
                    continue
                labels[label] = text
            rows_by_scene[scene_id] = labels

            known = scene_labels.get(scene_id)
            if known is None:
                problems.append(
                    ValidationProblem(
                        f"{where}: scene_id", f"unknown scene {scene_id!r} (not in scene table)"
                    )
                )
            else:
                prompt_labels = set(labels)
                missing = sorted(known - prompt_labels)
                extra = sorted(prompt_labels - known)
                if missing:
                    problems.append(
                        ValidationProblem(
                            f"{where}: outcome_prompts",
                            f"missing prompt(s) for outcome(s): {', '.join(missing)}",
                        )
                    )
                if extra:
                    problems.append(
                        ValidationProblem(
                            f"{where}: outcome_prompts",
                            f"prompt label(s) not in scene vocabulary: {', '.join(extra)}",
                        )
                    )

        missing_scenes = sorted(set(scene_labels) - set(rows_by_scene))
        if missing_scenes:
            problems.append(
                ValidationProblem(
                    f"prompt set {name!r}",
                    f"scene(s) without prompt rows: {', '.join(missing_scenes)}",
                )
            )
        data[name] = rows_by_scene

    return data


def _build_model(
    raw: RawPackage, m: dict, scene_records: list[tuple[int, dict, list]], prompt_data
) -> Benchmark:
    scenes = tuple(
        Scene(
            scene_id=row["scene_id"],
            split=row["split"],
            split_order=row["split_order"],
            action=row["action"],
            base_prompt=row["base_prompt"],
            source_image_path=row["source_image_path"],
            judge_notes=row.get("judge_notes"),
            outcomes=tuple(
                Outcome(label=label, judge_notes=notes, reference_count=count)
                for label, notes, count in parsed
            ),
        )
        for _, row, parsed in scene_records
    )
    prompt_sets = {
        name: PromptSet(name=name, prompts_by_scene=rows) for name, rows in prompt_data.items()
    }
    return Benchmark(
        dataset_id=m["dataset_id"],
        benchmark_revision=m["benchmark_revision"],
        root=raw.root,
        scenes=scenes,
        prompt_sets=prompt_sets,
        formal_repeats=m["formal_repeats"],
    )
