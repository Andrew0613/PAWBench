"""Reconciliation of a result bundle against its benchmark and submission.

Trust here comes from recomputation (design force F3): the row set must
reproduce the submission's grid, statuses must be consistent with what each
side can know, and the summary must match what :func:`compute_rollups`
derives from the rows. A bundle that fails here was tampered with or
corrupted.
"""

from __future__ import annotations

import re

from pawbench.benchmark import Benchmark
from pawbench.errors import ResultError, ValidationProblem
from pawbench.results._model import (
    OUTCOME_READOUT_FIELDS,
    ROW_FAILURE_CODES,
    ROW_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    SUMMARY_STATUSES,
    TRUST_ACTION_FIELDS,
    TRUST_BLOCK_FIELDS,
    TRUSTWORTHINESS_FIELDS,
    DenominatorCounts,
    ResultRow,
    ResultSet,
    SceneRollup,
    SplitAggregate,
    Summary,
    compute_rollups,
    with_scene_scores,
)
from pawbench.submission import Submission

ROW_FIELDS = {
    "schema_version",
    "scene_id",
    "repeat_index",
    "split",
    "status",
    "failure_code",
    "outcome_readout",
    "trustworthiness_audit",
    "relaxed_included",
    "strict_included",
    "outcome_label",
}
SUMMARY_FIELDS = {
    "schema_version",
    "benchmark",
    "system",
    "prompt_set",
    "judge",
    "denominator",
    "status",
    "warning",
    "per_scene",
    "calibration",
    "coverage",
}
N_PASS_RE = re.compile(r"^[0-9]+/[0-9]+$")
MAX_SLOTS_IN_DETAIL = 10


def _unexpected(fields: set, allowed: set) -> list[str]:
    return sorted(str(field) for field in fields - allowed)


def _slots_detail(slots: list[tuple[str, int]]) -> str:
    names = [f"{scene_id} r{repeat}" for scene_id, repeat in slots]
    shown = ", ".join(names[:MAX_SLOTS_IN_DETAIL])
    if len(names) > MAX_SLOTS_IN_DETAIL:
        shown += f", ...and {len(names) - MAX_SLOTS_IN_DETAIL} more"
    return shown


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _payload_problems(payload: object, where: str, expected_fields: frozenset, problems) -> None:
    """Structural check of one axis payload; deep enums belong to the parser."""
    if not isinstance(payload, dict):
        problems.append(ValidationProblem(where, "must be an object"))
        return
    if set(payload) != expected_fields:
        problems.append(ValidationProblem(where, f"unexpected field set: {sorted(payload)}"))
        return
    if expected_fields is TRUSTWORTHINESS_FIELDS:
        for block_name in ("scene_grounding", "object_continuity", "physical_process"):
            if set(payload.get(block_name) or {}) != TRUST_BLOCK_FIELDS:
                problems.append(ValidationProblem(f"{where}.{block_name}", "unexpected field set"))
        if set(payload.get("action_execution") or {}) != TRUST_ACTION_FIELDS:
            problems.append(ValidationProblem(f"{where}.action_execution", "unexpected field set"))


def validate(
    rows: list[dict],
    summary: dict | None,
    benchmark: Benchmark,
    submission: Submission,
    problems: list[ValidationProblem] | None = None,
) -> ResultSet:
    """Reconcile a read bundle; raise ResultError with every problem found."""
    problems = problems if problems is not None else []
    parsed_rows = _check_rows(rows, benchmark, submission, problems)
    parsed_summary = _check_summary(summary, benchmark, submission, parsed_rows, problems)

    if problems:
        raise ResultError(problems)

    return ResultSet(rows=tuple(parsed_rows), summary=parsed_summary)


def _check_rows(
    rows: list[dict], benchmark: Benchmark, submission: Submission, problems
) -> list[ResultRow]:
    seen: dict[tuple[str, int], int] = {}
    parsed: list[ResultRow] = []

    for index, record in enumerate(rows):
        where = f"rows.jsonl item {index + 1}"
        unexpected = _unexpected(set(record), ROW_FIELDS)
        if unexpected:
            problems.append(ValidationProblem(where, f"unknown field(s): {', '.join(unexpected)}"))

        if record.get("schema_version") != ROW_SCHEMA_VERSION:
            problems.append(
                ValidationProblem(
                    f"{where}: schema_version",
                    f"expected {ROW_SCHEMA_VERSION!r}, got {record.get('schema_version')!r}",
                )
            )

        scene_id = record.get("scene_id")
        repeat_index = record.get("repeat_index")
        if not isinstance(scene_id, str) or not _is_int(repeat_index):
            problems.append(ValidationProblem(where, "scene_id/repeat_index malformed"))
            continue
        key = (scene_id, repeat_index)
        if key in seen:
            problems.append(
                ValidationProblem(
                    where,
                    f"duplicate slot {scene_id} r{repeat_index} (first at item {seen[key] + 1})",
                )
            )
            continue
        seen[key] = index

        try:
            scene = benchmark.scene_by_id(scene_id)
        except KeyError:
            problems.append(
                ValidationProblem(where, f"unknown scene {scene_id!r} (not in benchmark)")
            )
            continue

        if record.get("split") != scene.split:
            problems.append(
                ValidationProblem(
                    f"{where}: split",
                    f"row says {record.get('split')!r}, benchmark says {scene.split!r}",
                )
            )

        status = record.get("status")
        failure_code = record.get("failure_code")
        if status == "judged":
            if failure_code is not None:
                problems.append(
                    ValidationProblem(f"{where}: failure_code", "must be null for judged rows")
                )
            outcome_readout = record.get("outcome_readout")
            trustworthiness = record.get("trustworthiness_audit")
            _payload_problems(
                outcome_readout, f"{where}: outcome_readout", OUTCOME_READOUT_FIELDS, problems
            )
            _payload_problems(
                trustworthiness,
                f"{where}: trustworthiness_audit",
                TRUSTWORTHINESS_FIELDS,
                problems,
            )
            for payload_name, payload in (
                ("outcome_readout", outcome_readout),
                ("trustworthiness_audit", trustworthiness),
            ):
                if isinstance(payload, dict) and payload.get("scene_id") != scene_id:
                    problems.append(
                        ValidationProblem(
                            f"{where}: {payload_name}.scene_id",
                            f"payload echoes {payload.get('scene_id')!r}, "
                            f"row is {scene_id!r}",
                        )
                    )
        elif status in ("model_failure", "infrastructure_failure"):
            if failure_code not in ROW_FAILURE_CODES:
                problems.append(
                    ValidationProblem(
                        f"{where}: failure_code",
                        f"must be one of {sorted(ROW_FAILURE_CODES)}, got {failure_code!r}",
                    )
                )
            if record.get("outcome_readout") is not None or (
                record.get("trustworthiness_audit") is not None
            ):
                problems.append(
                    ValidationProblem(where, "failure rows must not carry axis payloads")
                )
        else:
            problems.append(
                ValidationProblem(
                    f"{where}: status",
                    f"must be 'judged', 'model_failure', or 'infrastructure_failure', "
                    f"got {status!r}",
                )
            )
            continue

        relaxed = record.get("relaxed_included")
        strict = record.get("strict_included")
        if not _is_bool(relaxed) or not _is_bool(strict):
            problems.append(
                ValidationProblem(f"{where}: relaxed_included/strict_included", "must be booleans")
            )
            continue
        if strict and not relaxed:
            problems.append(
                ValidationProblem(
                    f"{where}: strict_included", "cannot be true when relaxed_included is false"
                )
            )

        outcome_label = record.get("outcome_label")
        if strict:
            if not isinstance(outcome_label, str) or outcome_label not in scene.outcome_labels:
                problems.append(
                    ValidationProblem(
                        f"{where}: outcome_label",
                        f"strict rows must carry a label from the scene vocabulary "
                        f"{scene.outcome_labels}, got {outcome_label!r}",
                    )
                )
        elif outcome_label is not None:
            problems.append(
                ValidationProblem(
                    f"{where}: outcome_label",
                    "only strict rows carry a scored outcome_label",
                )
            )

        parsed.append(
            ResultRow(
                scene_id=scene_id,
                repeat_index=repeat_index,
                split=scene.split,
                status=record["status"],
                failure_code=failure_code if isinstance(failure_code, str) else None,
                outcome_readout=record.get("outcome_readout"),
                trustworthiness_audit=record.get("trustworthiness_audit"),
                relaxed_included=relaxed,
                strict_included=strict,
                outcome_label=outcome_label if isinstance(outcome_label, str) else None,
            )
        )

    # Grid: rows must cover the submission's slots exactly.
    expected = {item.key for item in submission.items}
    declared = set(seen)
    missing = sorted(expected - declared)
    extra = sorted(declared - expected)
    if missing:
        problems.append(
            ValidationProblem(
                "rows.jsonl",
                f"{len(missing)} expected slot(s) missing: {_slots_detail(missing)}",
            )
        )
    if extra:
        problems.append(
            ValidationProblem(
                "rows.jsonl",
                f"slot(s) outside the submission grid: {_slots_detail(extra)}",
            )
        )

    # Statuses must agree with what the submission declared.
    by_key = {row.key: row for row in parsed}
    for item in submission.items:
        row = by_key.get(item.key)
        if row is None:
            continue
        if item.status == "produced":
            if row.status == "model_failure":
                problems.append(
                    ValidationProblem(
                        "rows.jsonl",
                        f"{item.scene_id} r{item.repeat_index} is produced in the submission "
                        "but model_failure in the rows",
                    )
                )
        elif item.status == "model_failure":
            if row.status != "model_failure" or row.failure_code != item.failure_code:
                problems.append(
                    ValidationProblem(
                        "rows.jsonl",
                        f"{item.scene_id} r{item.repeat_index} is declared "
                        f"{item.failure_code} in the submission but "
                        f"{row.status}/{row.failure_code} in the rows",
                    )
                )
        elif item.status == "infrastructure_failure":
            if row.status != "infrastructure_failure" or row.failure_code != item.failure_code:
                problems.append(
                    ValidationProblem(
                        "rows.jsonl",
                        f"{item.scene_id} r{item.repeat_index} is declared "
                        f"{item.failure_code} in the submission but "
                        f"{row.status}/{row.failure_code} in the rows",
                    )
                )

    return parsed


def _check_summary(
    summary: dict | None,
    benchmark: Benchmark,
    submission: Submission,
    rows: list[ResultRow],
    problems,
) -> Summary:
    if summary is None:
        problems.append(ValidationProblem("summary.json", "summary is missing or unreadable"))
        return _placeholder_summary(benchmark, submission)

    unexpected = _unexpected(set(summary), SUMMARY_FIELDS)
    if unexpected:
        problems.append(
            ValidationProblem("summary.json", f"unknown field(s): {', '.join(unexpected)}")
        )
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        problems.append(
            ValidationProblem(
                "summary.json: schema_version",
                f"expected {SUMMARY_SCHEMA_VERSION!r}, got {summary.get('schema_version')!r}",
            )
        )

    binding = summary.get("benchmark")
    if (
        not isinstance(binding, dict)
        or binding.get("dataset_id") != benchmark.dataset_id
        or (binding.get("benchmark_revision") != benchmark.benchmark_revision)
    ):
        problems.append(
            ValidationProblem("summary.json: benchmark", "binding does not match the benchmark")
        )

    if summary.get("system") != submission.system:
        problems.append(
            ValidationProblem(
                "summary.json: system",
                f"summary says {summary.get('system')!r}, submission says {submission.system!r}",
            )
        )
    if summary.get("prompt_set") != submission.prompt_set:
        problems.append(
            ValidationProblem(
                "summary.json: prompt_set",
                f"summary says {summary.get('prompt_set')!r}, "
                f"submission says {submission.prompt_set!r}",
            )
        )
    judge = summary.get("judge")
    judge_model = judge.get("model") if isinstance(judge, dict) else None
    if not isinstance(judge_model, str) or not judge_model:
        problems.append(
            ValidationProblem("summary.json: judge.model", "must be a non-empty string")
        )

    status = summary.get("status")
    if status not in SUMMARY_STATUSES:
        problems.append(
            ValidationProblem("summary.json: status", f"must be one of {sorted(SUMMARY_STATUSES)}")
        )
    warning = summary.get("warning")
    if status != "formal" and not (isinstance(warning, str) and warning):
        problems.append(
            ValidationProblem("summary.json: warning", "diagnostic statuses must carry a warning")
        )
    if status == "formal" and warning is not None:
        problems.append(
            ValidationProblem("summary.json: warning", "formal status takes no warning")
        )

    # Recompute the mechanical rollups and compare — F3 in action.
    expected_rollups, expected_denominator = compute_rollups(tuple(rows), benchmark)
    _check_denominator(summary.get("denominator"), expected_denominator, problems)
    rollups = _check_per_scene(summary.get("per_scene"), expected_rollups, benchmark, problems)
    calibration = _check_aggregate(
        summary.get("calibration"), "calibration", expected_rollups, problems
    )
    coverage = _check_aggregate(summary.get("coverage"), "coverage", expected_rollups, problems)

    return Summary(
        dataset_id=benchmark.dataset_id,
        benchmark_revision=benchmark.benchmark_revision,
        system=submission.system,
        prompt_set=submission.prompt_set,
        judge_model=judge_model if isinstance(judge_model, str) else "",
        denominator=expected_denominator,
        status=status if isinstance(status, str) else "diagnostic_partial_grid",
        warning=warning if isinstance(warning, str) else None,
        per_scene=rollups,
        calibration=calibration,
        coverage=coverage,
    )


def _check_denominator(declared: object, expected: DenominatorCounts, problems) -> None:
    if not isinstance(declared, dict) or set(declared) != {
        "expected_slots",
        "judged",
        "model_failure",
        "infrastructure_failure",
    }:
        problems.append(ValidationProblem("summary.json: denominator", "unexpected field set"))
        return
    for field, value in expected.to_dict().items():
        if declared.get(field) != value:
            problems.append(
                ValidationProblem(
                    "summary.json: denominator",
                    f"{field} says {declared.get(field)!r}, rows account {value!r}",
                )
            )


def _check_per_scene(
    declared: object, expected: tuple[SceneRollup, ...], benchmark: Benchmark, problems
) -> tuple[SceneRollup, ...]:
    if not isinstance(declared, list):
        problems.append(ValidationProblem("summary.json: per_scene", "must be an array"))
        return expected
    by_scene = {record.get("scene_id"): record for record in declared if isinstance(record, dict)}
    if len(by_scene) != len(declared) or set(by_scene) != {rollup.scene_id for rollup in expected}:
        problems.append(
            ValidationProblem(
                "summary.json: per_scene",
                "scene set does not match the submission scope",
            )
        )
    scored: list[SceneRollup] = []
    for rollup in expected:
        record = by_scene.get(rollup.scene_id)
        if record is None:
            scored.append(rollup)
            continue
        where = f"summary.json: per_scene[{rollup.scene_id}]"
        for field, value in (
            ("split", rollup.split),
            ("split_order", rollup.split_order),
            ("n", rollup.n),
            ("error_count", rollup.error_count),
            ("judged_label_counts", dict(rollup.judged_label_counts)),
        ):
            if record.get(field) != value:
                problems.append(
                    ValidationProblem(
                        where,
                        f"{field} says {record.get(field)!r}, rows account {value!r}",
                    )
                )
        # Score fields: type/placement checks here; re-derivation is scoring's test.
        scores: dict[str, float | bool | None] = {}
        if rollup.split == "calibration":
            tvd = record.get("tvd")
            if tvd is not None and not isinstance(tvd, int | float):
                problems.append(ValidationProblem(f"{where}: tvd", "must be a number or null"))
            else:
                scores["tvd"] = float(tvd) if tvd is not None else None
        else:
            support = record.get("support_recovery")
            if support is not None and not isinstance(support, int | float):
                problems.append(
                    ValidationProblem(f"{where}: support_recovery", "must be a number or null")
                )
            else:
                scores["support_recovery"] = float(support) if support is not None else None
        scene_passed = record.get("scene_passed")
        if scene_passed is not None and not isinstance(scene_passed, bool):
            problems.append(
                ValidationProblem(f"{where}: scene_passed", "must be a boolean or null")
            )
        else:
            scores["scene_passed"] = scene_passed
        scored.append(with_scene_scores(rollup, **scores))
    return tuple(scored)


def _check_aggregate(
    declared: object, split: str, rollups: tuple[SceneRollup, ...], problems
) -> SplitAggregate | None:
    split_rollups = [rollup for rollup in rollups if rollup.split == split]
    if not split_rollups:
        if declared is not None:
            problems.append(
                ValidationProblem(
                    f"summary.json: {split}",
                    "aggregate present but the submission scope has no such scenes",
                )
            )
        return None
    if not isinstance(declared, dict):
        problems.append(
            ValidationProblem(f"summary.json: {split}", "aggregate missing for a selected split")
        )
        return SplitAggregate(split=split)  # type: ignore[arg-type]
    n_pass = declared.get("n_pass")
    if not isinstance(n_pass, str) or not N_PASS_RE.fullmatch(n_pass):
        problems.append(
            ValidationProblem(f"summary.json: {split}.n_pass", "must look like 'n/total'")
        )
    macro_field = "macro_tvd" if split == "calibration" else "macro_support_recovery"
    macro = declared.get(macro_field)
    if macro is not None and not isinstance(macro, int | float):
        problems.append(
            ValidationProblem(f"summary.json: {split}.{macro_field}", "must be a number or null")
        )
    if split == "calibration":
        return SplitAggregate(
            split=split,
            macro_tvd=float(macro) if isinstance(macro, int | float) else None,
            n_pass=n_pass if isinstance(n_pass, str) else "",
        )
    return SplitAggregate(
        split=split,
        macro_support_recovery=float(macro) if isinstance(macro, int | float) else None,
        n_pass=n_pass if isinstance(n_pass, str) else "",
    )


def _placeholder_summary(benchmark: Benchmark, submission: Submission) -> Summary:
    return Summary(
        dataset_id=benchmark.dataset_id,
        benchmark_revision=benchmark.benchmark_revision,
        system=submission.system,
        prompt_set=submission.prompt_set,
        judge_model="",
        denominator=DenominatorCounts(0, 0, 0, 0),
        status="diagnostic_partial_grid",
        warning=None,
        per_scene=(),
        calibration=None,
        coverage=None,
    )
