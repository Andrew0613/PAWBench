"""Interface-level tests for pawbench.results.

Round-trip determinism and reconciliation: the bundle must reproduce the
submission's grid, statuses must agree with what each side can know, and
the summary must match what the rows account for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pawbench import benchmark as benchmark_module
from pawbench import results, submission
from pawbench.errors import ResultError, ValidationError
from pawbench.results import (
    ResultRow,
    ResultSet,
    SplitAggregate,
    Summary,
    compute_rollups,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def benchmark() -> benchmark_module.Benchmark:
    return benchmark_module.load(EXAMPLES / "benchmark")


@pytest.fixture
def sub(benchmark) -> submission.Submission:
    return submission.load(EXAMPLES / "submission" / "submission.json", benchmark=benchmark)


def _outcome_payload(label: str, scene_id: str = "SYN-C01") -> dict:
    return {
        "scene_id": scene_id,
        "outcome_readable": True,
        "outcome_in_schema": "IN_SCHEMA",
        "outcome_label": label,
        "observed_result": "synthetic observation",
        "failure_axes": [],
    }


def _trust_payload(scene_id: str = "SYN-C01") -> dict:
    return {
        "scene_id": scene_id,
        "status": "TRUSTED",
        "scene_grounding": {"status": "PASS", "notes": ""},
        "action_execution": {
            "status": "PASS",
            "target_hit": "YES",
            "object_acquired": "YES",
            "action_spec_followed": "YES",
            "notes": "",
        },
        "object_continuity": {"status": "PASS", "notes": ""},
        "physical_process": {"status": "PASS", "notes": ""},
        "failure_axes": [],
    }


def _synthetic_rows(benchmark, sub) -> tuple[ResultRow, ...]:
    """3 judged rows + 1 model_failure row matching the example submission."""
    rows = []
    for item in sub.items:
        scene = benchmark.scene_by_id(item.scene_id)
        if item.status == "model_failure":
            rows.append(
                ResultRow.failed(
                    scene_id=item.scene_id,
                    repeat_index=item.repeat_index,
                    split=scene.split,
                    status="model_failure",
                    failure_code=item.failure_code,
                )
            )
            continue
        label = scene.outcome_labels[item.repeat_index % len(scene.outcome_labels)]
        rows.append(
            ResultRow.judged(
                scene_id=item.scene_id,
                repeat_index=item.repeat_index,
                split=scene.split,
                outcome_readout=_outcome_payload(label, scene_id=item.scene_id),
                trustworthiness_audit=_trust_payload(scene_id=item.scene_id),
                relaxed_included=True,
                strict_included=True,
                outcome_label=label,
            )
        )
    return tuple(rows)


def _synthetic_summary(benchmark, sub, rows) -> Summary:
    rollups, denominator = compute_rollups(rows, benchmark)
    return Summary(
        dataset_id=benchmark.dataset_id,
        benchmark_revision=benchmark.benchmark_revision,
        system=sub.system,
        prompt_set=sub.prompt_set,
        judge_model="synthetic-judge",
        denominator=denominator,
        status="diagnostic_partial_grid",
        warning="example bundle: synthetic data, not comparable to formal results",
        per_scene=rollups,
        calibration=SplitAggregate(split="calibration", n_pass="1/1"),
        coverage=SplitAggregate(split="coverage", n_pass="1/1"),
    )


@pytest.fixture
def result_set(benchmark, sub) -> ResultSet:
    rows = _synthetic_rows(benchmark, sub)
    return ResultSet(rows=rows, summary=_synthetic_summary(benchmark, sub, rows))


def _problems_of(exc: ResultError) -> list[str]:
    return [f"{p.where}: {p.detail}" for p in exc.problems]


# --- round trip -----------------------------------------------------------


def test_write_load_round_trip(benchmark, sub, result_set, tmp_path) -> None:
    results.write(result_set, tmp_path / "bundle")
    loaded = results.load(tmp_path / "bundle", benchmark=benchmark, submission=sub)

    assert loaded == result_set
    assert loaded.summary.denominator.expected_slots == 4
    assert loaded.summary.denominator.model_failure == 1


def test_write_is_deterministic(result_set, tmp_path) -> None:
    results.write(result_set, tmp_path / "a")
    results.write(result_set, tmp_path / "b")
    assert (tmp_path / "a" / "rows.jsonl").read_bytes() == (
        tmp_path / "b" / "rows.jsonl"
    ).read_bytes()
    assert (tmp_path / "a" / "summary.json").read_bytes() == (
        tmp_path / "b" / "summary.json"
    ).read_bytes()


def test_rows_are_sorted_on_write(benchmark, sub, result_set, tmp_path) -> None:
    shuffled = ResultSet(rows=tuple(reversed(result_set.rows)), summary=result_set.summary)
    results.write(shuffled, tmp_path / "bundle")
    results.write(result_set, tmp_path / "other")
    assert (tmp_path / "bundle" / "rows.jsonl").read_bytes() == (
        tmp_path / "other" / "rows.jsonl"
    ).read_bytes()


def test_rollups_account_the_denominator(benchmark, sub, result_set) -> None:
    rollups, denominator = compute_rollups(result_set.rows, benchmark)
    by_scene = {rollup.scene_id: rollup for rollup in rollups}
    assert by_scene["SYN-C01"].n == 2
    assert by_scene["SYN-C01"].error_count == 0
    assert by_scene["SYN-C01"].strict_count == 2
    assert by_scene["SYN-P01"].error_count == 1  # the model_failure row
    assert denominator.expected_slots == 4
    assert denominator.judged == 3
    assert denominator.model_failure == 1
    assert denominator.infrastructure_failure == 0


def test_error_counts_explode_on_untrusted_rows(benchmark, sub) -> None:
    rows = []
    for row in _synthetic_rows(benchmark, sub):
        if row.status == "judged" and row.key == ("SYN-C01", 0):
            # judge succeeded but trust failed -> not strict, still an error
            rows.append(
                ResultRow.judged(
                    scene_id=row.scene_id,
                    repeat_index=row.repeat_index,
                    split=row.split,
                    outcome_readout=row.outcome_readout,
                    trustworthiness_audit=_trust_payload() | {"status": "UNTRUSTED"},  # SYN-C01 row
                    relaxed_included=True,
                    strict_included=False,
                    outcome_label=None,
                )
            )
        else:
            rows.append(row)
    rollups, denominator = compute_rollups(tuple(rows), benchmark)
    assert denominator.judged == 3
    by_scene = {rollup.scene_id: rollup for rollup in rollups}
    assert by_scene["SYN-C01"].error_count == 1  # relaxed but not strict


# --- reconciliation failures ----------------------------------------------


def _write_and_load(benchmark, sub, result_set, tmp_path, mutate):
    results.write(result_set, tmp_path / "bundle")
    mutate(tmp_path / "bundle")
    return results.load(tmp_path / "bundle", benchmark=benchmark, submission=sub)


def test_rejects_missing_row(benchmark, sub, result_set, tmp_path) -> None:
    def mutate(bundle: Path) -> None:
        lines = (bundle / "rows.jsonl").read_text().splitlines()
        (bundle / "rows.jsonl").write_text("\n".join(lines[1:]) + "\n")

    with pytest.raises(ResultError) as excinfo:
        _write_and_load(benchmark, sub, result_set, tmp_path, mutate)
    assert any("missing" in p for p in _problems_of(excinfo.value))


def test_rejects_row_contradicting_submission(benchmark, sub, result_set, tmp_path) -> None:
    def mutate(bundle: Path) -> None:
        rows = [json.loads(line) for line in (bundle / "rows.jsonl").read_text().splitlines()]
        for row in rows:
            if row["status"] == "model_failure":
                # SYN-P01 r1 is declared generation_timeout by the submission
                row["failure_code"] = "generation_refused"
        (bundle / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ResultError) as excinfo:
        _write_and_load(benchmark, sub, result_set, tmp_path, mutate)
    assert any("generation_timeout" in p for p in _problems_of(excinfo.value))


def test_rejects_produced_slot_becoming_model_failure(benchmark, sub, result_set, tmp_path) -> None:
    def mutate(bundle: Path) -> None:
        rows = [json.loads(line) for line in (bundle / "rows.jsonl").read_text().splitlines()]
        for row in rows:
            if row["scene_id"] == "SYN-C01" and row["status"] == "judged":
                row["status"] = "model_failure"
                row["failure_code"] = "generation_failed"
                row["outcome_readout"] = None
                row["trustworthiness_audit"] = None
                row["relaxed_included"] = False
                row["strict_included"] = False
                row["outcome_label"] = None
        (bundle / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ResultError) as excinfo:
        _write_and_load(benchmark, sub, result_set, tmp_path, mutate)
    assert any("produced in the submission" in p for p in _problems_of(excinfo.value))


def test_rejects_wrong_denominator_in_summary(benchmark, sub, result_set, tmp_path) -> None:
    def mutate(bundle: Path) -> None:
        summary = json.loads((bundle / "summary.json").read_text())
        summary["denominator"]["judged"] = 99  # rows account 3
        (bundle / "summary.json").write_text(json.dumps(summary, indent=2))

    with pytest.raises(ResultError) as excinfo:
        _write_and_load(benchmark, sub, result_set, tmp_path, mutate)
    assert any("judged" in p for p in _problems_of(excinfo.value))


def test_rejects_wrong_per_scene_error_count(benchmark, sub, result_set, tmp_path) -> None:
    def mutate(bundle: Path) -> None:
        summary = json.loads((bundle / "summary.json").read_text())
        summary["per_scene"][1]["error_count"] = 0  # rows account 1
        (bundle / "summary.json").write_text(json.dumps(summary, indent=2))

    with pytest.raises(ResultError) as excinfo:
        _write_and_load(benchmark, sub, result_set, tmp_path, mutate)
    assert any("error_count" in p for p in _problems_of(excinfo.value))


def test_rejects_diagnostic_without_warning(benchmark, sub, result_set, tmp_path) -> None:
    def mutate(bundle: Path) -> None:
        summary = json.loads((bundle / "summary.json").read_text())
        summary.pop("warning")
        (bundle / "summary.json").write_text(json.dumps(summary, indent=2))

    with pytest.raises(ResultError) as excinfo:
        _write_and_load(benchmark, sub, result_set, tmp_path, mutate)
    assert any("warning" in p for p in _problems_of(excinfo.value))


def test_rejects_strict_row_with_foreign_label(benchmark, sub, result_set, tmp_path) -> None:
    def mutate(bundle: Path) -> None:
        rows = [json.loads(line) for line in (bundle / "rows.jsonl").read_text().splitlines()]
        for row in rows:
            if row["status"] == "judged":
                row["outcome_label"] = "edge"  # not in either scene's vocabulary
                row["outcome_readout"]["outcome_label"] = "edge"
                break
        (bundle / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ResultError) as excinfo:
        _write_and_load(benchmark, sub, result_set, tmp_path, mutate)
    assert any("vocabulary" in p for p in _problems_of(excinfo.value))


def test_rejects_strict_without_relaxed(benchmark, sub, result_set, tmp_path) -> None:
    def mutate(bundle: Path) -> None:
        rows = [json.loads(line) for line in (bundle / "rows.jsonl").read_text().splitlines()]
        for row in rows:
            if row["status"] == "judged":
                row["relaxed_included"] = False  # strict still true
                break
        (bundle / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ResultError) as excinfo:
        _write_and_load(benchmark, sub, result_set, tmp_path, mutate)
    assert any("strict_included" in p for p in _problems_of(excinfo.value))


def test_rejects_failure_row_with_payloads(benchmark, sub, result_set, tmp_path) -> None:
    def mutate(bundle: Path) -> None:
        rows = [json.loads(line) for line in (bundle / "rows.jsonl").read_text().splitlines()]
        for row in rows:
            if row["status"] == "model_failure":
                row["outcome_readout"] = _outcome_payload("heads")
        (bundle / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ResultError) as excinfo:
        _write_and_load(benchmark, sub, result_set, tmp_path, mutate)
    assert any("axis payloads" in p for p in _problems_of(excinfo.value))


def test_rejects_missing_bundle(benchmark, sub, tmp_path) -> None:
    with pytest.raises(ResultError) as excinfo:
        results.load(tmp_path / "nope", benchmark=benchmark, submission=sub)
    assert any("not a directory" in p for p in _problems_of(excinfo.value))


def test_score_fields_round_trip(benchmark, sub, result_set, tmp_path) -> None:
    """Wave-5 preview: declared score values survive a load round trip."""
    from pawbench.results._model import with_scene_scores

    scored = tuple(
        with_scene_scores(rollup, tvd=0.25)
        if rollup.split == "calibration"
        else with_scene_scores(rollup, support_recovery=1 / 3)
        for rollup in result_set.summary.per_scene
    )
    summary = result_set.summary
    scored_summary = Summary(
        dataset_id=summary.dataset_id,
        benchmark_revision=summary.benchmark_revision,
        system=summary.system,
        prompt_set=summary.prompt_set,
        judge_model=summary.judge_model,
        denominator=summary.denominator,
        status=summary.status,
        warning=summary.warning,
        per_scene=scored,
        calibration=SplitAggregate(split="calibration", macro_tvd=0.25, n_pass="1/1"),
        coverage=SplitAggregate(split="coverage", macro_support_recovery=1 / 3, n_pass="1/1"),
    )
    scored_set = ResultSet(rows=result_set.rows, summary=scored_summary)
    results.write(scored_set, tmp_path / "bundle")
    loaded = results.load(tmp_path / "bundle", benchmark=benchmark, submission=sub)
    assert loaded.summary.calibration == scored_summary.calibration
    by_scene = {rollup.scene_id: rollup for rollup in loaded.summary.per_scene}
    assert by_scene["SYN-C01"].tvd == pytest.approx(0.25)
    assert by_scene["SYN-P01"].support_recovery == pytest.approx(1 / 3)


def test_error_hierarchy() -> None:
    assert issubclass(ResultError, ValidationError)
