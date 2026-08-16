"""Typed model of a result bundle: rows.jsonl + summary.json.

The row set is the fixed denominator carried end to end: one ResultRow per
expected (scene, repeat) slot, failure slots included. The summary is a pure
function of the rows and the benchmark — :func:`compute_rollups` is the
single source for the mechanical rollup arithmetic, reused by scoring (wave
5) and by load-time reconciliation so the two can never drift.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from pawbench.benchmark import Benchmark, Split

ROW_SCHEMA_VERSION = "paweval.result_row/v1"
SUMMARY_SCHEMA_VERSION = "paweval.run_summary/v1"

RowStatus = Literal["judged", "model_failure", "infrastructure_failure"]

ROW_FAILURE_CODES = frozenset(
    {
        # generation (declared by the submission)
        "generation_failed",
        "generation_timeout",
        "generation_refused",
        # media (discovered at evaluation preflight)
        "media_missing",
        "media_corrupt",
        "media_undecodable",
        # judge transport/parsing (discovered during evaluation)
        "judge_timeout",
        "judge_network",
        "judge_rate_limited",
        "judge_server_error",
        "judge_malformed_response",
    }
)

SUMMARY_STATUSES = frozenset({"formal", "diagnostic_partial_grid", "diagnostic_infra_failure"})

OUTCOME_READOUT_FIELDS = frozenset(
    {"outcome_readable", "outcome_in_schema", "outcome_label", "observed_result", "failure_axes"}
)
TRUST_BLOCK_FIELDS = frozenset({"status", "notes"})
TRUST_ACTION_FIELDS = frozenset(
    {"status", "target_hit", "object_acquired", "action_spec_followed", "notes"}
)
TRUSTWORTHINESS_FIELDS = frozenset(
    {"status", "scene_grounding", "action_execution", "object_continuity", "physical_process",
     "failure_axes"}
)


@dataclass(frozen=True)
class ResultRow:
    """One expected slot after evaluation: judged, or an explicit failure."""

    scene_id: str
    repeat_index: int
    split: Split
    status: RowStatus
    failure_code: str | None
    outcome_readout: Mapping | None
    trustworthiness_audit: Mapping | None
    relaxed_included: bool
    strict_included: bool
    outcome_label: str | None

    @classmethod
    def judged(
        cls,
        *,
        scene_id: str,
        repeat_index: int,
        split: Split,
        outcome_readout: Mapping,
        trustworthiness_audit: Mapping,
        relaxed_included: bool,
        strict_included: bool,
        outcome_label: str | None,
    ) -> ResultRow:
        return cls(
            scene_id=scene_id,
            repeat_index=repeat_index,
            split=split,
            status="judged",
            failure_code=None,
            outcome_readout=outcome_readout,
            trustworthiness_audit=trustworthiness_audit,
            relaxed_included=relaxed_included,
            strict_included=strict_included,
            outcome_label=outcome_label,
        )

    @classmethod
    def failed(
        cls, *, scene_id: str, repeat_index: int, split: Split, status: RowStatus,
        failure_code: str,
    ) -> ResultRow:
        return cls(
            scene_id=scene_id,
            repeat_index=repeat_index,
            split=split,
            status=status,
            failure_code=failure_code,
            outcome_readout=None,
            trustworthiness_audit=None,
            relaxed_included=False,
            strict_included=False,
            outcome_label=None,
        )

    @property
    def key(self) -> tuple[str, int]:
        return (self.scene_id, self.repeat_index)

    @property
    def counts_as_error(self) -> bool:
        """Errors shrink the scoreable mass; infrastructure failures do not."""
        return self.status != "infrastructure_failure" and not self.strict_included

    def to_dict(self) -> dict:
        return {
            "schema_version": ROW_SCHEMA_VERSION,
            "scene_id": self.scene_id,
            "repeat_index": self.repeat_index,
            "split": self.split,
            "status": self.status,
            "failure_code": self.failure_code,
            "outcome_readout": dict(self.outcome_readout) if self.outcome_readout else None,
            "trustworthiness_audit": (
                dict(self.trustworthiness_audit) if self.trustworthiness_audit else None
            ),
            "relaxed_included": self.relaxed_included,
            "strict_included": self.strict_included,
            "outcome_label": self.outcome_label,
        }


@dataclass(frozen=True)
class DenominatorCounts:
    """The denominator, fully accounted: the four counts sum to expected."""

    expected_slots: int
    judged: int
    model_failure: int
    infrastructure_failure: int

    def to_dict(self) -> dict:
        return {
            "expected_slots": self.expected_slots,
            "judged": self.judged,
            "model_failure": self.model_failure,
            "infrastructure_failure": self.infrastructure_failure,
        }


@dataclass(frozen=True)
class SceneRollup:
    """Per-scene accounting; score fields are filled by scoring, not IO."""

    scene_id: str
    split: Split
    split_order: int
    n: int
    error_count: int
    judged_label_counts: Mapping[str, int]
    strict_count: int
    tvd: float | None = None
    support_recovery: float | None = None
    scene_passed: bool | None = None

    def to_dict(self) -> dict:
        record = {
            "scene_id": self.scene_id,
            "split": self.split,
            "split_order": self.split_order,
            "n": self.n,
            "error_count": self.error_count,
            "judged_label_counts": dict(self.judged_label_counts),
            "scene_passed": self.scene_passed,
        }
        if self.split == "calibration":
            record["tvd"] = self.tvd
        else:
            record["support_recovery"] = self.support_recovery
        return record


@dataclass(frozen=True)
class SplitAggregate:
    """Model summary for one split: macro over passing scenes, n_pass."""

    split: Split
    macro_tvd: float | None = None
    macro_support_recovery: float | None = None
    n_pass: str = ""

    def to_dict(self) -> dict:
        if self.split == "calibration":
            return {"macro_tvd": self.macro_tvd, "n_pass": self.n_pass}
        return {"macro_support_recovery": self.macro_support_recovery, "n_pass": self.n_pass}


@dataclass(frozen=True)
class Summary:
    """Run summary: a pure function of rows and benchmark, serialized as-is."""

    dataset_id: str
    benchmark_revision: int
    system: str
    prompt_set: str
    judge_model: str
    denominator: DenominatorCounts
    status: str
    warning: str | None
    per_scene: tuple[SceneRollup, ...]
    calibration: SplitAggregate | None
    coverage: SplitAggregate | None

    def to_dict(self) -> dict:
        record: dict = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "benchmark": {
                "dataset_id": self.dataset_id,
                "benchmark_revision": self.benchmark_revision,
            },
            "system": self.system,
            "prompt_set": self.prompt_set,
            "judge": {"model": self.judge_model},
            "denominator": self.denominator.to_dict(),
            "status": self.status,
            "per_scene": [rollup.to_dict() for rollup in self.per_scene],
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "coverage": self.coverage.to_dict() if self.coverage else None,
        }
        if self.warning is not None:
            record["warning"] = self.warning
        return record


@dataclass(frozen=True)
class ResultSet:
    """A complete result bundle: one row per expected slot, plus the summary."""

    rows: tuple[ResultRow, ...]
    summary: Summary


def compute_rollups(
    rows: tuple[ResultRow, ...], benchmark: Benchmark
) -> tuple[tuple[SceneRollup, ...], DenominatorCounts]:
    """Mechanical rollup arithmetic over a complete row set.

    Single source of truth: scoring fills the tvd/support/scene_passed
    fields on top of these rollups, and load-time reconciliation recomputes
    with the same function, so serialization and scoring cannot drift.
    """
    by_scene: dict[str, list[ResultRow]] = {}
    for row in rows:
        by_scene.setdefault(row.scene_id, []).append(row)

    rollups: list[SceneRollup] = []
    for scene in benchmark.scenes:
        scene_rows = by_scene.get(scene.scene_id, [])
        if not scene_rows:
            continue
        strict_rows = [row for row in scene_rows if row.strict_included]
        label_counts: dict[str, int] = {}
        for row in strict_rows:
            label_counts[row.outcome_label] = label_counts.get(row.outcome_label, 0) + 1
        n = len(scene_rows)
        error_count = sum(1 for row in scene_rows if row.counts_as_error)
        rollups.append(
            SceneRollup(
                scene_id=scene.scene_id,
                split=scene.split,
                split_order=scene.split_order,
                n=n,
                error_count=error_count,
                judged_label_counts=label_counts,
                strict_count=len(strict_rows),
            )
        )

    denominator = DenominatorCounts(
        expected_slots=len(rows),
        judged=sum(1 for row in rows if row.status == "judged"),
        model_failure=sum(1 for row in rows if row.status == "model_failure"),
        infrastructure_failure=sum(
            1 for row in rows if row.status == "infrastructure_failure"
        ),
    )
    return tuple(rollups), denominator


def with_scene_scores(rollup: SceneRollup, **scores: float | None | bool) -> SceneRollup:
    """Return a copy of ``rollup`` with score fields set (used by scoring)."""
    return replace(rollup, **scores)
