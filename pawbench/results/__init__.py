"""Serialize and load public result bundles.

A bundle is exactly two files: ``rows.jsonl`` (one row per expected slot,
failures included) and ``summary.json``. Output is deterministic; loading
reconciles the bundle against its benchmark and submission — trust comes
from recomputation, not from digests.
"""

from __future__ import annotations

from pathlib import Path

from pawbench.benchmark import Benchmark
from pawbench.results._io import read_bundle
from pawbench.results._io import write as _write_bundle
from pawbench.results._model import (
    DenominatorCounts,
    ResultRow,
    ResultSet,
    SceneRollup,
    SplitAggregate,
    Summary,
    compute_rollups,
)
from pawbench.results._validate import validate
from pawbench.submission import Submission

__all__ = [
    "DenominatorCounts",
    "ResultRow",
    "ResultSet",
    "SceneRollup",
    "SplitAggregate",
    "Summary",
    "compute_rollups",
    "load",
    "write",
]


def write(result_set: ResultSet, path: str | Path) -> None:
    """Write ``rows.jsonl`` and ``summary.json`` into directory ``path``."""
    _write_bundle(result_set, path)


def load(path: str | Path, *, benchmark: Benchmark, submission: Submission) -> ResultSet:
    """Read and reconcile a result bundle.

    Raises :class:`~pawbench.errors.ResultError` listing every problem:
    missing slots, statuses contradicting the submission, or summary
    numbers that do not match what the rows account for.
    """
    rows, summary, problems = read_bundle(path)
    return validate(rows, summary, benchmark, submission, problems)
