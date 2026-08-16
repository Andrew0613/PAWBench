"""Load and validate a PAWBench benchmark package.

The benchmark package is a directory containing ``benchmark.json``, a scene
table (JSONL), and one prompt-set file per declared prompt set. All paths
inside the package are relative to the package root.
"""

from __future__ import annotations

from pathlib import Path

from pawbench.benchmark._load import read_package
from pawbench.benchmark._model import (
    Benchmark,
    Outcome,
    PromptSet,
    ReferenceCount,
    Scene,
    Split,
)
from pawbench.benchmark._validate import validate
from pawbench.errors import BenchmarkError

__all__ = [
    "Benchmark",
    "Outcome",
    "PromptSet",
    "ReferenceCount",
    "Scene",
    "Split",
    "load",
]


def load(path: str | Path) -> Benchmark:
    """Load and validate the benchmark package at directory ``path``.

    Raises :class:`~pawbench.errors.BenchmarkError` listing every problem
    found across the manifest, the scene table, and the prompt sets.
    """
    raw = read_package(path)
    if raw.manifest is None:
        raise BenchmarkError(list(raw.problems))
    return validate(raw)
