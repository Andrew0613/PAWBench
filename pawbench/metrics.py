"""The future deterministic PAWBench metric entry point."""

from __future__ import annotations

from typing import Any


def compute_metrics(*args: Any, **kwargs: Any) -> None:
    """Compute PAWBench metrics once the released metric path is available."""

    del args, kwargs
    raise NotImplementedError("The public metric path is not available yet.")
