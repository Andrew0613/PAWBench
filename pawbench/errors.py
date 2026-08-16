"""Error taxonomy for PAWBench.

Validation errors are the wave-2 scope. ``PreflightError``, ``ResultError``
and friends arrive with the waves that raise them.

The design constraint here is aggregation: a submission manifest with twelve
bad rows should surface all twelve problems in one error, not the first one.
Validators collect problems and raise once, at the end.
"""

from __future__ import annotations

from dataclasses import dataclass


class PawbenchError(Exception):
    """Base class for every error raised by this library."""


@dataclass(frozen=True)
class ValidationProblem:
    """A single problem found in a public contract document.

    ``where`` locates the problem for a human, e.g.
    ``"scenes.jsonl line 3"`` or ``"benchmark.json: prompt_sets.gt_guided"``.
    """

    where: str
    detail: str


class ValidationError(PawbenchError):
    """A contract document failed validation.

    Carries *all* problems found, not just the first. ``str()`` renders them
    one per line, truncated after ``max_problems_displayed`` entries.
    """

    max_problems_displayed = 50

    def __init__(self, problems: list[ValidationProblem]) -> None:
        self.problems: tuple[ValidationProblem, ...] = tuple(problems)
        super().__init__(self._render())

    def _render(self) -> str:
        if not self.problems:
            return "validation failed"
        lines = [
            f"{problem.where}: {problem.detail}"
            for problem in self.problems[: self.max_problems_displayed]
        ]
        hidden = len(self.problems) - self.max_problems_displayed
        if hidden > 0:
            lines.append(f"...and {hidden} more")
        return "\n".join(lines)


class BenchmarkError(ValidationError):
    """A benchmark package failed validation."""


class SubmissionError(ValidationError):
    """A submission manifest failed validation."""


class ResultError(ValidationError):
    """A result bundle failed to read or reconcile.

    By the time a result bundle exists, its inputs were valid, so problems
    here indicate tampering or corruption rather than authoring mistakes.
    Uses the same batch presentation as the other validation errors.
    """
