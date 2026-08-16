"""Read a submission manifest from disk. Byte/JSON concerns only."""

from __future__ import annotations

import json
from pathlib import Path

from pawbench.errors import ValidationProblem


class _DuplicateKey(Exception):
    def __init__(self, key: str) -> None:
        self.key = key


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def read_envelope(path: str | Path) -> tuple[dict | None, list[ValidationProblem]]:
    """Read submission.json into an envelope dict, rejecting duplicate keys."""
    path = Path(path)
    problems: list[ValidationProblem] = []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return None, [
            ValidationProblem(str(path), f"cannot read submission file: {exc.strerror or exc}")
        ]
    try:
        decoded = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateKey as exc:
        problems.append(ValidationProblem(str(path), f"duplicate JSON key {exc.key!r}"))
        return None, problems
    except ValueError as exc:
        problems.append(ValidationProblem(str(path), f"invalid JSON: {exc}"))
        return None, problems
    if not isinstance(decoded, dict):
        problems.append(ValidationProblem(str(path), "submission file must contain a JSON object"))
        return None, problems
    return decoded, problems
