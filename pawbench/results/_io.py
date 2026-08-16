"""Deterministic serialization of a result bundle.

Output contract: ``path/rows.jsonl`` (one JSON object per expected slot,
sorted by (split, scene_id, repeat_index)) and ``path/summary.json``.
Identical ResultSets serialize to identical bytes — no timestamps, no run
IDs, no environment details.
"""

from __future__ import annotations

import json
from pathlib import Path

from pawbench.errors import ValidationProblem
from pawbench.results._model import ResultSet

ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.json"


def _row_sort_key(row) -> tuple[str, str, int]:
    return (row.split, row.scene_id, row.repeat_index)


def write(result_set: ResultSet, path: str | Path) -> None:
    """Write rows.jsonl and summary.json into directory ``path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    rows = sorted(result_set.rows, key=_row_sort_key)
    lines = "".join(
        json.dumps(row.to_dict(), ensure_ascii=False, separators=(", ", ": ")) + "\n"
        for row in rows
    )
    (directory / ROWS_NAME).write_text(lines, encoding="utf-8")
    (directory / SUMMARY_NAME).write_text(
        json.dumps(result_set.summary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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


def read_bundle(path: str | Path) -> tuple[list[dict], dict | None, list[ValidationProblem]]:
    """Read rows.jsonl and summary.json; parse-level problems are collected."""
    directory = Path(path)
    problems: list[ValidationProblem] = []
    if not directory.is_dir():
        return [], None, [
            ValidationProblem(str(directory), "result bundle path is not a directory")
        ]

    rows: list[dict] = []
    rows_path = directory / ROWS_NAME
    try:
        text = rows_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        problems.append(
            ValidationProblem(ROWS_NAME, f"cannot read file: {exc.strerror or exc}")
        )
        text = None
    if text is not None:
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            decoded = _decode(line, f"{ROWS_NAME} line {line_number}", problems)
            if isinstance(decoded, dict):
                rows.append(decoded)
            elif decoded is not None:
                problems.append(
                    ValidationProblem(f"{ROWS_NAME} line {line_number}", "row must be an object")
                )

    summary: dict | None = None
    try:
        summary_text = (directory / SUMMARY_NAME).read_text(encoding="utf-8-sig")
    except OSError as exc:
        problems.append(
            ValidationProblem(SUMMARY_NAME, f"cannot read file: {exc.strerror or exc}")
        )
        summary_text = None
    if summary_text is not None:
        decoded = _decode(summary_text, SUMMARY_NAME, problems)
        if isinstance(decoded, dict):
            summary = decoded
        elif decoded is not None:
            problems.append(ValidationProblem(SUMMARY_NAME, "summary must be an object"))

    return rows, summary, problems


def _decode(text: str, where: str, problems: list[ValidationProblem]) -> object | None:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateKey as exc:
        problems.append(ValidationProblem(where, f"duplicate JSON key {exc.key!r}"))
    except ValueError as exc:
        problems.append(ValidationProblem(where, f"invalid JSON: {exc}"))
    return None
