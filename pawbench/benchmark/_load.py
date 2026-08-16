"""Read a benchmark package from disk into raw structures.

This module owns byte/JSON-level concerns only: locating files, decoding
UTF-8, rejecting duplicate JSON object keys, and preserving line numbers.
Semantic invariants live in ``_validate``. Parse-level defects are recorded
as :class:`~pawbench.errors.ValidationProblem` so everything can be reported
in a single error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pawbench.errors import ValidationProblem

MANIFEST_NAME = "benchmark.json"


@dataclass(frozen=True)
class RawPackage:
    """Raw file contents of a benchmark package, plus parse-level problems."""

    root: Path
    manifest: dict | None
    scenes: tuple[tuple[int, dict], ...]  # (1-based line number, row)
    prompt_sets: tuple[tuple[str, tuple[tuple[int, dict], ...]], ...]
    problems: tuple[ValidationProblem, ...]


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


def _decode_json(text: str, where: str, problems: list[ValidationProblem]) -> object | None:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateKey as exc:
        problems.append(ValidationProblem(where, f"duplicate JSON key {exc.key!r}"))
    except ValueError as exc:
        problems.append(ValidationProblem(where, f"invalid JSON: {exc}"))
    return None


def _read_text(path: Path, where: str, problems: list[ValidationProblem]) -> str | None:
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        problems.append(ValidationProblem(where, f"cannot read file: {exc.strerror or exc}"))
        return None


def _read_jsonl(path: Path, where: str, problems: list[ValidationProblem]):
    """Read a JSONL file into (line_number, row) pairs; problems per bad line."""
    text = _read_text(path, where, problems)
    if text is None:
        return ()
    rows: list[tuple[int, dict]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        decoded = _decode_json(line, f"{where} line {line_number}", problems)
        if decoded is None:
            continue
        if isinstance(decoded, dict):
            rows.append((line_number, decoded))
        else:
            problems.append(
                ValidationProblem(
                    f"{where} line {line_number}", "line must be a JSON object"
                )
            )
    return tuple(rows)


def read_package(root: str | Path) -> RawPackage:
    """Read the package under ``root`` without applying any invariants."""
    root = Path(root)
    problems: list[ValidationProblem] = []
    if not root.is_dir():
        return RawPackage(
            root=root,
            manifest=None,
            scenes=(),
            prompt_sets=(),
            problems=(
                ValidationProblem(str(root), "benchmark package path is not a directory"),
            ),
        )

    manifest: dict | None = None
    manifest_text = _read_text(root / MANIFEST_NAME, MANIFEST_NAME, problems)
    if manifest_text is not None:
        decoded = _decode_json(manifest_text, MANIFEST_NAME, problems)
        if isinstance(decoded, dict):
            manifest = decoded
        elif decoded is not None:
            problems.append(ValidationProblem(MANIFEST_NAME, "manifest must be a JSON object"))

    scenes: tuple[tuple[int, dict], ...] = ()
    prompt_sets: list[tuple[str, tuple[tuple[int, dict], ...]]] = []
    if manifest is not None:
        scene_table = manifest.get("scene_table")
        if isinstance(scene_table, str) and scene_table:
            scenes = _read_jsonl(root / scene_table, scene_table, problems)

        declared = manifest.get("prompt_sets")
        if isinstance(declared, dict):
            for name, entry in declared.items():
                path = entry.get("path") if isinstance(entry, dict) else None
                if isinstance(path, str) and path:
                    prompt_sets.append((name, _read_jsonl(root / path, path, problems)))

    return RawPackage(
        root=root,
        manifest=manifest,
        scenes=scenes,
        prompt_sets=tuple(prompt_sets),
        problems=tuple(problems),
    )
