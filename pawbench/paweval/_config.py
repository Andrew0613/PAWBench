"""Judge configuration shared by the client and the evaluator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeConfig:
    """Where to reach the judge and how to authenticate.

    Positional ``endpoint``/``model``/``api_key_env`` are the frozen
    contract; the rest are keyword-only knobs with safe defaults.
    """

    endpoint: str
    model: str
    api_key_env: str
    max_workers: int = 4
    timeout_s: float = 120.0
    max_tokens: int = 2048
