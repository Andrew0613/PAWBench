"""PIC-112: the unreleased package exposes only its future public journey."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import pawbench


def test_public_package_exports_only_the_two_journey_entry_points() -> None:
    assert pawbench.__all__ == ["compute_metrics", "evaluate"]
    assert callable(pawbench.evaluate)
    assert callable(pawbench.compute_metrics)


@pytest.mark.parametrize("module", ["benchmark", "submission", "results"])
def test_invented_public_protocol_modules_are_not_shipped(module: str) -> None:
    assert importlib.util.find_spec(f"pawbench.{module}") is None


def test_evaluation_is_truthful_until_its_reference_path_is_released() -> None:
    with pytest.raises(NotImplementedError, match="not available yet"):
        pawbench.evaluate()


def test_package_has_no_download_or_media_extras() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'dependencies = []' in pyproject
    assert "huggingface-hub" not in pyproject
    assert "opencv-python" not in pyproject
