"""PIC-112: the unreleased package exposes only its future public journey."""

from __future__ import annotations

import importlib.util
import inspect
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


def test_evaluation_exposes_the_complete_public_journey() -> None:
    assert tuple(inspect.signature(pawbench.evaluate).parameters) == (
        "benchmark_path",
        "videos",
        "model_or_lane",
        "vlm",
        "output_dir",
    )


def test_package_has_no_download_surface_and_media_is_opt_in() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'dependencies = ["PyYAML>=6.0"]' in pyproject
    assert "huggingface-hub" not in pyproject
    assert 'eval = ["opencv-python-headless>=4.9"]' in pyproject


def test_paweval_is_a_visible_implementation_package_with_versioned_rubrics() -> None:
    root = Path(__file__).resolve().parents[1]
    import pawbench.paweval as paweval

    assert (root / "pawbench" / "paweval").is_dir()
    assert not (root / "pawbench" / "_paweval").exists()
    assert (root / "pawbench" / "paweval" / "rubrics" / "outcome" / "A-03.yaml").is_file()
    assert (root / "pawbench" / "paweval" / "rubrics" / "trustworthiness" / "A-03.yaml").is_file()
    assert len(list((root / "pawbench" / "paweval" / "rubrics" / "outcome").glob("*.yaml"))) == 50
    assert (
        len(list((root / "pawbench" / "paweval" / "rubrics" / "trustworthiness").glob("*.yaml")))
        == 50
    )
    assert "judge" not in paweval.__all__
    assert not callable(getattr(paweval, "judge", None))
    assert not (root / "pawbench" / "paweval" / "schemas").exists()


@pytest.mark.parametrize(
    "scene_id",
    ["BA-02-S2", "BC-01-I1", "BC-07", "BC-10", "BS-01-S1", "BS-04", "TB-02", "TB-14"],
)
def test_non_v2_rubrics_are_not_shipped(scene_id: str) -> None:
    root = Path(__file__).resolve().parents[1] / "pawbench" / "paweval" / "rubrics"
    assert not (root / "outcome" / f"{scene_id}.yaml").exists()
    assert not (root / "trustworthiness" / f"{scene_id}.yaml").exists()


@pytest.mark.parametrize(
    "scene_id",
    ["BA-01-02", "BA-01-S1", "BC-01-I2", "BC-03-I1", "BS-01-I1"],
)
def test_v2_scene_variants_are_shipped(scene_id: str) -> None:
    root = Path(__file__).resolve().parents[1] / "pawbench" / "paweval" / "rubrics"
    assert (root / "outcome" / f"{scene_id}.yaml").is_file()
    assert (root / "trustworthiness" / f"{scene_id}.yaml").is_file()
