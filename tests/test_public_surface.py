"""Checks for the repository's direct, script-based public workflow."""

from __future__ import annotations

import importlib.util
import inspect
import re
from pathlib import Path

import pytest

from pawbench.evaluation import evaluate


def test_repository_uses_a_direct_script_instead_of_an_installable_package() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert (root / "evaluate.py").is_file()
    assert "pip install -r requirements.txt" in readme
    assert "python evaluate.py" in readme
    assert "## Quick start" in readme
    assert "## Detailed setup" in readme
    assert "pip install -e" not in readme
    assert "## Python API" not in readme
    assert "RESULTS-pre--release" not in readme
    assert "PROJECT-PAGE-coming_soon" not in readme
    assert "[project]" not in pyproject
    assert "[build-system]" not in pyproject


@pytest.mark.parametrize("module", ["benchmark", "submission", "results"])
def test_invented_public_protocol_modules_are_not_shipped(module: str) -> None:
    assert importlib.util.find_spec(f"pawbench.{module}") is None


def test_internal_evaluation_has_the_complete_script_journey() -> None:
    assert tuple(inspect.signature(evaluate).parameters) == (
        "benchmark_path",
        "videos",
        "model_or_lane",
        "vlm",
        "output_dir",
    )


def test_requirements_keep_evaluation_and_generation_dependencies_separate() -> None:
    root = Path(__file__).resolve().parents[1]
    evaluation = (root / "requirements.txt").read_text(encoding="utf-8")
    generation = (root / "requirements-generate.txt").read_text(encoding="utf-8")

    assert "PyYAML>=6.0" in evaluation
    assert "opencv-python-headless>=4.9" in evaluation
    assert "diffusers>=0.35" not in evaluation
    assert "-r requirements.txt" in generation
    assert "diffusers>=0.35" in generation


def test_diffusers_example_documents_one_continuous_generation_to_evaluation_path() -> None:
    root = Path(__file__).resolve().parents[1]
    examples = (root / "examples" / "README.md").read_text(encoding="utf-8")

    assert "python examples/generate_diffusers.py" in examples
    assert '  --output "$RUN_DIR/rollouts"' in examples
    assert "python evaluate.py" in examples
    assert '  --videos "$RUN_DIR/rollouts"' in examples


def test_readme_visuals_use_semantic_labels_and_existing_assets() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")

    for internal_id in ("A-01", "BC-01", "a01", "bc01"):
        assert internal_id not in readme

    assert '<img src="assets/' not in readme
    local_images = re.findall(r"!\[[^\]]+\]\((assets/[^)]+)\)", readme)
    assert local_images
    assert all((root / source).is_file() for source in local_images)
    assert "assets/paper/figure-1.png" in local_images
    assert "assets/paper/table-1.png" in local_images
    assert "assets/paweval-overview.png" in local_images


def test_paweval_is_a_visible_implementation_with_versioned_rubrics() -> None:
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
