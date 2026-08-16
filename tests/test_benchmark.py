"""Interface-level tests for pawbench.benchmark.

Every invariant gets a counter-example test; the per-split split_order
semantics get an explicit regression test for the internal global-order bug
this package replaces.
"""

from __future__ import annotations

import json
import shutil
from fractions import Fraction
from pathlib import Path

import pytest

from pawbench import benchmark
from pawbench.errors import (
    BenchmarkError,
    PawbenchError,
    ValidationError,
    ValidationProblem,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "benchmark"


def _load_docs() -> tuple[dict, list[dict], list[dict]]:
    manifest = json.loads((EXAMPLE / "benchmark.json").read_text())
    scenes = [
        json.loads(line)
        for line in (EXAMPLE / "scenes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    prompts = [
        json.loads(line)
        for line in (EXAMPLE / "prompts" / "gt_guided.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return manifest, scenes, prompts


def _write_package(
    tmp_path: Path,
    *,
    manifest: dict | None = None,
    scenes: list[dict] | None = None,
    prompts: list[dict] | None = None,
    raw_scenes_text: str | None = None,
) -> Path:
    """Materialize a package from (patched) example documents."""
    default_manifest, default_scenes, default_prompts = _load_docs()
    root = tmp_path / "pkg"
    shutil.copytree(EXAMPLE, root)
    (root / "benchmark.json").write_text(json.dumps(manifest or default_manifest, indent=2))
    if raw_scenes_text is not None:
        (root / "scenes.jsonl").write_text(raw_scenes_text)
    else:
        rows = default_scenes if scenes is None else scenes
        (root / "scenes.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    if prompts is not None:
        (root / "prompts" / "gt_guided.jsonl").write_text(
            "\n".join(json.dumps(row) for row in prompts) + "\n"
        )
    return root


def _problems_of(exc: BenchmarkError) -> list[str]:
    return [f"{p.where}: {p.detail}" for p in exc.problems]


# --- happy path -----------------------------------------------------------


def test_loads_example_package() -> None:
    result = benchmark.load(EXAMPLE)

    assert result.dataset_id == "PAWBench-example"
    assert result.benchmark_revision == 1
    assert result.formal_repeats == 50
    assert [scene.scene_id for scene in result.scenes] == ["SYN-C01", "SYN-P01"]
    assert sorted(result.prompt_sets) == ["gt_guided"]

    scene = result.scene_by_id("SYN-C01")
    assert scene.outcome_labels == ("heads", "tails")
    assert scene.is_calibration
    assert scene.reference_distribution() == {
        "heads": Fraction(1, 2),
        "tails": Fraction(1, 2),
    }

    coverage = result.scene_by_id("SYN-P01")
    assert not coverage.is_calibration
    assert coverage.reference_distribution() is None
    assert coverage.judge_notes is not None


def test_split_order_is_per_split_not_global() -> None:
    """Regression: the internal loader turned split_order into a global order.

    Both splits reusing split_order 0 must be valid here.
    """
    result = benchmark.load(EXAMPLE)
    orders = [scene.split_order for scene in result.scenes]
    assert orders == [0, 0]  # one scene per split, both order 0


def test_scenes_in_split_is_ordered() -> None:
    result = benchmark.load(EXAMPLE)
    assert [s.scene_id for s in result.scenes_in_split("calibration")] == ["SYN-C01"]
    assert [s.scene_id for s in result.scenes_in_split("coverage")] == ["SYN-P01"]


def test_scene_by_id_unknown_raises() -> None:
    result = benchmark.load(EXAMPLE)
    with pytest.raises(KeyError):
        result.scene_by_id("NOPE-01")


def test_resolve_image_path_stays_in_package() -> None:
    result = benchmark.load(EXAMPLE)
    path = result.resolve_image_path(result.scene_by_id("SYN-C01"))
    assert path.is_relative_to(result.root.resolve())
    assert path.name == "SYN-C01.png"


# --- manifest -------------------------------------------------------------


def test_rejects_unknown_manifest_field(tmp_path: Path) -> None:
    manifest, _, _ = _load_docs()
    manifest["surprise"] = True
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, manifest=manifest))
    assert any("surprise" in p for p in _problems_of(excinfo.value))


def test_rejects_wrong_schema_version(tmp_path: Path) -> None:
    manifest, _, _ = _load_docs()
    manifest["schema_version"] = "pawbench.benchmark_package/v0"
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, manifest=manifest))
    assert any("schema_version" in p for p in _problems_of(excinfo.value))


def test_rejects_bad_formal_repeats(tmp_path: Path) -> None:
    manifest, _, _ = _load_docs()
    manifest["formal_repeats"] = 0
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, manifest=manifest))
    assert any("formal_repeats" in p for p in _problems_of(excinfo.value))


def test_rejects_not_a_directory(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(tmp_path / "missing")
    assert any("not a directory" in p for p in _problems_of(excinfo.value))


def test_rejects_prompt_set_count_fields(tmp_path: Path) -> None:
    """v2: manifest prompt-set entries carry only a path; counts invite drift."""
    manifest, _, _ = _load_docs()
    manifest["prompt_sets"]["gt_guided"]["scene_count"] = 2
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, manifest=manifest))
    assert any("scene_count" in p for p in _problems_of(excinfo.value))


# --- scene table ----------------------------------------------------------


def test_rejects_duplicate_scene_id(tmp_path: Path) -> None:
    _, scenes, _ = _load_docs()
    scenes.append(dict(scenes[0]))
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, scenes=scenes))
    assert any("duplicate scene_id" in p for p in _problems_of(excinfo.value))


def test_rejects_duplicate_split_order_within_split(tmp_path: Path) -> None:
    _, scenes, _ = _load_docs()
    extra = dict(scenes[0], scene_id="SYN-C02")  # same calibration split, same order
    scenes.append(extra)
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, scenes=scenes))
    assert any("duplicate split_order" in p for p in _problems_of(excinfo.value))


def test_rejects_bad_scene_id_pattern(tmp_path: Path) -> None:
    _, scenes, _ = _load_docs()
    scenes[0]["scene_id"] = "0bad id"
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, scenes=scenes))
    assert any("scene_id" in p for p in _problems_of(excinfo.value))


def test_rejects_unknown_split(tmp_path: Path) -> None:
    _, scenes, _ = _load_docs()
    scenes[0]["split"] = "holdout"
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, scenes=scenes))
    assert any("split" in p for p in _problems_of(excinfo.value))


def test_rejects_empty_action(tmp_path: Path) -> None:
    _, scenes, _ = _load_docs()
    scenes[0]["action"] = ""
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, scenes=scenes))
    assert any("action" in p for p in _problems_of(excinfo.value))


def test_rejects_unknown_scene_field(tmp_path: Path) -> None:
    _, scenes, _ = _load_docs()
    scenes[0]["source_image_sha256"] = "abc"
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, scenes=scenes))
    assert any("source_image_sha256" in p for p in _problems_of(excinfo.value))


@pytest.mark.parametrize(
    "bad_path",
    [
        "/abs/path.png",
        "../escape.png",
        "images/../escape.png",
        "./here.png",
        "C:/windows.png",
        "file://host/x.png",
        "images//double.png",
        "images/trailing/",
        "",
    ],
)
def test_rejects_unsafe_image_paths(tmp_path: Path, bad_path: str) -> None:
    _, scenes, _ = _load_docs()
    scenes[0]["source_image_path"] = bad_path
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, scenes=scenes))
    assert any("source_image_path" in p for p in _problems_of(excinfo.value))


# --- outcomes -------------------------------------------------------------


def test_rejects_calibration_without_reference_count(tmp_path: Path) -> None:
    _, scenes, _ = _load_docs()
    scenes[0]["outcomes"][0].pop("reference_count")
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, scenes=scenes))
    assert any("required for calibration" in p for p in _problems_of(excinfo.value))


def test_rejects_coverage_with_reference_count(tmp_path: Path) -> None:
    _, scenes, _ = _load_docs()
    scenes[1]["outcomes"][0]["reference_count"] = {"numerator": 1, "denominator": 3}
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, scenes=scenes))
    assert any("forbidden for coverage" in p for p in _problems_of(excinfo.value))


def test_rejects_reference_counts_not_summing_to_one(tmp_path: Path) -> None:
    _, scenes, _ = _load_docs()
    scenes[0]["outcomes"][1]["reference_count"] = {"numerator": 1, "denominator": 3}
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, scenes=scenes))
    assert any("sum to exactly 1" in p for p in _problems_of(excinfo.value))


def test_accepts_nonuniform_exact_distribution(tmp_path: Path) -> None:
    """Two-dice-style skew: 1/3 + 2/3 must load and stay exact."""
    _, scenes, prompts = _load_docs()
    scenes[0]["outcomes"] = [
        {"label": "heads", "reference_count": {"numerator": 1, "denominator": 3}},
        {"label": "tails", "reference_count": {"numerator": 2, "denominator": 3}},
    ]
    result = benchmark.load(_write_package(tmp_path, scenes=scenes, prompts=prompts))
    assert result.scene_by_id("SYN-C01").reference_distribution() == {
        "heads": Fraction(1, 3),
        "tails": Fraction(2, 3),
    }


def test_rejects_duplicate_outcome_label(tmp_path: Path) -> None:
    _, scenes, _ = _load_docs()
    scenes[0]["outcomes"][1]["label"] = "heads"
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, scenes=scenes))
    assert any("duplicate outcome label" in p for p in _problems_of(excinfo.value))


def test_rejects_empty_outcomes(tmp_path: Path) -> None:
    _, scenes, _ = _load_docs()
    scenes[1]["outcomes"] = []
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, scenes=scenes))
    assert any("outcomes" in p for p in _problems_of(excinfo.value))


# --- prompt sets ----------------------------------------------------------


def test_rejects_unknown_scene_in_prompt_set(tmp_path: Path) -> None:
    _, _, prompts = _load_docs()
    prompts.append(dict(prompts[0], scene_id="GHOST-01"))
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, prompts=prompts))
    assert any("unknown scene" in p for p in _problems_of(excinfo.value))


def test_rejects_scene_missing_from_prompt_set(tmp_path: Path) -> None:
    _, _, prompts = _load_docs()
    prompts.pop()
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, prompts=prompts))
    assert any("without prompt rows" in p for p in _problems_of(excinfo.value))


def test_rejects_prompt_label_outside_vocabulary(tmp_path: Path) -> None:
    _, _, prompts = _load_docs()
    prompts[0]["outcome_prompts"].append(
        {"outcome_label": "edge", "prompt": "The coin lands on its edge."}
    )
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, prompts=prompts))
    assert any("not in scene vocabulary" in p for p in _problems_of(excinfo.value))


def test_rejects_missing_prompt_for_outcome(tmp_path: Path) -> None:
    _, _, prompts = _load_docs()
    prompts[0]["outcome_prompts"].pop()
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, prompts=prompts))
    assert any("missing prompt" in p for p in _problems_of(excinfo.value))


# --- parse level and error presentation -----------------------------------


def test_rejects_duplicate_json_key(tmp_path: Path) -> None:
    raw = (
        '{"scene_id":"SYN-C01","scene_id":"SYN-C01","split":"calibration",'
        '"split_order":0,"action":"a","source_image_path":"images/SYN-C01.png",'
        '"base_prompt":"b","outcomes":[{"label":"heads",'
        '"reference_count":{"numerator":1,"denominator":1}}]}'
    )
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, raw_scenes_text=raw + "\n"))
    assert any("duplicate JSON key" in p for p in _problems_of(excinfo.value))


def test_collects_all_problems_across_files(tmp_path: Path) -> None:
    manifest, scenes, _ = _load_docs()
    manifest["dataset_id"] = ""  # manifest defect
    scenes[0]["action"] = ""  # scene defect
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, manifest=manifest, scenes=scenes))
    problems = _problems_of(excinfo.value)
    assert any("dataset_id" in p for p in problems)
    assert any("action" in p for p in problems)
    assert len(excinfo.value.problems) >= 2


def test_truncates_problem_rendering_at_fifty(tmp_path: Path) -> None:
    _, scenes, _ = _load_docs()
    for index in range(60):
        scenes.append(
            dict(
                scenes[1],
                scene_id=f"SYN-T{index:02d}",
                split_order=index,
                action="",  # every extra scene has one defect
            )
        )
    with pytest.raises(BenchmarkError) as excinfo:
        benchmark.load(_write_package(tmp_path, scenes=scenes))
    assert len(excinfo.value.problems) > 50
    assert "...and " in str(excinfo.value)


def test_error_hierarchy() -> None:
    assert issubclass(BenchmarkError, ValidationError)
    assert issubclass(ValidationError, PawbenchError)
    problem = ValidationProblem(where="x", detail="y")
    assert problem == ValidationProblem(where="x", detail="y")
