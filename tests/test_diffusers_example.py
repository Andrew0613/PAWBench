from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "examples" / "generate_diffusers.py"
SPEC = importlib.util.spec_from_file_location("generate_diffusers", SCRIPT)
assert SPEC and SPEC.loader
generate_diffusers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generate_diffusers
SPEC.loader.exec_module(generate_diffusers)


def write_benchmark(root: Path) -> None:
    root.mkdir()
    images = root / "source_images"
    prompts = root / "prompts"
    images.mkdir()
    prompts.mkdir()
    rows = []
    for index in range(50):
        scene_id = f"S-{index:02d}"
        (images / f"{scene_id}.png").write_bytes(b"image")
        row = {
            "scene_id": scene_id,
            "source_image_path": f"source_images/{scene_id}.png",
            "action": f"Perform action {index}.",
        }
        if index == 1:
            (prompts / f"{scene_id}.txt").write_text("Prompt from file.\n", encoding="utf-8")
            row["prompt_path"] = f"prompts/{scene_id}.txt"
        rows.append(row)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "pawbench.benchmark_inputs/v1",
                "scene_table": "scenes.jsonl",
            }
        ),
        encoding="utf-8",
    )
    (root / "scenes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_load_scenes_and_build_evaluator_compatible_paths(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    write_benchmark(benchmark)

    scenes = generate_diffusers.load_scenes(benchmark, ["S-01", "S-00", "S-01"])
    tasks = generate_diffusers.build_tasks(scenes, tmp_path / "rollouts", num_rollouts=2, seed=100)

    assert [scene.scene_id for scene in scenes] == ["S-01", "S-00"]
    assert scenes[0].prompt == "Prompt from file."
    assert [task.seed for task in tasks] == [100, 101, 100, 101]
    assert tasks[0].output_path == tmp_path / "rollouts" / "S-01" / "r000.mp4"


def test_generate_uses_configured_pipeline_and_writes_video(tmp_path: Path) -> None:
    image = tmp_path / "source.png"
    image.write_bytes(b"image")
    scene = generate_diffusers.Scene("A-01", image, "Flick the coin once.")
    task = generate_diffusers.GenerationTask(scene, 0, 42, tmp_path / "A-01" / "r000.mp4")
    calls: dict[str, object] = {}

    class FakeGenerator:
        def __init__(self, *, device: str) -> None:
            calls["device"] = device

        def manual_seed(self, seed: int) -> "FakeGenerator":
            calls["seed"] = seed
            return self

    class FakePipe:
        def __call__(self, **kwargs: object) -> SimpleNamespace:
            calls["pipeline"] = kwargs
            return SimpleNamespace(frames=[["frame-0", "frame-1"]])

    def export(frames: list[str], path: str, *, fps: int) -> None:
        calls["export"] = (frames, path, fps)
        Path(path).write_bytes(b"video")

    written = generate_diffusers.generate(
        [task],
        model_id="example/model",
        dtype_name="bfloat16",
        device="cuda",
        device_map="cuda",
        fps=16,
        overwrite=False,
        runtime=(
            SimpleNamespace(Generator=FakeGenerator),
            FakePipe(),
            lambda path: f"image:{path}",
            export,
        ),
    )

    assert written == 1
    assert calls["seed"] == 42
    assert calls["pipeline"]["prompt"] == "Flick the coin once."
    assert calls["export"] == (["frame-0", "frame-1"], str(task.output_path), 16)
    assert task.output_path.read_bytes() == b"video"


def test_dry_run_never_imports_diffusers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    benchmark = tmp_path / "benchmark"
    write_benchmark(benchmark)

    status = generate_diffusers.main(
        [
            "--benchmark",
            str(benchmark),
            "--output",
            str(tmp_path / "rollouts"),
            "--model-id",
            "example/model",
            "--scene",
            "S-00",
            "--num-rollouts",
            "1",
            "--dry-run",
        ]
    )

    assert status == 0
    assert "planned rollouts: 1 across 1 scene(s)" in capsys.readouterr().out


def test_scene_selection_must_be_explicit(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    write_benchmark(benchmark)

    with pytest.raises(SystemExit) as exc_info:
        generate_diffusers.parse_args(
            ["--benchmark", str(benchmark), "--output", str(tmp_path / "rollouts")]
        )
    assert exc_info.value.code == 2
