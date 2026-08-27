from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "examples" / "generate_openrouter.py"
SPEC = importlib.util.spec_from_file_location("generate_openrouter", SCRIPT)
assert SPEC and SPEC.loader
generate_openrouter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generate_openrouter
SPEC.loader.exec_module(generate_openrouter)


def write_benchmark(root: Path) -> None:
    root.mkdir()
    images = root / "source_images"
    images.mkdir()
    rows = []
    for index in range(50):
        scene_id = f"S-{index:02d}"
        (images / f"{scene_id}.png").write_bytes(b"image")
        rows.append(
            {
                "scene_id": scene_id,
                "source_image_path": f"source_images/{scene_id}.png",
                "action": f"Perform action {index}.",
            }
        )
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


def request_options(
    *,
    model_id: str = "google/veo-3.1-lite",
    source_image_base_url: str = "https://example.test/pawbench/",
    duration: int | None = None,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    seed: int | None = None,
) -> generate_openrouter.RequestOptions:
    return generate_openrouter.RequestOptions(
        model_id=model_id,
        source_image_base_url=source_image_base_url,
        duration=duration,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        seed=seed,
    )


def test_builds_image_to_video_requests_and_evaluator_paths(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    write_benchmark(benchmark)

    scenes = generate_openrouter.load_scenes(benchmark, ["S-01"])
    tasks = generate_openrouter.build_tasks(scenes, tmp_path / "rollouts", num_rollouts=2)
    options = request_options(
        duration=4,
        resolution="720p",
        aspect_ratio="16:9",
        seed=100,
    )
    payload = generate_openrouter.build_request(tasks[1], options)

    assert tasks[1].output_path == tmp_path / "rollouts" / "S-01" / "r001.mp4"
    assert payload == {
        "model": "google/veo-3.1-lite",
        "prompt": "Perform action 1.",
        "generate_audio": False,
        "frame_images": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.test/pawbench/source_images/S-01.png"},
                "frame_type": "first_frame",
            }
        ],
        "duration": 4,
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "seed": 101,
    }


def test_preview_is_default_and_never_requires_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    benchmark = tmp_path / "benchmark"
    write_benchmark(benchmark)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    status = generate_openrouter.main(
        [
            "--benchmark",
            str(benchmark),
            "--output",
            str(tmp_path / "rollouts"),
            "--model-id",
            "google/veo-3.1-lite",
            "--scene",
            "S-00",
        ]
    )

    assert status == 0
    assert "preview only" in capsys.readouterr().out
    assert not (tmp_path / "rollouts").exists()


def test_execute_saves_job_before_polling_and_downloads_video(tmp_path: Path) -> None:
    scene = generate_openrouter.Scene(
        scene_id="S-00",
        source_image_path="source_images/S-00.png",
        prompt="Perform action 0.",
    )
    task = generate_openrouter.GenerationTask(
        scene=scene,
        repeat_index=0,
        output_path=tmp_path / "rollouts" / "S-00" / "r000.mp4",
    )
    calls: list[tuple[str, object]] = []

    class FakeClient:
        def submit(self, payload: dict[str, object]) -> dict[str, object]:
            calls.append(("submit", payload))
            return {"id": "job-1", "polling_url": "/api/v1/videos/job-1", "status": "pending"}

        def poll(self, job: dict[str, object], **_: object) -> dict[str, object]:
            checkpoint = generate_openrouter.checkpoint_path(task)
            calls.append(("checkpoint_before_poll", checkpoint.is_file()))
            return {**job, "status": "completed"}

        def download(self, job: dict[str, object], output_path: Path) -> None:
            calls.append(("download", job["id"]))
            output_path.write_bytes(b"video")

    written = generate_openrouter.generate(
        [task],
        client=FakeClient(),
        request_options=request_options(),
        poll_interval=0,
        max_polls=1,
        overwrite=False,
    )

    assert written == 1
    assert calls[1] == ("checkpoint_before_poll", True)
    assert task.output_path.read_bytes() == b"video"
    checkpoint = generate_openrouter.checkpoint_path(task)
    record = json.loads(checkpoint.read_text())
    assert record["schema_version"] == generate_openrouter.CHECKPOINT_SCHEMA
    assert record["job"]["status"] == "completed"


def test_execute_resumes_saved_job_without_resubmitting(tmp_path: Path) -> None:
    scene = generate_openrouter.Scene("S-00", "source_images/S-00.png", "Perform action 0.")
    task = generate_openrouter.GenerationTask(scene, 0, tmp_path / "rollouts/S-00/r000.mp4")
    checkpoint = generate_openrouter.checkpoint_path(task)
    checkpoint.parent.mkdir(parents=True)
    options = request_options()
    payload = generate_openrouter.build_request(task, options)
    generate_openrouter._write_checkpoint(
        checkpoint,
        payload,
        {"id": "job-1", "polling_url": "/api/v1/videos/job-1", "status": "pending"},
    )

    class FakeClient:
        def submit(self, _: dict[str, object]) -> dict[str, object]:
            pytest.fail("saved jobs must be resumed instead of resubmitted")

        def poll(self, job: dict[str, object], **_: object) -> dict[str, object]:
            return {**job, "status": "completed"}

        def download(self, _: dict[str, object], output_path: Path) -> None:
            output_path.write_bytes(b"video")

    written = generate_openrouter.generate(
        [task],
        client=FakeClient(),
        request_options=options,
        poll_interval=0,
        max_polls=1,
        overwrite=False,
        before_submit=lambda: pytest.fail("resuming must not require model discovery"),
    )

    assert written == 1
    assert task.output_path.read_bytes() == b"video"


def test_resume_rejects_a_checkpoint_from_different_options(tmp_path: Path) -> None:
    scene = generate_openrouter.Scene("S-00", "source_images/S-00.png", "Perform action 0.")
    task = generate_openrouter.GenerationTask(scene, 0, tmp_path / "rollouts/S-00/r000.mp4")
    checkpoint = generate_openrouter.checkpoint_path(task)
    old_payload = generate_openrouter.build_request(task, request_options(duration=4))
    generate_openrouter._write_checkpoint(
        checkpoint,
        old_payload,
        {"id": "job-1", "polling_url": "/api/v1/videos/job-1", "status": "pending"},
    )

    with pytest.raises(RuntimeError, match="different generation options"):
        generate_openrouter.generate(
            [task],
            client=object(),
            request_options=request_options(duration=8),
            poll_interval=0,
            max_polls=1,
            overwrite=False,
        )


def test_existing_output_still_requires_a_matching_checkpoint(tmp_path: Path) -> None:
    scene = generate_openrouter.Scene("S-00", "source_images/S-00.png", "Perform action 0.")
    task = generate_openrouter.GenerationTask(scene, 0, tmp_path / "rollouts/S-00/r000.mp4")
    task.output_path.parent.mkdir(parents=True)
    task.output_path.write_bytes(b"video")

    with pytest.raises(RuntimeError, match="has no request checkpoint"):
        generate_openrouter.generate(
            [task],
            client=object(),
            request_options=request_options(),
            poll_interval=0,
            max_polls=1,
            overwrite=False,
        )


def test_rejects_scene_ids_that_escape_the_output_tree(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    write_benchmark(benchmark)
    table = benchmark / "scenes.jsonl"
    rows = [json.loads(line) for line in table.read_text().splitlines()]
    rows[0]["scene_id"] = "../escaped"
    table.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid or duplicate scene IDs"):
        generate_openrouter.load_scenes(benchmark)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://example.test/pawbench/",
        "https://user:pass@example.test/pawbench/",
        "https://example.test/pawbench/?token=secret",
        "https://example.test/pawbench/#fragment",
    ],
)
def test_rejects_untrusted_source_image_base_urls(base_url: str, tmp_path: Path) -> None:
    scene = generate_openrouter.Scene("S-00", "source images/frame 1.png", "Action")
    task = generate_openrouter.GenerationTask(scene, 0, tmp_path / "S-00/r000.mp4")

    with pytest.raises(ValueError, match="HTTPS|credentials"):
        generate_openrouter.build_request(task, request_options(source_image_base_url=base_url))


def test_source_image_path_is_percent_encoded(tmp_path: Path) -> None:
    scene = generate_openrouter.Scene("S-00", "source images/frame #1.png", "Action")
    task = generate_openrouter.GenerationTask(scene, 0, tmp_path / "S-00/r000.mp4")

    payload = generate_openrouter.build_request(task, request_options())

    assert payload["frame_images"][0]["image_url"]["url"].endswith(
        "/source%20images/frame%20%231.png"
    )


def test_model_capability_validation_fails_closed() -> None:
    model = {
        "id": "video-model",
        "supported_frame_images": ["first_frame"],
        "supported_durations": [4],
        "supported_resolutions": ["720p"],
        "supported_aspect_ratios": ["16:9"],
    }
    generate_openrouter._validate_model(
        [model],
        model_id="video-model",
        duration=4,
        resolution="720p",
        aspect_ratio="16:9",
    )

    with pytest.raises(ValueError, match="does not support duration"):
        generate_openrouter._validate_model(
            [model],
            model_id="video-model",
            duration=8,
            resolution=None,
            aspect_ratio=None,
        )
    with pytest.raises(ValueError, match="cannot verify resolution"):
        generate_openrouter._validate_model(
            [{key: value for key, value in model.items() if key != "supported_resolutions"}],
            model_id="video-model",
            duration=None,
            resolution="720p",
            aspect_ratio=None,
        )


def test_authenticated_requests_do_not_follow_redirects() -> None:
    handler = generate_openrouter._NoRedirectHandler()

    assert handler.redirect_request(None, None, 302, "Found", {}, "https://attacker.test") is None


def test_output_directory_allows_only_one_generator(tmp_path: Path) -> None:
    with generate_openrouter._RunLock(tmp_path):
        with pytest.raises(RuntimeError, match="another OpenRouter generator"):
            with generate_openrouter._RunLock(tmp_path):
                pytest.fail("the second generator must not acquire the same output lock")


def test_polling_never_sends_the_api_key_to_an_untrusted_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = generate_openrouter.OpenRouterVideoClient("secret")
    monkeypatch.setattr(
        client,
        "_json",
        lambda *_args, **_kwargs: pytest.fail("untrusted polling URL must not be requested"),
    )

    with pytest.raises(RuntimeError, match="untrusted polling_url"):
        client.poll(
            {
                "id": "job-1",
                "polling_url": "https://attacker.example/steal-key",
                "status": "pending",
            },
            poll_interval=0,
            max_polls=1,
        )
