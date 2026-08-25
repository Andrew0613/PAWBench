from __future__ import annotations

import json
from pathlib import Path

import pytest

import pawbench.evaluation as evaluation
import pawbench.metrics as metrics
from pawbench.paweval.evidence.extraction import safe_name
from pawbench.paweval.evidence.frames import EvidenceFrame
from pawbench.paweval.judge.responses import JudgeResponse, parse_judge_response
from pawbench.paweval.rubrics.loader import load_rubric


def _scene_ids(track: str) -> list[str]:
    outcome_root = Path(evaluation.__file__).parent / "paweval" / "rubrics" / "outcome"
    all_ids = sorted(path.stem for path in outcome_root.glob("*.yaml"))
    return (
        [scene_id for scene_id in all_ids if scene_id.startswith("A")]
        if track == "calibration"
        else [scene_id for scene_id in all_ids if not scene_id.startswith("A")]
    )


def write_benchmark(root: Path) -> None:
    root.mkdir()
    images = root / "images"
    images.mkdir()
    scenes = []
    for track in ("calibration", "coverage"):
        for scene_id in _scene_ids(track):
            labels = list(load_rubric("outcome", scene_id)["canonical_labels"])
            source = images / f"{scene_id}.png"
            source.write_bytes(b"source-image")
            scene = {
                "scene_id": scene_id,
                "split": track,
                "action": "Perform the scene action.",
                "source_image_path": str(source.relative_to(root)),
                "outcome_labels": labels if track == "calibration" else [labels[0]],
            }
            if track == "calibration":
                scene["reference_distribution"] = {labels[0]: 1.0}
            scenes.append(scene)
    assert len(scenes) == 50
    (root / "tiny.mp4").write_bytes(b"fixture")
    (root / "manifest.json").write_text(
        json.dumps(
            {"schema_version": "pawbench.benchmark_inputs/v1", "scene_table": "scenes.jsonl"}
        ),
        encoding="utf-8",
    )
    (root / "scenes.jsonl").write_text(
        "\n".join(json.dumps(scene) for scene in scenes) + "\n", encoding="utf-8"
    )


def videos(root: Path) -> list[dict]:
    return [
        {
            "sample_id": f"model-x::{scene_id}::r{repeat:03d}",
            "scene_id": scene_id,
            "repeat_index": repeat,
            "video_path": root / "tiny.mp4",
        }
        for track in ("calibration", "coverage")
        for scene_id in _scene_ids(track)
        for repeat in range(evaluation.EXPECTED_ROLLOUTS)
    ]


def fake_extract(*, sample_id: str, output_dir: Path, **_: object) -> tuple[EvidenceFrame, ...]:
    frame = output_dir / f"{safe_name(sample_id)}.jpg"
    frame.write_bytes(b"frame")
    return (
        EvidenceFrame(f"{sample_id}:initial", "initial", frame.as_uri(), 0.0),
        EvidenceFrame(f"{sample_id}:action", "action", frame.as_uri(), 0.5),
        EvidenceFrame(f"{sample_id}:terminal", "terminal", frame.as_uri(), 1.0),
    )


class FakeJudgeClient:
    def complete(self, request: object) -> JudgeResponse:
        axis = request.axis
        scene_id = request.scene_id
        if axis == "outcome_readout":
            payload = {
                "scene_id": scene_id,
                "outcome_readable": True,
                "outcome_in_schema": "IN_SCHEMA",
                "outcome_label": load_rubric("outcome", scene_id)["canonical_labels"][0],
                "observed_result": "fixture",
                "failure_axes": [],
            }
        else:
            payload = {
                "scene_id": scene_id,
                "status": "TRUSTED",
                "scene_grounding": {"status": "PASS"},
                "action_execution": {
                    "status": "PASS",
                    "target_hit": "YES",
                    "object_acquired": "YES",
                    "action_spec_followed": "YES",
                },
                "object_continuity": {"status": "PASS"},
                "physical_process": {"status": "PASS", "failure_phase": "NONE"},
                "failure_axes": [],
            }
        return parse_judge_response(json.dumps(payload), axis=axis, scene_id=scene_id)


def call(root: Path, rows: list[dict]) -> dict:
    return evaluation.evaluate(
        root,
        rows,
        model_or_lane="model-x",
        vlm={"base_url": "https://judge.example/v1", "model": "judge", "api_key_env": "TEST_KEY"},
    )


@pytest.fixture(autouse=True)
def fake_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "EXPECTED_ROLLOUTS", 1)
    monkeypatch.setattr(metrics, "EXPECTED_ROLLOUTS", 1)
    monkeypatch.setattr(evaluation, "extract_video_frames", fake_extract)
    monkeypatch.setattr(evaluation, "OpenAICompatibleJudgeClient", lambda _: FakeJudgeClient())
    monkeypatch.setenv("TEST_KEY", "test-key")


def test_public_evaluation_journey_runs_paweval_and_official_metrics(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    write_benchmark(root)

    result = call(root, videos(root))

    assert result["status"] == "ok"
    assert len(result["rows"]) == 50
    assert result["metrics"]["status"] == "ok"
    tracks = result["metrics"]["tracks"]
    assert tracks["calibration"]["models"]["model-x"]["track_average"]["value"] == 0.0
    assert tracks["coverage"]["models"]["model-x"]["track_average"]["value"] == 100.0


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_missing_or_duplicate_videos_block_without_shrinking_the_grid(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "benchmark"
    write_benchmark(root)
    rows = videos(root)
    if mutation == "missing":
        rows.pop()
    else:
        rows.append(dict(rows[0]))

    result = call(root, rows)

    assert result["status"] == "blocked"
    assert len(result["rows"]) == 50
    assert result["metrics"]["status"] == "blocked"
    assert any(mutation in blocker for blocker in result["blockers"])


def test_invalid_benchmark_contract_fails_before_media_or_judging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "benchmark"
    write_benchmark(root)
    scenes_path = root / "scenes.jsonl"
    scenes = [json.loads(line) for line in scenes_path.read_text(encoding="utf-8").splitlines()]
    scenes[0]["split"] = "coverage"
    scenes_path.write_text(
        "\n".join(json.dumps(scene) for scene in scenes) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        evaluation,
        "extract_video_frames",
        lambda **_: pytest.fail("invalid benchmark must fail before media extraction"),
    )

    with pytest.raises(ValueError, match="calibration track policy must declare exactly 25 scenes"):
        call(root, videos(root))


def test_missing_rubric_fails_before_media_or_judging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "benchmark"
    write_benchmark(root)
    scenes_path = root / "scenes.jsonl"
    scenes = [json.loads(line) for line in scenes_path.read_text(encoding="utf-8").splitlines()]
    scenes[0]["scene_id"] = "missing-rubric"
    scenes_path.write_text(
        "\n".join(json.dumps(scene) for scene in scenes) + "\n", encoding="utf-8"
    )
    rows = videos(root)
    rows[0] = {**rows[0], "scene_id": "missing-rubric"}
    monkeypatch.setattr(
        evaluation,
        "extract_video_frames",
        lambda **_: pytest.fail("missing rubric must fail before media extraction"),
    )

    with pytest.raises(
        ValueError, match="benchmark scene has no valid outcome rubric: missing-rubric"
    ):
        call(root, rows)


def test_missing_credentials_fail_before_media_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "benchmark"
    write_benchmark(root)
    monkeypatch.delenv("TEST_KEY")
    monkeypatch.setattr(
        evaluation,
        "extract_video_frames",
        lambda **_: pytest.fail("provider preflight must happen before media extraction"),
    )

    result = call(root, videos(root))

    assert result["status"] == "blocked"
    assert {row["failure_code"] for row in result["rows"]} == {"missing_credentials"}
    assert "provider:missing_credentials" in result["blockers"]


def test_output_checkpoint_resumes_completed_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "benchmark"
    output = tmp_path / "evaluation"
    write_benchmark(root)
    first = evaluation.evaluate(
        root,
        videos(root),
        model_or_lane="model-x",
        vlm={"base_url": "https://judge.example/v1", "model": "judge", "api_key_env": "TEST_KEY"},
        output_dir=output,
    )
    monkeypatch.setattr(
        evaluation,
        "extract_video_frames",
        lambda **_: pytest.fail("completed rows must resume without media extraction"),
    )
    monkeypatch.setattr(
        evaluation,
        "OpenAICompatibleJudgeClient",
        lambda _: pytest.fail("completed rows must resume without provider calls"),
    )

    resumed = evaluation.evaluate(
        root,
        videos(root),
        model_or_lane="model-x",
        vlm={"base_url": "https://judge.example/v1", "model": "judge", "api_key_env": "TEST_KEY"},
        output_dir=output,
    )

    assert first["status"] == resumed["status"] == "ok"
    assert first["rows"] == resumed["rows"]
    assert set(resumed["artifacts"]) == {"run", "checkpoint", "rows", "metrics"}
    assert len((output / "rows.jsonl").read_text(encoding="utf-8").splitlines()) == 50
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert len(run["benchmark"]["digest"]) == 64
    assert len(run["evaluator_digest"]) == 64
