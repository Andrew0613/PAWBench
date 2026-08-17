from __future__ import annotations

import json
from pathlib import Path

import pytest

import pawbench.evaluation as evaluation


def write_benchmark(root: Path) -> None:
    root.mkdir()
    (root / "tiny.mp4").write_bytes(b"synthetic-video-fixture")
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": "pawbench.benchmark_inputs/v1", "scene_table": "scenes.jsonl"}),
        encoding="utf-8",
    )
    scenes = []
    for track in ("calibration", "coverage"):
        for index in range(25):
            scene = {
                "scene_id": f"{'A' if track == 'calibration' else 'C'}-{index:02d}",
                "split": track,
                "action": "Roll the object once.",
                "source_image_path": f"images/{track}-{index:02d}.png",
                "outcome_labels": ["a", "b"],
            }
            if track == "calibration":
                scene["reference_distribution"] = {"a": 0.5, "b": 0.5}
            scenes.append(scene)
    (root / "scenes.jsonl").write_text(
        "\n".join(json.dumps(scene) for scene in scenes) + "\n", encoding="utf-8"
    )


def videos(root: Path) -> list[dict]:
    return [
        {
            "sample_id": f"{track}-{index:02d}__r{repeat:03d}",
            "scene_id": f"{'A' if track == 'calibration' else 'C'}-{index:02d}",
            "repeat_index": repeat,
            "video_path": root / "tiny.mp4",
        }
        for track in ("calibration", "coverage")
        for index in range(25)
        for repeat in range(50)
    ]


def fake_judge(item: dict, config: object) -> dict:
    assert Path(item["video_path"]).is_file()
    label = item["scene"]["outcome_labels"][item["repeat_index"] % 2]
    return {
        "sample_id": item["sample_id"],
        "scene_id": item["scene_id"],
        "track": item["track"],
        "model_or_lane": item["model_or_lane"],
        "repeat_index": item["repeat_index"],
        "observation": "outcome",
        "outcome_label": label,
        "outcome_readout": {"scene_id": item["scene_id"], "outcome_label": label},
        "trustworthiness_audit": {"scene_id": item["scene_id"], "status": "TRUSTED"},
    }


def call(root: Path, rows: list[dict]) -> dict:
    return evaluation.evaluate(
        root,
        rows,
        model_or_lane="model-x",
        vlm={"base_url": "https://judge.example/v1", "model": "judge", "api_key_env": "TEST_KEY"},
    )


def test_public_evaluation_journey_produces_judgments_and_official_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "benchmark"
    write_benchmark(root)
    monkeypatch.setattr(evaluation, "judge_item", fake_judge)

    result = call(root, videos(root))

    assert result["status"] == "ok"
    assert len(result["rows"]) == 2500
    assert result["metrics"]["status"] == "ok"
    model = result["metrics"]["tracks"]
    assert model["calibration"]["models"]["model-x"]["track_average"]["value"] == 0.0
    assert model["coverage"]["models"]["model-x"]["track_average"]["value"] == 100.0


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_missing_or_duplicate_videos_block_without_shrinking_the_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "benchmark"
    write_benchmark(root)
    rows = videos(root)
    if mutation == "missing":
        rows.pop()
    else:
        rows.append(dict(rows[0]))
    monkeypatch.setattr(evaluation, "judge_item", fake_judge)

    result = call(root, rows)

    assert result["status"] == "blocked"
    assert len(result["rows"]) == 2500
    assert result["metrics"]["status"] == "blocked"
    assert any(mutation in blocker for blocker in result["blockers"])
