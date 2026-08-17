"""The complete public PAWBench evaluation journey."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pawbench._paweval._client import VLMConfig
from pawbench._paweval._judge import judge_item
from pawbench.metrics import EXPECTED_ROLLOUTS, compute_metrics


def evaluate(
    benchmark_path: str | Path,
    videos: Sequence[Mapping[str, Any]],
    *,
    model_or_lane: str,
    vlm: Mapping[str, Any],
) -> dict[str, Any]:
    """Judge a complete local PAWBench video grid and compute official metrics.

    ``benchmark_path`` is an already-downloaded package containing
    ``manifest.json`` and its ``scenes.jsonl`` table. Each video row provides
    `sample_id`, `scene_id`, `repeat_index`, and `video_path`. The benchmark's
    50 scenes and the official 50 repeats per scene define the denominator.

    Evaluation media is sent to the configured VLM provider. Returned rows do
    not contain credentials, provider bodies, or local media paths.
    """

    if not isinstance(model_or_lane, str) or not model_or_lane:
        raise ValueError("model_or_lane must be a nonempty string")
    root = Path(benchmark_path)
    policy, source_paths = _read_benchmark(root, model_or_lane)
    config = _vlm_config(vlm)
    expected = {
        (scene["scene_id"], repeat): scene
        for track in policy["tracks"].values()
        for scene in track["scenes"]
        for repeat in scene["expected_repeat_indices"]
    }
    supplied, item_blockers = _index_videos(videos, expected)

    rows: list[dict[str, Any]] = []
    for (scene_id, repeat), scene in expected.items():
        video = supplied.get((scene_id, repeat))
        fallback_id = f"{model_or_lane}::{scene_id}::r{repeat:03d}"
        if video is None:
            item_blockers.append(f"missing_video:{scene_id}:{repeat}")
            rows.append(_failure(fallback_id, scene_id, scene["track"], model_or_lane, repeat, "missing_video"))
            continue
        if "error" in video:
            rows.append(
                _failure(
                    str(video.get("sample_id") or fallback_id),
                    scene_id,
                    scene["track"],
                    model_or_lane,
                    repeat,
                    str(video["error"]),
                )
            )
            continue
        rows.append(
            judge_item(
                {
                    "sample_id": video["sample_id"],
                    "scene_id": scene_id,
                    "track": scene["track"],
                    "model_or_lane": model_or_lane,
                    "repeat_index": repeat,
                    "source_image_path": source_paths[scene_id],
                    "video_path": video["video_path"],
                    "scene": scene,
                },
                config,
            )
        )

    metrics = compute_metrics(rows, policy)
    blockers = sorted(set(item_blockers + list(metrics["blockers"])))
    return {
        "status": "blocked" if blockers else "ok",
        "blockers": blockers,
        "rows": rows,
        "metrics": metrics,
    }


def _read_benchmark(root: Path, model_or_lane: str) -> tuple[dict[str, Any], dict[str, Path]]:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        scene_table = root / str(manifest["scene_table"])
        scenes = [json.loads(line) for line in scene_table.read_text(encoding="utf-8").splitlines() if line]
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid local PAWBench package") from exc
    if manifest.get("schema_version") != "pawbench.benchmark_inputs/v1" or len(scenes) != 50:
        raise ValueError("benchmark package must contain the released 50-scene contract")

    tracks: dict[str, dict[str, list[dict[str, Any]]]] = {
        "calibration": {"scenes": []},
        "coverage": {"scenes": []},
    }
    source_paths: dict[str, Path] = {}
    for source in scenes:
        if not isinstance(source, Mapping):
            raise ValueError("benchmark scene rows must be JSON objects")
        track = source.get("split")
        scene_id = source.get("scene_id")
        labels = source.get("outcome_labels")
        if track not in tracks or not isinstance(scene_id, str) or not isinstance(labels, list):
            raise ValueError("benchmark scene has an invalid policy")
        scene = {
            "scene_id": scene_id,
            "track": track,
            "group": source.get("group") or track,
            "action": source.get("action") or "",
            "outcome_labels": labels,
            "expected_repeat_indices": list(range(EXPECTED_ROLLOUTS)),
        }
        if track == "calibration":
            scene["reference_distribution"] = source.get("reference_distribution")
        else:
            scene["support_labels"] = labels
        tracks[track]["scenes"].append(scene)
        source_paths[scene_id] = root / str(source.get("source_image_path") or "")
    return {"model_or_lanes": [model_or_lane], "tracks": tracks}, source_paths


def _vlm_config(values: Mapping[str, Any]) -> VLMConfig:
    if not isinstance(values, Mapping):
        raise ValueError("vlm must be an object")
    try:
        return VLMConfig(**dict(values))
    except TypeError as exc:
        raise ValueError("vlm must contain base_url, model, and api_key_env") from exc


def _index_videos(
    videos: Sequence[Mapping[str, Any]], expected: Mapping[tuple[str, int], Mapping[str, Any]]
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
    supplied: dict[tuple[str, int], dict[str, Any]] = {}
    blockers: list[str] = []
    sample_ids: dict[str, set[str]] = {}
    for video in videos:
        if not isinstance(video, Mapping):
            blockers.append("malformed_video_item")
            continue
        scene_id, repeat, sample_id, video_path = (
            video.get("scene_id"),
            video.get("repeat_index"),
            video.get("sample_id"),
            video.get("video_path"),
        )
        if not isinstance(scene_id, str) or not isinstance(repeat, int) or (scene_id, repeat) not in expected:
            blockers.append("unexpected_video_item")
            continue
        slot = (scene_id, repeat)
        if slot in supplied:
            supplied[slot] = {"sample_id": sample_id, "error": "duplicate_video_item"}
            blockers.append(f"duplicate_video_item:{scene_id}:{repeat}")
            continue
        if not isinstance(sample_id, str) or not sample_id or not isinstance(video_path, (str, Path)):
            supplied[slot] = {"sample_id": sample_id, "error": "malformed_video_item"}
            blockers.append(f"malformed_video_item:{scene_id}:{repeat}")
            continue
        seen = sample_ids.setdefault(scene_id, set())
        if sample_id in seen:
            supplied[slot] = {"sample_id": sample_id, "error": "duplicate_sample_id"}
            blockers.append(f"duplicate_sample_id:{scene_id}:{sample_id}")
            continue
        seen.add(sample_id)
        supplied[slot] = {"sample_id": sample_id, "video_path": str(video_path)}
    return supplied, blockers


def _failure(
    sample_id: str, scene_id: str, track: str, model_or_lane: str, repeat_index: int, code: str
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "track": track,
        "model_or_lane": model_or_lane,
        "repeat_index": repeat_index,
        "observation": "infrastructure_failure",
        "failure_code": code,
    }
