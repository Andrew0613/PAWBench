"""The complete public PAWBench evaluation journey."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pawbench.metrics import EXPECTED_ROLLOUTS, compute_metrics, validate_scene_policy
from pawbench.paweval.evidence.extraction import extract_video_frames
from pawbench.paweval.evidence.package import EvidencePackage
from pawbench.paweval.evidence.sampling import FrameSamplingSpec
from pawbench.paweval.evidence.source_image import SourceImageIdentity
from pawbench.paweval.judge.client import OpenAICompatibleJudgeClient, ProviderConfig
from pawbench.paweval.judge.requests import RowIdentity
from pawbench.paweval.judgment import JudgmentConfig, JudgmentPreflightError, JudgmentSample, judge
from pawbench.paweval.rubrics.loader import load_rubric


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
    ``sample_id``, ``scene_id``, ``repeat_index``, and ``video_path``. PAWEval
    turns each supplied video plus its source image into an evidence
    package, judges it with the scene's two rubrics, and returns metric-ready
    rows without provider payloads or local media paths.
    """

    if not isinstance(model_or_lane, str) or not model_or_lane:
        raise ValueError("model_or_lane must be a nonempty string")
    root = Path(benchmark_path)
    policy, source_paths = _read_benchmark(root, model_or_lane)
    provider = _provider_config(vlm)
    expected = {
        (scene["scene_id"], repeat): scene
        for track in policy["tracks"].values()
        for scene in track["scenes"]
        for repeat in scene["expected_repeat_indices"]
    }
    supplied, blockers = _index_videos(videos, expected)
    rows: list[dict[str, Any]] = []
    samples: list[JudgmentSample] = []

    with TemporaryDirectory(prefix="pawbench-paweval-") as temporary_dir:
        frame_root = Path(temporary_dir)
        for (scene_id, repeat), scene in expected.items():
            video = supplied.get((scene_id, repeat))
            fallback_id = f"{model_or_lane}::{scene_id}::r{repeat:03d}"
            if video is None:
                blockers.append(f"missing_video:{scene_id}:{repeat}")
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
            try:
                samples.append(
                    _build_sample(
                        video=video,
                        scene=scene,
                        source_image_path=source_paths[scene_id],
                        model_or_lane=model_or_lane,
                        frame_root=frame_root,
                    )
                )
            except (OSError, RuntimeError, ValueError) as exc:
                rows.append(
                    _failure(
                        str(video["sample_id"]),
                        scene_id,
                        scene["track"],
                        model_or_lane,
                        repeat,
                        f"evidence_build_failed:{type(exc).__name__}",
                    )
                )

        if samples:
            try:
                rows.extend(_judge_samples(samples, provider))
            except JudgmentPreflightError as exc:
                rows.extend(
                    _failure(
                        sample.row_identity.sample_id,
                        sample.row_identity.scene_id,
                        sample.track,
                        sample.row_identity.model_or_lane,
                        int(sample.row_identity.repeat_index),
                        str(exc),
                    )
                    for sample in samples
                )

    metrics = compute_metrics(rows, policy)
    blockers = sorted(set(blockers + list(metrics["blockers"])))
    return {
        "status": "blocked" if blockers else "ok",
        "blockers": blockers,
        "rows": rows,
        "metrics": metrics,
    }


def _build_sample(
    *,
    video: Mapping[str, Any],
    scene: Mapping[str, Any],
    source_image_path: Path,
    model_or_lane: str,
    frame_root: Path,
) -> JudgmentSample:
    sample_id = str(video["sample_id"])
    scene_id = str(scene["scene_id"])
    video_path = Path(str(video["video_path"]))
    frames = extract_video_frames(
        sample_id=sample_id,
        video_path=video_path,
        output_dir=frame_root,
        sampling=FrameSamplingSpec(),
    )
    source = _source_identity(source_image_path)
    package = EvidencePackage(
        sample_id=sample_id,
        scene_id=scene_id,
        frames=frames,
        source_image=source,
    )
    return JudgmentSample(
        package=package,
        track=scene["track"],
        row_identity=RowIdentity.from_row(video, model_or_lane=model_or_lane),
    )


def _source_identity(path: Path) -> SourceImageIdentity:
    if path.is_file():
        return SourceImageIdentity(attached=True, uri=path.resolve().as_uri())
    return SourceImageIdentity(attached=False, absent_reason="source_image_missing_in_local_package")


def _judge_samples(samples: list[JudgmentSample], provider: ProviderConfig) -> list[dict[str, Any]]:
    return judge(
        samples=samples,
        adapter=OpenAICompatibleJudgeClient(provider),
        config=JudgmentConfig(),
    ).normalized_rows()


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
    scene_ids: set[str] = set()
    for source in scenes:
        if not isinstance(source, Mapping):
            raise ValueError("benchmark scene rows must be JSON objects")
        track = source.get("split")
        scene_id = source.get("scene_id")
        labels = source.get("outcome_labels")
        if track not in tracks or not isinstance(scene_id, str) or not isinstance(labels, list):
            raise ValueError("benchmark scene has an invalid policy")
        if scene_id in scene_ids:
            raise ValueError(f"benchmark package contains duplicate scene_id: {scene_id}")
        scene_ids.add(scene_id)
        scene = {
            "scene_id": scene_id,
            "track": track,
            "group": source.get("group") or track,
            "expected_repeat_indices": list(range(EXPECTED_ROLLOUTS)),
        }
        if track == "calibration":
            scene["reference_distribution"] = source.get("reference_distribution")
        else:
            scene["support_labels"] = labels
        tracks[track]["scenes"].append(scene)
        source_paths[scene_id] = root / str(source.get("source_image_path") or "")
    policy = {"model_or_lanes": [model_or_lane], "tracks": tracks}
    validate_scene_policy(policy)
    for scene_id in scene_ids:
        for axis in ("outcome", "trustworthiness"):
            try:
                load_rubric(axis, scene_id)
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(f"benchmark scene has no valid {axis} rubric: {scene_id}") from exc
    return policy, source_paths


def _provider_config(values: Mapping[str, Any]) -> ProviderConfig:
    if not isinstance(values, Mapping):
        raise ValueError("vlm must be an object")
    try:
        return ProviderConfig(
            provider=str(values.get("provider") or "openai_compatible"),
            model=str(values["model"]),
            base_url=str(values["base_url"]),
            api_key_env=str(values["api_key_env"]),
            timeout=int(values.get("timeout", 120)),
            max_tokens=int(values.get("max_tokens", 2048)),
        )
    except (KeyError, TypeError, ValueError) as exc:
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
        supplied[slot] = {"sample_id": sample_id, "scene_id": scene_id, "repeat_index": repeat, "video_path": str(video_path)}
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
