"""Evaluate one model's local PAWBench rollouts.

Set the following environment variables before running this file:

    PAWBENCH_DATA_DIR      local directory downloaded from Hugging Face
    PAWBENCH_RESULTS_DIR   <scene_id>/r000.mp4 ... r049.mp4 directories
    PAWBENCH_MODEL         name recorded in the evaluation rows
    PAWBENCH_VLM_BASE_URL  OpenAI-compatible evaluator endpoint
    PAWBENCH_VLM_MODEL     VLM used by PAWEval
    PAWBENCH_VLM_API_KEY   key for that endpoint

Then run ``python examples/quickstart.py`` from the repository root. This is a
plain, editable Python example; PAWBench intentionally does not provide an
evaluator CLI or a model registry.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

try:
    from pawbench import evaluate
except ModuleNotFoundError as exc:
    if exc.name != "pawbench":
        raise
    raise SystemExit('Install PAWBench first: python -m pip install -e ".[eval]"') from exc

VIDEO_RE = re.compile(r"^r(\d{3,})$", re.IGNORECASE)
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".avi"}


def required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set {name}; see the README's 'Evaluate your videos' section.")
    return Path(value).expanduser()


def required_value(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set {name}; see the README's 'Evaluate your videos' section.")
    return value


def collect_video_items(results_dir: Path, model_or_lane: str) -> list[dict[str, object]]:
    """Discover ``<scene_id>/r###.<video extension>`` rollout files."""

    if not results_dir.is_dir():
        raise SystemExit(f"PAWBENCH_RESULTS_DIR does not exist: {results_dir}")
    items: list[dict[str, object]] = []
    for scene_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        for video in sorted(scene_dir.iterdir()):
            match = VIDEO_RE.fullmatch(video.stem)
            if not match or not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            items.append(
                {
                    "sample_id": f"{model_or_lane}::{scene_dir.name}::{video.stem}",
                    "scene_id": scene_dir.name,
                    "repeat_index": int(match.group(1)),
                    "video_path": str(video.resolve()),
                }
            )
    if not items:
        raise SystemExit(f"No <scene_id>/r###.mp4 rollouts found under {results_dir}")
    return items


def main() -> None:
    benchmark_dir = required_path("PAWBENCH_DATA_DIR")
    results_dir = required_path("PAWBENCH_RESULTS_DIR")
    model = required_value("PAWBENCH_MODEL")
    if not (benchmark_dir / "manifest.json").is_file():
        raise SystemExit(f"PAWBENCH_DATA_DIR is not a benchmark package: {benchmark_dir}")

    videos = collect_video_items(results_dir, model)
    print(f"Collected {len(videos)} videos from {results_dir}")
    result = evaluate(
        benchmark_dir,
        videos,
        model_or_lane=model,
        vlm={
            "base_url": required_value("PAWBENCH_VLM_BASE_URL"),
            "model": required_value("PAWBENCH_VLM_MODEL"),
            "api_key_env": "PAWBENCH_VLM_API_KEY",
        },
    )
    print(f"status: {result['status']}")
    for blocker in result["blockers"]:
        print(f"blocker: {blocker}")
    print(f"metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
