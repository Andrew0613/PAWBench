"""Minimal PAWBench evaluation workflow.

Runs the full local journey -- rubric rendering, VLM judgment, official
metrics -- over a local directory of generated videos. Nothing is hardcoded:
the benchmark package, the rollout directory, and the VLM endpoint come from
command-line flags or environment variables. Run from the repository root
after `pip install -e ".[eval]"`.

    export YOUR_VLM_API_KEY="..."
    python examples/quickstart.py \
        --benchmark ~/data/PAWBench \
        --results ~/results/my-model \
        --vlm-base-url https://your-vlm-provider.example/v1 \
        --vlm-model your-vlm-model \
        --vlm-api-key-env YOUR_VLM_API_KEY

Expected rollout layout (one model's rollouts under --results):

    <results>/
    └── <scene_id>/
        └── r000.mp4 ... r049.mp4

Every discovered video becomes one evaluation item. evaluate() still derives
the full 50-scene x 50-rollout grid from the benchmark package, so missing
videos surface as explicit rows instead of shrinking the denominator.

Flag defaults read $PAWBENCH_DATA_DIR, $PAWBENCH_RESULTS_DIR,
$PAWBENCH_VLM_BASE_URL, $PAWBENCH_VLM_MODEL, and $PAWBENCH_VLM_API_KEY_ENV.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from pawbench import evaluate

VIDEO_RE = re.compile(r"^r(\d{3,})$", re.IGNORECASE)
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".avi"}


def collect_video_items(results_dir: Path, model_or_lane: str) -> list[dict]:
    """Build one evaluation item per ``<scene_id>/r###.mp4`` under results_dir."""
    scene_dirs = sorted(entry for entry in results_dir.iterdir() if entry.is_dir())
    if not scene_dirs:
        sys.exit(
            f"No scene directories under {results_dir}. "
            "Expected the layout <results>/<scene_id>/r###.mp4."
        )
    items: list[dict] = []
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        for video in sorted(scene_dir.iterdir()):
            stem_match = VIDEO_RE.fullmatch(video.stem)
            if not stem_match or video.suffix.lower() not in VIDEO_SUFFIXES or not video.is_file():
                continue
            repeat_index = int(stem_match.group(1))
            items.append(
                {
                    "sample_id": f"{model_or_lane}::{scene_id}::{video.stem}",
                    "scene_id": scene_id,
                    "repeat_index": repeat_index,
                    "video_path": str(video.resolve()),
                }
            )
        if not any(item["scene_id"] == scene_id for item in items):
            print(f"warning: no r### videos under {scene_dir}", file=sys.stderr)
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a directory of generated videos against the PAWBench package."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=os.environ.get("PAWBENCH_DATA_DIR"),
        help="Local benchmark package directory (default: $PAWBENCH_DATA_DIR)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=os.environ.get("PAWBENCH_RESULTS_DIR"),
        help="Directory of one model's rollouts as <scene_id>/r###.mp4 (default: $PAWBENCH_RESULTS_DIR)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("PAWBENCH_MODEL", "my-model"),
        help="Model or lane name recorded on every row (default: $PAWBENCH_MODEL or 'my-model')",
    )
    parser.add_argument(
        "--vlm-base-url",
        default=os.environ.get("PAWBENCH_VLM_BASE_URL"),
        help="OpenAI-compatible VLM base URL (default: $PAWBENCH_VLM_BASE_URL)",
    )
    parser.add_argument(
        "--vlm-model",
        default=os.environ.get("PAWBENCH_VLM_MODEL"),
        help="VLM model name used for PAWEval judgment (default: $PAWBENCH_VLM_MODEL)",
    )
    parser.add_argument(
        "--vlm-api-key-env",
        default=os.environ.get("PAWBENCH_VLM_API_KEY_ENV", "YOUR_VLM_API_KEY"),
        help="Environment variable holding the VLM API key (default: $PAWBENCH_VLM_API_KEY_ENV or 'YOUR_VLM_API_KEY')",
    )
    args = parser.parse_args()
    if args.benchmark is None:
        parser.error("--benchmark (or $PAWBENCH_DATA_DIR) is required; download the dataset first, see README")
    if args.results is None:
        parser.error("--results (or $PAWBENCH_RESULTS_DIR) is required")
    if not args.benchmark.is_dir() or not (args.benchmark / "manifest.json").is_file():
        sys.exit(
            f"{args.benchmark} is not a materialized benchmark package "
            "(expected manifest.json + scenes.jsonl). Download it with the "
            "command in the README's 'Benchmark data' section."
        )
    if not args.results.is_dir():
        sys.exit(f"{args.results} does not exist; point --results at one model's rollout directory")
    if not args.vlm_base_url or not args.vlm_model:
        parser.error("--vlm-base-url and --vlm-model (or their $PAWBENCH_VLM_* defaults) are required")
    return args


def main() -> None:
    args = parse_args()
    videos = collect_video_items(args.results, args.model)
    print(f"Collected {len(videos)} video items from {args.results}")

    result = evaluate(
        args.benchmark,
        videos,
        model_or_lane=args.model,
        vlm={
            "base_url": args.vlm_base_url,
            "model": args.vlm_model,
            "api_key_env": args.vlm_api_key_env,
        },
    )

    print(f"status: {result['status']}")
    if result["blockers"]:
        for blocker in result["blockers"]:
            print(f"blocker: {blocker}")
    print(f"metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
