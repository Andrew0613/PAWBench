"""Generate PAWBench rollouts with a Diffusers image-to-video model.

The output layout is consumed directly by ``evaluate.py``::

    <output>/<scene_id>/r000.mp4 ... r049.mp4

The official run covers the complete 50 x 50 grid. The model must be a
Diffusers image-to-video pipeline that accepts ``image``, ``prompt``, and
``generator`` and returns ``.frames``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


@dataclass(frozen=True)
class Scene:
    scene_id: str
    source_image: Path
    prompt: str


@dataclass(frozen=True)
class GenerationTask:
    scene: Scene
    repeat_index: int
    seed: int
    output_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=os.environ.get("PAWBENCH_DATA_DIR"),
        help="Downloaded PAWBench directory (default: $PAWBENCH_DATA_DIR)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=os.environ.get("PAWBENCH_RESULTS_DIR"),
        help="Rollout output directory (default: $PAWBENCH_RESULTS_DIR)",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("PAWBENCH_GENERATOR_MODEL", DEFAULT_MODEL_ID),
        help="Hugging Face ID or local path for a compatible Diffusers I2V model",
    )
    scene_selection = parser.add_mutually_exclusive_group(required=True)
    scene_selection.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Scene ID to generate; repeat the flag for multiple scenes",
    )
    scene_selection.add_argument(
        "--all-scenes",
        action="store_true",
        help="Generate all 50 scenes; combine with --num-rollouts 50 for the complete grid",
    )
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed used for r000; repeat index is added"
    )
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned outputs without loading the model",
    )
    args = parser.parse_args(argv)
    if args.benchmark is None:
        parser.error("--benchmark (or $PAWBENCH_DATA_DIR) is required")
    if args.output is None:
        parser.error("--output (or $PAWBENCH_RESULTS_DIR) is required")
    if args.num_rollouts <= 0:
        parser.error("--num-rollouts must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def _package_file(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"benchmark scene has no {field}")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"benchmark {field} must be relative: {value}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"benchmark {field} escapes the package: {value}")
    if not resolved.is_file():
        raise ValueError(f"benchmark {field} does not exist: {value}")
    return resolved


def _scene_prompt(root: Path, row: dict[str, Any]) -> str:
    for key in ("prompt", "generation_prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("prompt_path", "generation_prompt_path"):
        value = row.get(key)
        if value:
            prompt = _package_file(root, value, field=key).read_text(encoding="utf-8").strip()
            if prompt:
                return prompt
    action = row.get("action")
    if isinstance(action, str) and action.strip():
        return action.strip()
    raise ValueError(f"benchmark scene {row.get('scene_id', '<unknown>')} has no generation prompt")


def load_scenes(benchmark_dir: Path, selected_ids: Sequence[str] = ()) -> list[Scene]:
    root = benchmark_dir.resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "pawbench.benchmark_inputs/v1":
            raise ValueError("unsupported benchmark schema")
        scene_table = _package_file(root, manifest.get("scene_table"), field="scene_table")
        rows = [
            json.loads(line)
            for line in scene_table.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise ValueError("invalid local PAWBench package") from exc
    if len(rows) != 50 or not all(isinstance(row, dict) for row in rows):
        raise ValueError("benchmark package must contain the released 50-scene contract")

    by_id: dict[str, Scene] = {}
    for row in rows:
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id or scene_id in by_id:
            raise ValueError("benchmark package has invalid or duplicate scene IDs")
        by_id[scene_id] = Scene(
            scene_id=scene_id,
            source_image=_package_file(
                root, row.get("source_image_path"), field="source_image_path"
            ),
            prompt=_scene_prompt(root, row),
        )

    requested = list(dict.fromkeys(selected_ids))
    missing = [scene_id for scene_id in requested if scene_id not in by_id]
    if missing:
        raise ValueError(f"unknown PAWBench scene(s): {', '.join(missing)}")
    return [by_id[scene_id] for scene_id in requested] if requested else list(by_id.values())


def build_tasks(
    scenes: Sequence[Scene], output_dir: Path, *, num_rollouts: int, seed: int
) -> list[GenerationTask]:
    return [
        GenerationTask(
            scene=scene,
            repeat_index=repeat_index,
            seed=seed + repeat_index,
            output_path=output_dir / scene.scene_id / f"r{repeat_index:03d}.mp4",
        )
        for scene in scenes
        for repeat_index in range(num_rollouts)
    ]


def _diffusers_runtime(
    model_id: str, *, dtype_name: str, device: str, device_map: str
) -> tuple[Any, Any, Callable[..., Any], Callable[..., Any]]:
    try:
        import torch
        from diffusers import DiffusionPipeline
        from diffusers.utils import export_to_video, load_image
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Install generation dependencies first: "
            "pip install -r requirements-generate.txt"
        ) from exc

    dtype = getattr(torch, dtype_name)
    # ``device_map`` already places the pipeline. Diffusers rejects an explicit
    # ``pipe.to(...)`` while a device map is active.
    pipe = DiffusionPipeline.from_pretrained(model_id, dtype=dtype, device_map=device_map)
    return torch, pipe, load_image, export_to_video


def generate(
    tasks: Sequence[GenerationTask],
    *,
    model_id: str,
    dtype_name: str,
    device: str,
    device_map: str,
    fps: int,
    overwrite: bool,
    runtime: tuple[Any, Any, Callable[..., Any], Callable[..., Any]] | None = None,
) -> int:
    torch, pipe, load_image, export_to_video = runtime or _diffusers_runtime(
        model_id, dtype_name=dtype_name, device=device, device_map=device_map
    )
    written = 0
    for task in tasks:
        if task.output_path.exists() and not overwrite:
            print(f"skip existing: {task.output_path}")
            continue
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device=device).manual_seed(task.seed)
        result = pipe(
            image=load_image(str(task.scene.source_image)),
            prompt=task.scene.prompt,
            generator=generator,
        )
        frames = getattr(result, "frames", None)
        if frames is None or len(frames) == 0:
            raise RuntimeError(f"Diffusers pipeline returned no frames for {task.scene.scene_id}")
        export_to_video(frames[0], str(task.output_path), fps=fps)
        print(f"wrote {task.output_path} (seed={task.seed})")
        written += 1
    return written


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        scenes = load_scenes(args.benchmark, args.scene if not args.all_scenes else ())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    tasks = build_tasks(scenes, args.output, num_rollouts=args.num_rollouts, seed=args.seed)
    print(f"model: {args.model_id}")
    print(f"planned rollouts: {len(tasks)} across {len(scenes)} scene(s)")
    for task in tasks[:5]:
        print(
            f"  {task.scene.scene_id} r{task.repeat_index:03d} "
            f"seed={task.seed} -> {task.output_path}"
        )
    if len(tasks) > 5:
        print(f"  ... {len(tasks) - 5} more")
    if args.dry_run:
        return 0
    written = generate(
        tasks,
        model_id=args.model_id,
        dtype_name=args.dtype,
        device=args.device,
        device_map=args.device_map,
        fps=args.fps,
        overwrite=args.overwrite,
    )
    print(f"generated: {written}; output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
