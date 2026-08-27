"""Generate PAWBench image-to-video rollouts through OpenRouter.

OpenRouter's video API is asynchronous and requires the source image to be
available at a public HTTPS URL. This example maps each benchmark-relative
``source_image_path`` onto ``--source-image-base-url`` and writes videos in the
layout consumed directly by ``evaluate.py``::

    <output>/<scene_id>/r000.mp4 ... r049.mp4

The command is preview-only by default. Pass ``--execute`` to submit paid video
generation jobs.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

API_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_SOURCE_IMAGE_BASE_URL = "https://huggingface.co/datasets/Andrew613/PAWBench/resolve/main/"
FAILURE_STATUSES = {"failed", "cancelled", "expired"}
CHECKPOINT_SCHEMA = "pawbench.openrouter_job/v1"


@dataclass(frozen=True)
class Scene:
    scene_id: str
    source_image_path: str
    prompt: str


@dataclass(frozen=True)
class GenerationTask:
    scene: Scene
    repeat_index: int
    output_path: Path


@dataclass(frozen=True)
class RequestOptions:
    model_id: str
    source_image_base_url: str
    duration: int | None
    resolution: str | None
    aspect_ratio: str | None
    seed: int | None


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
        default=os.environ.get("OPENROUTER_VIDEO_MODEL"),
        help="OpenRouter video model ID (default: $OPENROUTER_VIDEO_MODEL)",
    )
    parser.add_argument(
        "--source-image-base-url",
        default=os.environ.get("PAWBENCH_SOURCE_IMAGE_BASE_URL", DEFAULT_SOURCE_IMAGE_BASE_URL),
        help="Public HTTPS base URL containing the benchmark's relative source image paths",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the OpenRouter API key",
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
        "--seed",
        type=int,
        help="Optional seed used for r000; repeat index is added",
    )
    parser.add_argument("--duration", type=int)
    parser.add_argument("--resolution")
    parser.add_argument("--aspect-ratio")
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--max-polls", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit paid OpenRouter jobs; without this flag the command only previews requests",
    )
    args = parser.parse_args(argv)
    if args.benchmark is None:
        parser.error("--benchmark (or $PAWBENCH_DATA_DIR) is required")
    if args.output is None:
        parser.error("--output (or $PAWBENCH_RESULTS_DIR) is required")
    if not args.model_id:
        parser.error("--model-id (or $OPENROUTER_VIDEO_MODEL) is required")
    if args.num_rollouts <= 0:
        parser.error("--num-rollouts must be positive")
    if args.num_rollouts > 999:
        parser.error("--num-rollouts must not exceed 999")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.poll_interval < 0:
        parser.error("--poll-interval must not be negative")
    if args.max_polls <= 0:
        parser.error("--max-polls must be positive")
    return args


def _package_file(root: Path, value: object, *, field: str) -> tuple[Path, str]:
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
    return resolved, relative.as_posix()


def _scene_prompt(root: Path, row: Mapping[str, object]) -> str:
    for key in ("prompt", "generation_prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("prompt_path", "generation_prompt_path"):
        value = row.get(key)
        if value:
            prompt_path, _ = _package_file(root, value, field=key)
            prompt = prompt_path.read_text(encoding="utf-8").strip()
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
        scene_table, _ = _package_file(root, manifest.get("scene_table"), field="scene_table")
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
        if (
            not isinstance(scene_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", scene_id) is None
            or scene_id in by_id
        ):
            raise ValueError("benchmark package has invalid or duplicate scene IDs")
        _, source_image_path = _package_file(
            root, row.get("source_image_path"), field="source_image_path"
        )
        by_id[scene_id] = Scene(
            scene_id=scene_id,
            source_image_path=source_image_path,
            prompt=_scene_prompt(root, row),
        )

    requested = list(dict.fromkeys(selected_ids))
    missing = [scene_id for scene_id in requested if scene_id not in by_id]
    if missing:
        raise ValueError(f"unknown PAWBench scene(s): {', '.join(missing)}")
    return [by_id[scene_id] for scene_id in requested] if requested else list(by_id.values())


def build_tasks(
    scenes: Sequence[Scene], output_dir: Path, *, num_rollouts: int
) -> list[GenerationTask]:
    return [
        GenerationTask(
            scene=scene,
            repeat_index=repeat_index,
            output_path=output_dir / scene.scene_id / f"r{repeat_index:03d}.mp4",
        )
        for scene in scenes
        for repeat_index in range(num_rollouts)
    ]


def _public_image_url(base_url: str, relative_path: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("--source-image-base-url must be a public HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("--source-image-base-url must not include credentials, query, or fragment")
    path = f"{parsed.path.rstrip('/')}/{urllib.parse.quote(relative_path, safe='/')}"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def build_request(task: GenerationTask, options: RequestOptions) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": options.model_id,
        "prompt": task.scene.prompt,
        "generate_audio": False,
        "frame_images": [
            {
                "type": "image_url",
                "image_url": {
                    "url": _public_image_url(
                        options.source_image_base_url, task.scene.source_image_path
                    )
                },
                "frame_type": "first_frame",
            }
        ],
    }
    if options.duration is not None:
        payload["duration"] = options.duration
    if options.resolution is not None:
        payload["resolution"] = options.resolution
    if options.aspect_ratio is not None:
        payload["aspect_ratio"] = options.aspect_ratio
    if options.seed is not None:
        payload["seed"] = options.seed + task.repeat_index
    return payload


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep bearer credentials on the single OpenRouter origin."""

    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


class OpenRouterVideoClient:
    def __init__(self, api_key: str, *, timeout: float = 60.0) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirectHandler())

    def _request(
        self, method: str, url: str, payload: Mapping[str, object] | None = None
    ) -> urllib.response.addinfourl:
        api = urllib.parse.urlsplit(API_BASE_URL)
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != api.scheme
            or parsed.netloc != api.netloc
            or parsed.username
            or parsed.password
        ):
            raise RuntimeError("refusing to send the OpenRouter API key to an untrusted URL")
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            return self._opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

    def _json(
        self, method: str, url: str, payload: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        with self._request(method, url, payload) as response:
            try:
                value = json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("OpenRouter returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError("OpenRouter returned a non-object response")
        return value

    def list_models(self) -> list[dict[str, object]]:
        response = self._json("GET", f"{API_BASE_URL}/videos/models")
        models = response.get("data")
        if not isinstance(models, list) or not all(isinstance(model, dict) for model in models):
            raise RuntimeError("OpenRouter returned an invalid video model list")
        return models

    def submit(self, payload: dict[str, object]) -> dict[str, object]:
        return self._json("POST", f"{API_BASE_URL}/videos", payload)

    def poll(
        self,
        job: dict[str, object],
        *,
        poll_interval: float,
        max_polls: int,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, object]:
        current = dict(job)
        for _ in range(max_polls):
            status = current.get("status")
            if status == "completed":
                return current
            if status in FAILURE_STATUSES:
                raise RuntimeError(f"OpenRouter video job {status}: {current.get('error', '')}")
            polling_url = current.get("polling_url")
            if not isinstance(polling_url, str) or not polling_url:
                raise RuntimeError("OpenRouter video job has no polling_url")
            if poll_interval:
                sleep(poll_interval)
            api = urllib.parse.urlsplit(API_BASE_URL)
            api_origin = urllib.parse.urlunsplit((api.scheme, api.netloc, "/", "", ""))
            url = urllib.parse.urljoin(api_origin, polling_url)
            parsed = urllib.parse.urlsplit(url)
            if (
                parsed.scheme != api.scheme
                or parsed.netloc != api.netloc
                or parsed.username
                or parsed.password
            ):
                raise RuntimeError("OpenRouter returned an untrusted polling_url")
            current = self._json("GET", url)
        if current.get("status") == "completed":
            return current
        raise TimeoutError(f"OpenRouter video job did not complete after {max_polls} polls")

    def download(self, job: dict[str, object], output_path: Path) -> None:
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("OpenRouter video job has no id")
        encoded_job_id = urllib.parse.quote(job_id, safe="")
        url = f"{API_BASE_URL}/videos/{encoded_job_id}/content?index=0"
        with self._request("GET", url) as response:
            with output_path.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)


def _validate_model(
    models: Sequence[Mapping[str, object]],
    *,
    model_id: str,
    duration: int | None,
    resolution: str | None,
    aspect_ratio: str | None,
) -> None:
    model = next((candidate for candidate in models if candidate.get("id") == model_id), None)
    if model is None:
        raise ValueError(f"OpenRouter video model not found: {model_id}")
    if "first_frame" not in (model.get("supported_frame_images") or []):
        raise ValueError(
            f"OpenRouter model does not support first-frame image-to-video: {model_id}"
        )
    for name, value, field in (
        ("duration", duration, "supported_durations"),
        ("resolution", resolution, "supported_resolutions"),
        ("aspect ratio", aspect_ratio, "supported_aspect_ratios"),
    ):
        supported = model.get(field)
        if value is None:
            continue
        if not isinstance(supported, list):
            raise ValueError(f"OpenRouter cannot verify {name} support for {model_id}")
        if str(value) not in {str(item) for item in supported}:
            raise ValueError(f"OpenRouter model {model_id} does not support {name} {value}")


def checkpoint_path(task: GenerationTask) -> Path:
    output_root = task.output_path.parents[1]
    return output_root / ".openrouter-jobs" / task.scene.scene_id / f"r{task.repeat_index:03d}.json"


def _request_fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_checkpoint(
    path: Path, payload: Mapping[str, object], job: Mapping[str, object]
) -> None:
    record = {
        "schema_version": CHECKPOINT_SCHEMA,
        "request_sha256": _request_fingerprint(payload),
        "request": payload,
        "job": job,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_checkpoint(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid OpenRouter checkpoint: {path}") from exc
    if not isinstance(record, dict) or record.get("schema_version") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"invalid OpenRouter checkpoint: {path}")
    if record.get("request_sha256") != _request_fingerprint(payload):
        raise RuntimeError(
            f"OpenRouter checkpoint uses different generation options; "
            f"use --overwrite to resubmit: {path}"
        )
    job = record.get("job")
    if not isinstance(job, dict):
        raise RuntimeError(f"invalid OpenRouter checkpoint: {path}")
    return job


class _RunLock:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / ".openrouter-jobs" / "run.lock"
        self._file: TextIO | None = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        try:
            fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._file.close()
            message = f"another OpenRouter generator is using this output: {self.path}"
            raise RuntimeError(message) from exc
        self._file.write(f"pid={os.getpid()}\n")
        self._file.flush()

    def __exit__(self, *_: object) -> None:
        assert self._file is not None
        fcntl.flock(self._file, fcntl.LOCK_UN)
        self._file.close()


def generate(
    tasks: Sequence[GenerationTask],
    *,
    client: OpenRouterVideoClient,
    request_options: RequestOptions,
    poll_interval: float,
    max_polls: int,
    overwrite: bool,
    before_submit: Callable[[], None] | None = None,
) -> int:
    written = 0
    submission_validated = False
    for task in tasks:
        payload = build_request(task, request_options)
        checkpoint = checkpoint_path(task)
        if task.output_path.exists() and not overwrite:
            if not checkpoint.is_file():
                raise RuntimeError(
                    f"existing OpenRouter output has no request checkpoint: {task.output_path}"
                )
            _read_checkpoint(checkpoint, payload)
            print(f"skip existing: {task.output_path}")
            continue
        if checkpoint.is_file() and not overwrite:
            job = _read_checkpoint(checkpoint, payload)
            if job.get("status") in FAILURE_STATUSES:
                raise RuntimeError(
                    f"OpenRouter checkpoint is {job.get('status')}; use --overwrite to resubmit: "
                    f"{checkpoint}"
                )
            print(f"resume job {job.get('id')}: {task.output_path}")
        else:
            if not submission_validated and before_submit is not None:
                before_submit()
                submission_validated = True
            job = client.submit(payload)
            if not isinstance(job.get("id"), str) or not job.get("id"):
                raise RuntimeError("OpenRouter submit response has no job id")
            print(f"submitted job {job['id']}: {task.output_path}", file=sys.stderr, flush=True)
            try:
                _write_checkpoint(checkpoint, payload, job)
            except OSError as exc:
                raise RuntimeError(
                    f"OpenRouter accepted job {job['id']} but its checkpoint could not be saved; "
                    "reconcile that job before rerunning"
                ) from exc

        completed = client.poll(job, poll_interval=poll_interval, max_polls=max_polls)
        _write_checkpoint(checkpoint, payload, completed)
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = task.output_path.with_name(f"{task.output_path.name}.{os.getpid()}.part")
        temporary.unlink(missing_ok=True)
        try:
            client.download(completed, temporary)
            temporary.replace(task.output_path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"wrote {task.output_path} (job={completed.get('id')})")
        written += 1
    return written


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        scenes = load_scenes(args.benchmark, args.scene if not args.all_scenes else ())
        tasks = build_tasks(scenes, args.output, num_rollouts=args.num_rollouts)
        request_options = RequestOptions(
            model_id=args.model_id,
            source_image_base_url=args.source_image_base_url,
            duration=args.duration,
            resolution=args.resolution,
            aspect_ratio=args.aspect_ratio,
            seed=args.seed,
        )
        first_request = build_request(tasks[0], request_options)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"model: {args.model_id}")
    print(f"planned rollouts: {len(tasks)} across {len(scenes)} scene(s)")
    print("first request:")
    print(json.dumps(first_request, indent=2))
    if not args.execute:
        print("preview only: add --execute to submit paid OpenRouter video jobs")
        return 0

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before using --execute")
    client = OpenRouterVideoClient(api_key)
    try:
        def validate_before_submit() -> None:
            _validate_model(
                client.list_models(),
                model_id=args.model_id,
                duration=args.duration,
                resolution=args.resolution,
                aspect_ratio=args.aspect_ratio,
            )

        with _RunLock(args.output):
            written = generate(
                tasks,
                client=client,
                request_options=request_options,
                poll_interval=args.poll_interval,
                max_polls=args.max_polls,
                overwrite=args.overwrite,
                before_submit=validate_before_submit,
            )
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"generated: {written}; output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
