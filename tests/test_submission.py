"""Interface-level tests for pawbench.submission.

The grid is the contract: every counter-example below attacks either the
slot coverage, the closed failure vocabulary, or the benchmark binding.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pawbench import benchmark as benchmark_module
from pawbench import submission
from pawbench.errors import SubmissionError

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def benchmark() -> benchmark_module.Benchmark:
    return benchmark_module.load(EXAMPLES / "benchmark")


def _load_envelope() -> dict:
    return json.loads((EXAMPLES / "submission" / "submission.json").read_text())


def _write_envelope(tmp_path: Path, envelope: dict) -> Path:
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(envelope, indent=2))
    return path


def _problems_of(exc: SubmissionError) -> list[str]:
    return [f"{p.where}: {p.detail}" for p in exc.problems]


def _flat_videos(tmp_path: Path, repeats: int = 2) -> Path:
    """A flat directory with one produced video per slot except SYN-P01 r1."""
    videos = tmp_path / "videos"
    videos.mkdir()
    source = EXAMPLES / "submission" / "videos"
    shutil.copy(source / "SYN-C01__r000.mp4", videos / "SYN-C01__r000.mp4")
    shutil.copy(source / "SYN-C01__r001.mp4", videos / "SYN-C01__r001.mp4")
    shutil.copy(source / "SYN-P01__r000.mp4", videos / "SYN-P01__r000.mp4")
    return videos


# --- load: happy path -----------------------------------------------------


def test_loads_example_submission(benchmark) -> None:
    sub = submission.load(EXAMPLES / "submission" / "submission.json", benchmark=benchmark)

    assert sub.submission_id == "example-system-smoke"
    assert sub.system == "example-video-model"
    assert sub.prompt_set == "gt_guided"
    assert sub.scene_ids == ("SYN-C01", "SYN-P01")
    assert sub.repeats_per_scene == 2
    assert sub.slot_count == 4 == len(sub.items)
    statuses = [item.status for item in sub.items]
    assert statuses.count("produced") == 3
    assert statuses.count("model_failure") == 1
    failed = next(item for item in sub.items if item.status == "model_failure")
    assert failed.key == ("SYN-P01", 1)
    assert failed.failure_code == "generation_timeout"


def test_scope_scene_ids_resolve_in_benchmark_order(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["scope"]["scene_ids"] = ["SYN-P01", "SYN-C01"]  # deliberately reversed
    sub = submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert sub.scene_ids == ("SYN-C01", "SYN-P01")


# --- load: envelope invariants --------------------------------------------


def test_rejects_wrong_schema_version(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["schema_version"] = "pawbench.submission/v0"
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert any("schema_version" in p for p in _problems_of(excinfo.value))


def test_rejects_benchmark_binding_mismatch(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["benchmark"]["benchmark_revision"] = 99
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert any("benchmark_revision" in p for p in _problems_of(excinfo.value))


def test_rejects_unknown_prompt_set(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["prompt_set"] = "self_guided"
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert any("prompt_set" in p for p in _problems_of(excinfo.value))


def test_rejects_unknown_scene_in_scope(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["scope"]["scene_ids"] = ["GHOST-01"]
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert any("unknown scene" in p for p in _problems_of(excinfo.value))


def test_rejects_scene_outside_selected_splits(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["scope"]["splits"] = ["calibration"]  # SYN-P01 is coverage
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert any("belong to split" in p for p in _problems_of(excinfo.value))


# --- load: items and the grid ---------------------------------------------


def test_rejects_missing_slot_instead_of_shrinking_denominator(
    benchmark, tmp_path
) -> None:
    envelope = _load_envelope()
    envelope["items"] = [item for item in envelope["items"] if item["repeat_index"] != 1]
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    problems = _problems_of(excinfo.value)
    assert any("missing slot" in p for p in problems)
    assert any("failure items" in p for p in problems)


def test_rejects_unexpected_slot(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["repeats_per_scene"] = 1  # r1 items become unexpected
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert any("unexpected slot" in p for p in _problems_of(excinfo.value))


def test_rejects_duplicate_slot(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["items"].append(dict(envelope["items"][0]))
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert any("duplicate slot" in p for p in _problems_of(excinfo.value))


def test_rejects_produced_item_without_video_path(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["items"][0].pop("video_path")
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert any("video_path" in p for p in _problems_of(excinfo.value))


def test_rejects_unsafe_video_path(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["items"][0]["video_path"] = "../outside.mp4"
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert any("video_path" in p for p in _problems_of(excinfo.value))


def test_rejects_unknown_failure_code(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["items"][3]["failure_code"] = "crashed_mysteriously"
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert any("failure_code" in p for p in _problems_of(excinfo.value))


def test_rejects_mismatched_failure_code_family(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["items"][3]["failure_code"] = "media_missing"  # infra code on model_failure
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert any("failure_code" in p for p in _problems_of(excinfo.value))


def test_rejects_unknown_status(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["items"][0]["status"] = "partially_produced"
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert any("status" in p for p in _problems_of(excinfo.value))


def test_rejects_unknown_envelope_field(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["secret_note"] = "hi"
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    assert any("secret_note" in p for p in _problems_of(excinfo.value))


def test_rejects_duplicate_json_key(benchmark, tmp_path) -> None:
    path = tmp_path / "submission.json"
    path.write_text('{"schema_version": "a", "schema_version": "b"}')
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(path, benchmark=benchmark)
    assert any("duplicate JSON key" in p for p in _problems_of(excinfo.value))


def test_collects_problems_across_sections(benchmark, tmp_path) -> None:
    envelope = _load_envelope()
    envelope["system"] = ""  # envelope defect
    envelope["items"][0]["repeat_index"] = 9  # grid defect
    with pytest.raises(SubmissionError) as excinfo:
        submission.load(_write_envelope(tmp_path, envelope), benchmark=benchmark)
    problems = _problems_of(excinfo.value)
    assert any("system" in p for p in problems)
    assert any(p.startswith(f"{tmp_path / 'submission.json'}") for p in problems)


# --- build ----------------------------------------------------------------


def test_build_and_write_and_load_round_trip(benchmark, tmp_path) -> None:
    videos = _flat_videos(tmp_path)
    sub = submission.build(
        benchmark,
        videos,
        system="Example Model v2!",
        prompt_set="gt_guided",
        repeats_per_scene=2,
        missing_policy="model_failure",
    )
    assert sub.submission_id == "Example-Model-v2"  # slug default
    assert sub.slot_count == 4
    assert sorted(item.status for item in sub.items) == [
        "model_failure",
        "produced",
        "produced",
        "produced",
    ]
    missing = next(i for i in sub.items if i.status == "model_failure")
    assert missing.key == ("SYN-P01", 1)
    assert missing.failure_code == "generation_failed"

    manifest = videos / "submission.json"
    submission.write(sub, manifest)
    reloaded = submission.load(manifest, benchmark=benchmark)
    assert reloaded == sub

    # Deterministic serialization: same model, same bytes.
    submission.write(sub, videos / "again.json")
    assert (videos / "again.json").read_bytes() == manifest.read_bytes()


def test_build_missing_policy_error_lists_all_missing_slots(benchmark, tmp_path) -> None:
    videos = _flat_videos(tmp_path)  # SYN-P01 r1 absent
    with pytest.raises(SubmissionError) as excinfo:
        submission.build(benchmark, videos, system="m", repeats_per_scene=2)
    problems = _problems_of(excinfo.value)
    assert any("missing video for slot SYN-P01 r1" in p for p in problems)
    assert any("missing_policy" in p for p in problems)


def test_build_defaults_repeats_to_formal(benchmark, tmp_path) -> None:
    videos = _flat_videos(tmp_path)
    sub = submission.build(
        benchmark, videos, system="m", missing_policy="model_failure"
    )  # no repeats given -> benchmark.formal_repeats = 50
    assert sub.repeats_per_scene == 50
    assert sub.slot_count == 2 * 50
    assert sum(1 for i in sub.items if i.status == "produced") == 3


def test_build_flags_stale_convention_files(benchmark, tmp_path) -> None:
    videos = _flat_videos(tmp_path)
    shutil.copy(
        EXAMPLES / "submission" / "videos" / "SYN-C01__r000.mp4",
        videos / "TYPO-99__r000.mp4",
    )
    with pytest.raises(SubmissionError) as excinfo:
        submission.build(
            benchmark, videos, system="m", repeats_per_scene=2, missing_policy="model_failure"
        )
    assert any(
        "TYPO-99__r000.mp4" in p and "not in the benchmark" in p
        for p in _problems_of(excinfo.value)
    )


def test_build_rejects_bad_naming_pattern(benchmark, tmp_path) -> None:
    videos = _flat_videos(tmp_path)
    with pytest.raises(SubmissionError) as excinfo:
        submission.build(
            benchmark, videos, system="m", repeats_per_scene=2, naming_pattern="{scene_id}"
        )
    assert any("naming_pattern" in p for p in _problems_of(excinfo.value))


def test_build_scopes_to_selected_split(benchmark, tmp_path) -> None:
    videos = _flat_videos(tmp_path)
    sub = submission.build(
        benchmark,
        videos,
        system="m",
        splits=("calibration",),
        repeats_per_scene=2,
    )
    assert sub.scene_ids == ("SYN-C01",)
    assert sub.slot_count == 2
    assert all(item.status == "produced" for item in sub.items)


def test_build_rejects_missing_videos_dir(benchmark, tmp_path) -> None:
    with pytest.raises(SubmissionError) as excinfo:
        submission.build(benchmark, tmp_path / "nope", system="m", repeats_per_scene=2)
    assert any("not a directory" in p for p in _problems_of(excinfo.value))
