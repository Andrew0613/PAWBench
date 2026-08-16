"""PAWBench quickstart.

Current status (wave 3): load the synthetic benchmark package and a
submission, then show the fixed denominator. Waves ahead: evaluate +
results (wave 6).

Run from the repository root:

    python examples/quickstart.py
"""

import sys
from pathlib import Path

# Allow running from a fresh checkout before `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pawbench.benchmark
import pawbench.submission

EXAMPLES = Path(__file__).resolve().parent


def main() -> None:
    benchmark = pawbench.benchmark.load(EXAMPLES / "benchmark")
    print(
        f"{benchmark.dataset_id} rev {benchmark.benchmark_revision} — "
        f"{len(benchmark.scenes)} scenes, prompt sets: {sorted(benchmark.prompt_sets)}, "
        f"formal repeats: {benchmark.formal_repeats}"
    )
    for split in ("calibration", "coverage"):
        for scene in benchmark.scenes_in_split(split):
            print(f"  [{scene.split}] {scene.scene_id} #{scene.split_order}: {scene.action}")

    submission = pawbench.submission.load(
        EXAMPLES / "submission" / "submission.json", benchmark=benchmark
    )
    produced = sum(1 for item in submission.items if item.status == "produced")
    failed = {
        status: sum(1 for item in submission.items if item.status == status)
        for status in ("model_failure", "infrastructure_failure")
    }
    print(
        f"\nsubmission {submission.submission_id!r} ({submission.system}): "
        f"{submission.slot_count} slots = "
        f"{len(submission.scene_ids)} scenes x {submission.repeats_per_scene} repeats"
    )
    print(
        f"  produced: {produced}, model_failure: {failed['model_failure']}, "
        f"infrastructure_failure: {failed['infrastructure_failure']}"
    )


if __name__ == "__main__":
    main()
