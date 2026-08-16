"""PAWBench quickstart.

Current status (wave 2): load and inspect the synthetic benchmark package.
Waves ahead: submission load/build (wave 3), evaluate + results (wave 6).

Run from the repository root:

    python examples/quickstart.py
"""

import sys
from pathlib import Path

# Allow running from a fresh checkout before `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pawbench.benchmark

EXAMPLE_PACKAGE = Path(__file__).resolve().parent / "benchmark"


def main() -> None:
    benchmark = pawbench.benchmark.load(EXAMPLE_PACKAGE)
    print(
        f"{benchmark.dataset_id} rev {benchmark.benchmark_revision} — "
        f"{len(benchmark.scenes)} scenes, prompt sets: {sorted(benchmark.prompt_sets)}, "
        f"formal repeats: {benchmark.formal_repeats}"
    )
    for split in ("calibration", "coverage"):
        for scene in benchmark.scenes_in_split(split):
            reference = scene.reference_distribution()
            if reference is None:
                detail = "coverage: no published reference"
            else:
                detail = "reference: " + ", ".join(
                    f"{label}={count}" for label, count in reference.items()
                )
            print(f"  [{scene.split}] {scene.scene_id} #{scene.split_order}: {scene.action}")
            print(f"      outcomes: {', '.join(scene.outcome_labels)} — {detail}")


if __name__ == "__main__":
    main()
