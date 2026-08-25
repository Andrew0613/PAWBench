"""Small release-surface checks that protect user-visible PAWBench assets."""

from __future__ import annotations

import re
from pathlib import Path

from pawbench.paweval.rubrics.loader import RUBRIC_ROOT, load_rubric

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCENE_IDS = frozenset(
    """
    A-01 A-02 A-03 A-04 A-05 A-06 A-07 A-08 A-09 A-10 A-11 A-12 A-13 A-14 A-15
    A16 A17 A18 A19 A20 A21 A22 A23 A24 A25
    BA-01 BA-01-02 BA-01-S1 BA-02 BA-03 BA-06 BA-07 BA-08
    BC-01 BC-01-I2 BC-02 BC-03 BC-03-I1 BC-04 BC-05 BC-06 BC-09
    BM-01 BM-02 BM-03 BM-04 BS-01 BS-01-I1 BS-02 BS-03
    """.split()
)


def test_readme_local_images_exist_without_internal_example_ids() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for internal_id in ("A-01", "BC-01", "a01", "bc01"):
        assert internal_id not in readme
    assert '<img src="assets/' not in readme
    local_images = re.findall(r"!\[[^\]]+\]\((assets/[^)]+)\)", readme)
    assert local_images
    assert all((ROOT / source).is_file() for source in local_images)


def test_released_rubric_inventory_is_exact_paired_and_loadable() -> None:
    for axis in ("outcome", "trustworthiness"):
        actual = {path.stem for path in (RUBRIC_ROOT / axis).glob("*.yaml")}
        assert actual == EXPECTED_SCENE_IDS
        for scene_id in actual:
            load_rubric(axis, scene_id)
