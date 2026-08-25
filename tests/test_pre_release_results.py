from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_readme_results_match_machine_readable_snapshot() -> None:
    payload = json.loads(
        (ROOT / "results" / "pre_release_oracle_pe.json").read_text(encoding="utf-8")
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert payload["status"] == "pre_release_aggregate_only"
    assert payload["protocol"]["rollouts_per_scene"] == 50
    assert payload["evidence_boundary"]["raw_judgment_rows_in_repository"] is False
    for result in payload["results"]:
        row = (
            f"| {result['model']} | {result['calibration_tvd_percent']:.1f} | "
            f"{result['coverage_percent']:.1f} |"
        )
        assert row in readme
