from __future__ import annotations

import json

import pytest

from pawbench.metrics import compute_metrics


def policy(*, calibration_groups: list[str] | None = None) -> dict:
    calibration_groups = calibration_groups or ["calibration"] * 25

    def scenes(track: str, groups: list[str]) -> list[dict]:
        output = []
        for index in range(25):
            row = {
                "scene_id": f"scene-{index:02d}",
                "track": track,
                "group": groups[index],
                "expected_repeat_indices": list(range(50)),
            }
            if track == "calibration":
                row["reference_distribution"] = {"a": 0.5, "b": 0.5}
            else:
                row["support_labels"] = ["a", "b"]
            output.append(row)
        return output

    return {
        "model_or_lanes": ["model"],
        "tracks": {
            "calibration": {"scenes": scenes("calibration", calibration_groups)},
            "coverage": {"scenes": scenes("coverage", ["coverage"] * 25)},
        },
    }


def rows_for(scene_id: str, track: str, labels: list[str | None]) -> list[dict]:
    rows = []
    for repeat_index, label in enumerate(labels):
        row = {
            "sample_id": f"{scene_id}__r{repeat_index:03d}",
            "scene_id": scene_id,
            "track": track,
            "model_or_lane": "model",
            "repeat_index": repeat_index,
            "observation": "outcome" if label is not None else "null_observation",
        }
        if label is not None:
            row["outcome_label"] = label
        rows.append(row)
    return rows


def complete_rows() -> list[dict]:
    return [
        row
        for track in ("calibration", "coverage")
        for index in range(25)
        for row in rows_for(f"scene-{index:02d}", track, ["a", "b"] * 25)
    ]


def scene(result: dict, track: str, scene_id: str) -> dict:
    return next(
        row
        for row in result["tracks"][track]["models"]["model"]["scenes"]
        if row["scene_id"] == scene_id
    )


def replace_scene(
    rows: list[dict], track: str, scene_id: str, labels: list[str | None]
) -> list[dict]:
    return [
        row for row in rows if not (row["track"] == track and row["scene_id"] == scene_id)
    ] + rows_for(scene_id, track, labels)


def test_calibration_uses_valid_only_tvd_and_the_30_null_boundary() -> None:
    rows = replace_scene(complete_rows(), "calibration", "scene-00", ["a"] * 20 + [None] * 30)
    result = compute_metrics(rows, policy())

    target = scene(result, "calibration", "scene-00")
    assert target["status"] == "pass"
    assert target["valid_outcomes"] == 20
    assert target["metric"] == {"name": "calibration_tvd_percent", "value": 50.0}

    rows = replace_scene(rows, "calibration", "scene-00", ["a"] * 20 + ["b"] * 20 + [None] * 10)
    assert (
        scene(compute_metrics(rows, policy()), "calibration", "scene-00")["metric"]["value"] == 0.0
    )

    rows = replace_scene(rows, "calibration", "scene-00", ["a"] * 19 + [None] * 31)
    assert scene(compute_metrics(rows, policy()), "calibration", "scene-00")["metric"] is None


def test_coverage_and_macro_outputs_remain_separate_without_a_ranking() -> None:
    groups = ["g1", "g1", "g2"] + ["other"] * 22
    rows = complete_rows()
    rows = replace_scene(rows, "calibration", "scene-00", ["a"] * 50)
    rows = replace_scene(rows, "calibration", "scene-01", ["a", "b"] * 25)
    rows = replace_scene(rows, "calibration", "scene-02", ["a"] * 19 + [None] * 31)
    rows = replace_scene(rows, "coverage", "scene-00", ["a"] * 20 + [None] * 30)

    result = compute_metrics(rows, policy(calibration_groups=groups))
    calibration = result["tracks"]["calibration"]["models"]["model"]

    assert calibration["groups"]["g1"]["metric"] == {
        "name": "calibration_tvd_percent",
        "value": 25.0,
    }
    assert calibration["groups"]["g2"]["metric"] is None
    assert calibration["track_average"] == {
        "name": "calibration_tvd_percent",
        "value": pytest.approx(50 / 24),
    }
    assert calibration["scene_pass_rate"] == {
        "passing_scenes": 24,
        "scene_denominator": 25,
        "value": 96.0,
    }
    assert scene(result, "coverage", "scene-00")["metric"] == {
        "name": "coverage_percent",
        "value": 50.0,
    }
    assert "combined_ranking" not in result


def test_infrastructure_failure_blocks_instead_of_becoming_a_null() -> None:
    rows = replace_scene(complete_rows(), "calibration", "scene-00", ["a"] * 50)
    rows[0] = {
        **rows[0],
        "observation": "infrastructure_failure",
        "outcome_label": None,
        "failure_code": "provider_timeout",
    }
    result = compute_metrics(rows, policy())

    calibration = result["tracks"]["calibration"]["models"]["model"]
    assert result["status"] == "blocked"
    assert calibration["track_average"] is None
    assert calibration["scene_pass_rate"] is None
    assert scene(result, "calibration", "scene-00")["null_observations"] == 0


def test_rows_and_result_are_json_compatible_and_invalid_inputs_fail_closed() -> None:
    rows = complete_rows()
    result = compute_metrics(rows, policy())
    assert json.loads(json.dumps(result)) == result

    duplicate = list(rows)
    duplicate.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate slot"):
        compute_metrics(duplicate, policy())

    invalid_label = list(rows)
    invalid_label[0] = {**rows[0], "outcome_label": "not-in-ontology"}
    with pytest.raises(ValueError, match="outside scene ontology"):
        compute_metrics(invalid_label, policy())
