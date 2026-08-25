from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pawbench.paweval.adapter import build_request_payload
from pawbench.paweval.evidence.frames import EvidenceFrame
from pawbench.paweval.evidence.package import EvidencePackage
from pawbench.paweval.evidence.source_image import SourceImageIdentity
from pawbench.paweval.judge.client import StaticJudgeClient, _completion_url
from pawbench.paweval.judge.requests import JudgeRequest, RowIdentity
from pawbench.paweval.judge.responses import JudgeResponse, parse_judge_response
from pawbench.paweval.judgment import JudgmentPreflightError, JudgmentSample, judge


def _write_media(tmp_path: Path, name: str, contents: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(contents)
    return path.as_uri()


def _package(tmp_path: Path) -> EvidencePackage:
    source_uri = _write_media(tmp_path, "source.png", b"source-image")
    return EvidencePackage(
        sample_id="s1",
        scene_id="A-03",
        frames=(
            EvidenceFrame(
                "s1-initial", "initial", _write_media(tmp_path, "initial.jpg", b"initial"), 0.0
            ),
            EvidenceFrame(
                "s1-action", "action", _write_media(tmp_path, "action.jpg", b"action"), 0.5
            ),
            EvidenceFrame(
                "s1-terminal", "terminal", _write_media(tmp_path, "terminal.jpg", b"terminal"), 1.0
            ),
        ),
        source_image=SourceImageIdentity(attached=True, uri=source_uri),
    )


def _outcome(*, readable: bool = True) -> str:
    return json.dumps(
        {
            "scene_id": "A-03",
            "outcome_readable": readable,
            "outcome_in_schema": "IN_SCHEMA" if readable else "NOT_READABLE",
            "outcome_label": "red" if readable else None,
            "observed_result": "red" if readable else "not visible",
            "failure_axes": [] if readable else ["INPUT_QUALITY"],
        }
    )


def _trust() -> str:
    return json.dumps(
        {
            "scene_id": "A-03",
            "status": "TRUSTED",
            "scene_grounding": {"status": "PASS"},
            "action_execution": {
                "status": "PASS",
                "target_hit": "YES",
                "object_acquired": "YES",
                "action_spec_followed": "YES",
            },
            "object_continuity": {"status": "PASS"},
            "physical_process": {"status": "PASS", "failure_phase": "NONE"},
            "failure_axes": [],
        }
    )


class ScriptedAdapter:
    def __init__(self) -> None:
        self.calls: list[JudgeRequest] = []

    def complete(self, request: JudgeRequest) -> JudgeResponse:
        self.calls.append(request)
        if request.row_identity.repeat_index == 2 and request.axis == "trustworthiness_audit":
            return parse_judge_response("not json", axis=request.axis, scene_id=request.scene_id)
        raw = (
            _trust()
            if request.axis == "trustworthiness_audit"
            else _outcome(readable=request.row_identity.repeat_index != 1)
        )
        return parse_judge_response(raw, axis=request.axis, scene_id=request.scene_id)


def _sample(package: EvidencePackage, repeat_index: int) -> JudgmentSample:
    return JudgmentSample(
        package=package,
        track="calibration",
        row_identity=RowIdentity("s1", "A-03", "model-x", repeat_index),
    )


def test_judge_keeps_each_repeat_and_classifies_outcome_null_and_infrastructure(
    tmp_path: Path,
) -> None:
    adapter = ScriptedAdapter()
    package = _package(tmp_path)
    batch = judge(
        samples=[_sample(package, repeat) for repeat in range(3)],
        adapter=adapter,
    )

    assert [item.row_identity.repeat_index for item in batch.judgments] == [0, 1, 2]
    assert [item.status for item in batch.judgments] == [
        "outcome",
        "null_observation",
        "infrastructure_failure",
    ]
    assert batch.judgments[0].outcome_label == "red"
    assert batch.judgments[1].outcome_label is None
    assert batch.judgments[2].failure_code == "malformed_trustworthiness_audit"
    assert [row["repeat_index"] for row in batch.normalized_rows()] == [0, 1, 2]
    assert "PAWEval Outcome Readout" in adapter.calls[0].prompt
    assert "Rubric:" in adapter.calls[0].prompt


def test_judge_rejects_invalid_track_before_calling_adapter(tmp_path: Path) -> None:
    calls = []
    adapter = StaticJudgeClient({}, calls=calls)
    invalid = replace(_sample(_package(tmp_path), 0), track="not-a-track")

    with pytest.raises(JudgmentPreflightError, match="invalid_track"):
        judge(samples=[invalid], adapter=adapter)

    assert calls == []


def test_judge_keeps_unavailable_evidence_as_one_infrastructure_row(tmp_path: Path) -> None:
    calls = []
    adapter = StaticJudgeClient({}, calls=calls)
    invalid = replace(_package(tmp_path), frames=())
    batch = judge(
        samples=[JudgmentSample(invalid, "calibration", RowIdentity("s1", "A-03", "model-x", 0))],
        adapter=adapter,
    )

    assert [(item.status, item.failure_code) for item in batch.judgments] == [
        ("infrastructure_failure", "evidence_unavailable")
    ]
    assert calls == []


def test_adapter_builds_multimodal_messages_from_local_media(tmp_path: Path) -> None:
    payload = build_request_payload(prompt="judge this", package=_package(tmp_path))

    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    content = messages[1]["content"]
    assert content[0] == {"type": "text", "text": "judge this"}
    assert [part["type"] for part in content].count("image_url") == 4
    assert all(
        part["image_url"]["url"].startswith("data:image/")
        for part in content
        if part["type"] == "image_url"
    )


def test_judge_keeps_media_transport_failure_as_one_infrastructure_row(tmp_path: Path) -> None:
    package = replace(
        _package(tmp_path),
        source_image=SourceImageIdentity(attached=True, uri=(tmp_path / "missing.png").as_uri()),
    )
    calls = []
    adapter = StaticJudgeClient({}, calls=calls)

    batch = judge(
        samples=[JudgmentSample(package, "calibration", RowIdentity("s1", "A-03", "model-x", 0))],
        adapter=adapter,
    )

    assert [(item.status, item.failure_code) for item in batch.judgments] == [
        ("infrastructure_failure", "media_transport_failed")
    ]
    assert calls == []


def test_openai_compatible_adapter_accepts_a_normal_local_endpoint() -> None:
    assert (
        _completion_url("http://localhost:8000/v1") == "http://localhost:8000/v1/chat/completions"
    )
