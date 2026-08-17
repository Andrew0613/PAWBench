from __future__ import annotations

import json

import pytest

from pawbench._paweval import _client, _judge, _media
from pawbench._paweval._client import ProviderFailure, VLMConfig
from pawbench._paweval._parser import MalformedResponse, parse_axis
from pawbench._paweval._prompt import OUTCOME_AXIS, TRUST_AXIS, render_prompt


def item() -> dict:
    return {
        "sample_id": "sample-01__r000",
        "scene_id": "A-01",
        "track": "calibration",
        "model_or_lane": "model-x",
        "repeat_index": 0,
        "source_image_path": "/private/source.png",
        "video_path": "/private/video.mp4",
        "scene": {
            "scene_id": "A-01",
            "action": "Flick the coin once.",
            "outcome_labels": ["heads", "tails"],
            "reference_distribution": {"heads": 0.5, "tails": 0.5},
        },
    }


def outcome(*, readable: bool = True) -> str:
    return json.dumps(
        {
            "scene_id": "A-01",
            "outcome_readable": readable,
            "outcome_in_schema": "IN_SCHEMA" if readable else "NOT_READABLE",
            "outcome_label": "heads" if readable else None,
            "observed_result": "heads" if readable else "not visible",
            "failure_axes": [],
        }
    )


def trust() -> str:
    return json.dumps(
        {
            "scene_id": "A-01",
            "status": "TRUSTED",
            "scene_grounding": {"status": "PASS", "notes": ""},
            "action_execution": {
                "status": "PASS",
                "target_hit": "YES",
                "object_acquired": "YES",
                "action_spec_followed": "YES",
                "notes": "",
            },
            "object_continuity": {"status": "PASS", "notes": ""},
            "physical_process": {"status": "PASS", "failure_phase": "NONE", "notes": ""},
            "failure_axes": [],
        }
    )


def config() -> VLMConfig:
    return VLMConfig(base_url="https://judge.example/v1", model="judge", api_key_env="TEST_KEY")


def test_prompts_render_scene_rubrics_without_leaking_calibration_reference() -> None:
    prompt = render_prompt(OUTCOME_AXIS, item())

    assert "Flick the coin once." in prompt
    assert "- heads" in prompt
    assert "reference_distribution" not in prompt
    assert "0.5" not in prompt


def test_judgment_returns_separate_axes_and_safe_metric_row(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter([outcome(), trust()])
    monkeypatch.setattr(_judge, "build_messages", lambda prompt, item: [{"type": "text", "text": prompt}])
    monkeypatch.setattr(_judge, "complete", lambda config, messages: next(responses))

    row = _judge.judge_item(item(), config())

    assert row["observation"] == "outcome"
    assert row["outcome_label"] == "heads"
    assert row["outcome_readout"]["scene_id"] == "A-01"
    assert row["trustworthiness_audit"]["status"] == "TRUSTED"
    assert not {"source_image_path", "video_path", "base_url", "api_key_env", "raw_response"} & set(row)


def test_unreadable_outcome_is_a_null_but_keeps_the_trustworthiness_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter([outcome(readable=False), trust()])
    monkeypatch.setattr(_judge, "build_messages", lambda prompt, item: [])
    monkeypatch.setattr(_judge, "complete", lambda config, messages: next(responses))

    row = _judge.judge_item(item(), config())

    assert row["observation"] == "null_observation"
    assert "outcome_label" not in row
    assert row["trustworthiness_audit"]["status"] == "TRUSTED"


def test_malformed_axis_and_missing_media_become_infrastructure_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_judge, "build_messages", lambda prompt, item: [])
    monkeypatch.setattr(_judge, "complete", lambda config, messages: "not-json")
    assert _judge.judge_item(item(), config())["failure_code"] == "malformed_outcome_readout"

    with pytest.raises(_media.MediaFailure, match="missing_source_image_path"):
        _media.build_messages("prompt", item())
    assert (_media.FRAME_COUNT, _media.MAX_IMAGE_EDGE) == (8, 768)


def test_parser_rejects_labels_outside_the_scene_rubric() -> None:
    payload = json.loads(outcome())
    payload["outcome_label"] = "not-a-label"
    with pytest.raises(MalformedResponse, match="scene-policy label"):
        parse_axis(OUTCOME_AXIS, json.dumps(payload), item()["scene"])


def test_client_retries_only_the_bounded_transient_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "secret")
    attempts = iter([ProviderFailure("timeout"), '{"choices":[{"message":{"content":"{}"}}]}'])

    def fake_post(config: VLMConfig, api_key: str, payload: dict) -> str:
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(_client, "_post", fake_post)
    monkeypatch.setattr(_client.time, "sleep", lambda seconds: None)
    assert _client.complete(config(), []) == '{"choices":[{"message":{"content":"{}"}}]}'
