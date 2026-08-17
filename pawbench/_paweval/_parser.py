"""Extract and validate structured PAWEval axis responses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class MalformedResponse(ValueError):
    """The provider did not return a valid response for the requested axis."""


def parse_axis(axis: str, text: str, scene: Mapping[str, Any]) -> dict[str, Any]:
    """Extract one JSON object and validate it against the fixed PAWEval axes."""

    payload = _extract_object(text)
    scene_id = scene.get("scene_id")
    if payload.get("scene_id") != scene_id:
        raise MalformedResponse("response scene_id does not match the benchmark scene")
    if axis == "outcome_readout":
        return _parse_outcome(payload, scene)
    if axis == "trustworthiness_audit":
        return _parse_trustworthiness(payload)
    raise ValueError(f"unknown PAWEval axis: {axis!r}")


def _extract_object(text: str) -> dict[str, Any]:
    candidates = [text]
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(json.dumps(payload))
            break
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise MalformedResponse("no JSON object found in provider response")


def _parse_outcome(payload: Mapping[str, Any], scene: Mapping[str, Any]) -> dict[str, Any]:
    required = {"scene_id", "outcome_readable", "outcome_in_schema", "outcome_label", "observed_result", "failure_axes"}
    _require_exact_fields(payload, required)
    readable = payload["outcome_readable"]
    status = payload["outcome_in_schema"]
    label = payload["outcome_label"]
    if not isinstance(readable, bool) or status not in {"IN_SCHEMA", "NOT_READABLE", "OUT_OF_SCHEMA"}:
        raise MalformedResponse("invalid outcome status")
    if label is not None and not isinstance(label, str):
        raise MalformedResponse("outcome_label must be a string or null")
    if not isinstance(payload["observed_result"], (str, type(None))) or not _strings(payload["failure_axes"]):
        raise MalformedResponse("invalid outcome response fields")
    labels = scene.get("outcome_labels")
    if not isinstance(labels, list):
        raise MalformedResponse("scene policy has no outcome labels")
    canonical = {value.lower(): value for value in labels if isinstance(value, str)}
    if status == "IN_SCHEMA":
        if not readable or not isinstance(label, str) or label.lower() not in canonical:
            raise MalformedResponse("IN_SCHEMA outcome must name a scene-policy label")
        label = canonical[label.lower()]
    elif label is not None:
        raise MalformedResponse("non-readable outcome must not carry a label")
    return {**payload, "outcome_label": label}


def _parse_trustworthiness(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {"scene_id", "status", "scene_grounding", "action_execution", "object_continuity", "physical_process", "failure_axes"}
    _require_exact_fields(payload, required)
    if payload["status"] not in {"TRUSTED", "QUESTIONABLE", "UNTRUSTED", "UNCERTAIN"}:
        raise MalformedResponse("invalid trustworthiness status")
    if not _strings(payload["failure_axes"]):
        raise MalformedResponse("trustworthiness failure_axes must be strings")
    for name in ("scene_grounding", "object_continuity", "physical_process"):
        block = payload[name]
        if not isinstance(block, dict) or block.get("status") not in {"PASS", "FAIL", "UNCLEAR", "NOT_APPLICABLE"}:
            raise MalformedResponse(f"invalid {name} block")
    action = payload["action_execution"]
    if not isinstance(action, dict) or action.get("status") not in {"PASS", "FAIL", "UNCLEAR", "NOT_APPLICABLE"}:
        raise MalformedResponse("invalid action_execution block")
    for field in ("target_hit", "object_acquired", "action_spec_followed"):
        if action.get(field) not in {"YES", "NO", "UNCLEAR", "NOT_APPLICABLE"}:
            raise MalformedResponse(f"invalid action_execution.{field}")
    return dict(payload)


def _require_exact_fields(payload: Mapping[str, Any], required: set[str]) -> None:
    if set(payload) != required:
        raise MalformedResponse("response fields do not match the PAWEval axis schema")


def _strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
