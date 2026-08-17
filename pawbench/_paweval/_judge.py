"""Private PAWEval judgment path for one benchmark item and generated video."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._client import ProviderFailure, VLMConfig, complete
from ._media import MediaFailure, build_messages
from ._parser import MalformedResponse, parse_axis
from ._prompt import AXES, OUTCOME_AXIS, render_prompt


def judge_item(item: Mapping[str, Any], config: VLMConfig) -> dict[str, Any]:
    """Return one metric-ready PAWEval judgment without metric aggregation.

    ``item`` binds a normalized sample identity, scene policy, source image,
    and generated-video path. Paths and provider configuration are used only
    during the request and never appear in the returned row.
    """

    identity = _identity(item)
    payloads: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        try:
            prompt = render_prompt(axis, item)
            messages = build_messages(prompt, item)
            payloads[axis] = parse_axis(axis, complete(config, messages), item["scene"])
        except MediaFailure as exc:
            return _failure(identity, str(exc))
        except ProviderFailure as exc:
            return _failure(identity, str(exc))
        except MalformedResponse:
            return _failure(identity, f"malformed_{axis}")

    outcome = payloads[OUTCOME_AXIS]
    is_null = outcome["outcome_readable"] is False or outcome["outcome_in_schema"] in {
        "NOT_READABLE",
        "OUT_OF_SCHEMA",
    }
    row = {
        **identity,
        "observation": "null_observation" if is_null else "outcome",
        "outcome_readout": outcome,
        "trustworthiness_audit": payloads["trustworthiness_audit"],
    }
    if not is_null:
        row["outcome_label"] = outcome["outcome_label"]
    return row


def _identity(item: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("sample_id", "scene_id", "track", "model_or_lane", "repeat_index")
    identity = {field: item.get(field) for field in fields}
    if any(value in (None, "") for value in identity.values()):
        raise ValueError("judgment item has an incomplete identity")
    if identity["track"] not in {"calibration", "coverage"}:
        raise ValueError("judgment item track must be calibration or coverage")
    scene = item.get("scene")
    if not isinstance(scene, Mapping) or scene.get("scene_id") != identity["scene_id"]:
        raise ValueError("judgment item scene policy does not match scene_id")
    return identity


def _failure(identity: Mapping[str, Any], code: str) -> dict[str, Any]:
    return {**identity, "observation": "infrastructure_failure", "failure_code": code}
