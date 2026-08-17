"""Render the two PAWEval prompts from one benchmark scene policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


OUTCOME_AXIS = "outcome_readout"
TRUST_AXIS = "trustworthiness_audit"
AXES = (OUTCOME_AXIS, TRUST_AXIS)


def render_prompt(axis: str, item: Mapping[str, Any]) -> str:
    """Render one fixed PAWEval axis prompt without exposing reference counts."""

    if axis not in AXES:
        raise ValueError(f"unknown PAWEval axis: {axis!r}")
    scene = item.get("scene")
    if not isinstance(scene, Mapping):
        raise ValueError("judgment item requires a scene policy")
    scene_id = _required(scene, "scene_id")
    action = _required(scene, "action")
    labels = scene.get("outcome_labels")
    if not isinstance(labels, list) or not all(isinstance(label, str) and label for label in labels):
        raise ValueError(f"scene {scene_id} requires outcome_labels")
    repeat = _required(item, "repeat_index")
    rubric = "\n".join(f"- {label}" for label in labels)
    evidence = (
        f"Scene {scene_id}. Action: {action}\n"
        "The first image is the source image; subsequent images are uniformly sampled video frames."
    )
    if axis == OUTCOME_AXIS:
        return f"""# PAWEval Outcome Readout

Scene: {scene_id}
Repeat: {repeat}

Use the evidence to return one outcome label. Do not judge trustworthiness in this axis.
Return exactly one JSON object, with no markdown or prose:

{{
  \"scene_id\": \"{scene_id}\",
  \"outcome_readable\": true,
  \"outcome_in_schema\": \"IN_SCHEMA\",
  \"outcome_label\": \"<one rubric label or null>\",
  \"observed_result\": \"<short evidence-grounded observation>\",
  \"failure_axes\": []
}}

If the result is unreadable, use `false`, `NOT_READABLE`, and a null label. If it is readable but outside the rubric, use `OUT_OF_SCHEMA` and a null label.

Evidence:
{evidence}

Canonical outcome labels:
{rubric}
"""
    return f"""# PAWEval Trustworthiness Audit

Scene: {scene_id}
Repeat: {repeat}

Audit whether the evidence supports strict reporting. Do not choose the outcome label in this axis.
Return exactly one JSON object, with no markdown or prose:

{{
  \"scene_id\": \"{scene_id}\",
  \"status\": \"TRUSTED\",
  \"scene_grounding\": {{\"status\": \"PASS\", \"notes\": \"\"}},
  \"action_execution\": {{\"status\": \"PASS\", \"target_hit\": \"YES\", \"object_acquired\": \"YES\", \"action_spec_followed\": \"YES\", \"notes\": \"\"}},
  \"object_continuity\": {{\"status\": \"PASS\", \"notes\": \"\"}},
  \"physical_process\": {{\"status\": \"PASS\", \"failure_phase\": \"NONE\", \"notes\": \"\"}},
  \"failure_axes\": []
}}

The top-level status is one of TRUSTED, QUESTIONABLE, UNTRUSTED, or UNCERTAIN. Keep notes empty and place concrete problems in failure_axes.

Evidence:
{evidence}

Scene policy:
Action: {action}
"""


def _required(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, (str, int)) or value == "":
        raise ValueError(f"judgment item requires {field}")
    return str(value)
