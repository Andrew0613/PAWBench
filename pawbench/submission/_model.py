"""Typed model of a video submission.

The denominator is a value here: a :class:`Submission` always covers its
scope x repeats grid exactly, one :class:`Item` per slot. A slot that
produced no video is a failure item, never a missing row — the grid cannot
shrink silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pawbench.benchmark import Split

ItemStatus = Literal["produced", "model_failure", "infrastructure_failure"]

MODEL_FAILURE_CODES = frozenset({"generation_failed", "generation_timeout", "generation_refused"})
INFRASTRUCTURE_FAILURE_CODES = frozenset(
    {"media_missing", "media_corrupt", "media_undecodable"}
)

VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov")
DEFAULT_NAMING = "{scene_id}__r{repeat_index:03d}"

SCHEMA_VERSION = "pawbench.submission/v1"


@dataclass(frozen=True)
class Item:
    """One (scene, repeat) slot: a produced video, or an explicit failure."""

    scene_id: str
    repeat_index: int
    status: ItemStatus
    video_path: str | None = None  # required for produced items
    failure_code: str | None = None  # required for failure items

    @property
    def key(self) -> tuple[str, int]:
        return (self.scene_id, self.repeat_index)

    def to_dict(self) -> dict:
        record = {"scene_id": self.scene_id, "repeat_index": self.repeat_index}
        if self.status == "produced":
            record["status"] = "produced"
            record["video_path"] = self.video_path
        else:
            record["status"] = self.status
            record["failure_code"] = self.failure_code
        return record


@dataclass(frozen=True)
class Submission:
    """A validated submission, bound to a benchmark package."""

    submission_id: str
    dataset_id: str
    benchmark_revision: int
    system: str
    prompt_set: str
    splits: tuple[Split, ...]
    scene_ids: tuple[str, ...]  # resolved scope, in benchmark order
    repeats_per_scene: int
    items: tuple[Item, ...]

    @property
    def slot_count(self) -> int:
        """Always len(scene_ids) * repeats_per_scene: the fixed denominator."""
        return len(self.scene_ids) * self.repeats_per_scene

    def to_envelope(self) -> dict:
        """Serialize to the public submission.json shape (explicit scope)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "submission_id": self.submission_id,
            "benchmark": {
                "dataset_id": self.dataset_id,
                "benchmark_revision": self.benchmark_revision,
            },
            "system": self.system,
            "prompt_set": self.prompt_set,
            "scope": {
                "splits": list(self.splits),
                "scene_ids": list(self.scene_ids),
            },
            "repeats_per_scene": self.repeats_per_scene,
            "items": [item.to_dict() for item in self.items],
        }
