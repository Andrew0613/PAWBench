"""Typed model of a benchmark package.

This is the single typed representation of the benchmark's science; every
other module (submission validation, judge evaluation, scoring) consumes
scenes, vocabularies, and reference distributions through these objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

Split = Literal["calibration", "coverage"]


@dataclass(frozen=True)
class ReferenceCount:
    """Exact rational reference mass for one calibration outcome."""

    numerator: int
    denominator: int

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True)
class Outcome:
    """One canonical outcome label of a scene."""

    label: str
    judge_notes: str | None
    reference_count: ReferenceCount | None  # calibration only


@dataclass(frozen=True)
class Scene:
    """One benchmark scene: a source image, an action, and an outcome space."""

    scene_id: str
    split: Split
    split_order: int
    action: str
    base_prompt: str
    source_image_path: str  # safe path relative to the package root
    outcomes: tuple[Outcome, ...]
    judge_notes: str | None

    @property
    def outcome_labels(self) -> tuple[str, ...]:
        """The scene's canonical outcome vocabulary."""
        return tuple(outcome.label for outcome in self.outcomes)

    @property
    def is_calibration(self) -> bool:
        return self.split == "calibration"

    def reference_distribution(self) -> Mapping[str, Fraction] | None:
        """Exact reference distribution; ``None`` for coverage scenes."""
        if not self.is_calibration:
            return None
        return {
            outcome.label: outcome.reference_count.fraction  # type: ignore[union-attr]
            for outcome in self.outcomes
        }


@dataclass(frozen=True)
class PromptSet:
    """A named prompt bank: per scene, one prompt per outcome label."""

    name: str
    prompts_by_scene: Mapping[str, Mapping[str, str]]  # scene_id -> label -> prompt


@dataclass(frozen=True)
class Benchmark:
    """A validated benchmark package, materialized from a local directory."""

    dataset_id: str
    benchmark_revision: int
    root: Path
    scenes: tuple[Scene, ...]
    prompt_sets: Mapping[str, PromptSet]
    formal_repeats: int | None

    def scene_by_id(self, scene_id: str) -> Scene:
        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        raise KeyError(f"unknown scene_id: {scene_id!r}")

    def scenes_in_split(self, split: Split) -> tuple[Scene, ...]:
        return tuple(
            sorted(
                (scene for scene in self.scenes if scene.split == split),
                key=lambda scene: scene.split_order,
            )
        )

    def resolve_image_path(self, scene: Scene) -> Path:
        """Resolve a scene's source image, guaranteed to stay inside the package."""
        resolved = (self.root / scene.source_image_path).resolve()
        if not resolved.is_relative_to(self.root.resolve()):
            # Unreachable after validation; kept as cheap defense in depth.
            from pawbench.errors import PawbenchError

            raise PawbenchError(
                f"scene {scene.scene_id}: image path escapes the package root"
            )
        return resolved
