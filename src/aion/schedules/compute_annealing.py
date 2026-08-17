"""Compute annealing: maps training progress to a set of capacity fractions.

This is a pure, framework-level schedule abstraction. It has no dependency
on any real model class and does not touch a Transformer yet, because we
do not have the AWS starter model code.

TODO: integrate official AWS Trainium Frontier starter pipeline
  Wire `ComputeAnnealingSchedule.capacity_at(progress)` into the real
  training loop's per-step hook, and have the real model consult
  `active_layer_fraction` / `mlp_width_fraction` / `sequence_length_fraction`
  to decide how much of itself to execute at that step. Until that
  integration exists, this module is exercised only by unit tests
  (tests/test_schedules.py) with synthetic progress values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    """One control point in the annealing schedule.

    progress is training progress in [0.0, 1.0] (e.g. current_step / total_steps).
    The three fractions are each in (0.0, 1.0].
    """

    progress: float
    active_layer_fraction: float
    mlp_width_fraction: float
    sequence_length_fraction: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.progress <= 1.0):
            raise ValueError(f"progress must be in [0, 1], got {self.progress}")
        for name in ("active_layer_fraction", "mlp_width_fraction", "sequence_length_fraction"):
            value = getattr(self, name)
            if not (0.0 < value <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], got {value}")


@dataclass(frozen=True)
class CapacityFractions:
    active_layer_fraction: float
    mlp_width_fraction: float
    sequence_length_fraction: float


class ComputeAnnealingSchedule:
    """Piecewise-linear interpolation over a list of Stage control points.

    Stages must be sorted by ascending `progress` and must include a stage
    at progress=0.0 and a stage at progress=1.0 — the schedule does not
    extrapolate beyond the given control points.
    """

    def __init__(self, stages: list[Stage]):
        if len(stages) < 2:
            raise ValueError("at least two stages are required (start and end)")
        sorted_stages = sorted(stages, key=lambda s: s.progress)
        if sorted_stages != list(stages):
            raise ValueError("stages must be provided sorted by ascending progress")
        if sorted_stages[0].progress != 0.0:
            raise ValueError("first stage must have progress == 0.0")
        if sorted_stages[-1].progress != 1.0:
            raise ValueError("last stage must have progress == 1.0")
        progresses = [s.progress for s in sorted_stages]
        if len(set(progresses)) != len(progresses):
            raise ValueError("stage progress values must be unique")
        self.stages = sorted_stages

    def capacity_at(self, progress: float) -> CapacityFractions:
        if not (0.0 <= progress <= 1.0):
            raise ValueError(f"progress must be in [0, 1], got {progress}")

        # Exact match or before first stage.
        if progress <= self.stages[0].progress:
            s = self.stages[0]
            return CapacityFractions(
                s.active_layer_fraction, s.mlp_width_fraction, s.sequence_length_fraction
            )

        for lo, hi in zip(self.stages, self.stages[1:]):
            if lo.progress <= progress <= hi.progress:
                span = hi.progress - lo.progress
                t = 0.0 if span == 0 else (progress - lo.progress) / span
                return CapacityFractions(
                    active_layer_fraction=_lerp(lo.active_layer_fraction, hi.active_layer_fraction, t),
                    mlp_width_fraction=_lerp(lo.mlp_width_fraction, hi.mlp_width_fraction, t),
                    sequence_length_fraction=_lerp(
                        lo.sequence_length_fraction, hi.sequence_length_fraction, t
                    ),
                )

        # progress == 1.0, matched by the loop above in practice, but keep
        # an explicit fallback for float edge cases.
        s = self.stages[-1]
        return CapacityFractions(
            s.active_layer_fraction, s.mlp_width_fraction, s.sequence_length_fraction
        )

    @classmethod
    def from_config(cls, stage_dicts: list[dict]) -> "ComputeAnnealingSchedule":
        stages = [
            Stage(
                progress=d["progress"],
                active_layer_fraction=d["active_layer_fraction"],
                mlp_width_fraction=d["mlp_width_fraction"],
                sequence_length_fraction=d["sequence_length_fraction"],
            )
            for d in stage_dicts
        ]
        return cls(stages)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
