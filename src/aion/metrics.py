"""Metric calculations for experiment tracking.

No hardware dependency — these are pure functions over numbers you supply
(measured wall-clock time, step counts, etc.) from a real run. Nothing in
this module invents or estimates a value that should come from hardware.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunMetrics:
    """Measured results for a single training run.

    Every field must come from an actual run. Do not populate this with
    guessed or estimated values.
    """

    wall_clock_seconds: float
    steps: int
    tokens_processed: int
    val_bpb: float
    compile_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.wall_clock_seconds <= 0:
            raise ValueError("wall_clock_seconds must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.tokens_processed <= 0:
            raise ValueError("tokens_processed must be positive")
        if self.compile_time_seconds < 0:
            raise ValueError("compile_time_seconds cannot be negative")


def tokens_per_second(tokens_processed: int, wall_clock_seconds: float) -> float:
    """Throughput in tokens/sec, including whatever overhead is baked into
    wall_clock_seconds (e.g. compile time), unless the caller has already
    subtracted it out."""
    if wall_clock_seconds <= 0:
        raise ValueError("wall_clock_seconds must be positive")
    return tokens_processed / wall_clock_seconds


def steps_per_second(steps: int, wall_clock_seconds: float) -> float:
    if wall_clock_seconds <= 0:
        raise ValueError("wall_clock_seconds must be positive")
    return steps / wall_clock_seconds


def val_bpb_delta(candidate_val_bpb: float, baseline_val_bpb: float) -> float:
    """Positive means candidate is worse (higher bpb); negative means better."""
    return candidate_val_bpb - baseline_val_bpb


def relative_improvement(candidate_val_bpb: float, baseline_val_bpb: float) -> float:
    """Fractional improvement over baseline. Positive = improvement.

    e.g. 0.05 means candidate's val_bpb is 5% lower (better) than baseline.
    """
    if baseline_val_bpb <= 0:
        raise ValueError("baseline_val_bpb must be positive")
    return (baseline_val_bpb - candidate_val_bpb) / baseline_val_bpb


def compile_overhead_fraction(compile_time_seconds: float, wall_clock_seconds: float) -> float:
    """Fraction of the total wall-clock budget spent on compilation."""
    if wall_clock_seconds <= 0:
        raise ValueError("wall_clock_seconds must be positive")
    if compile_time_seconds < 0:
        raise ValueError("compile_time_seconds cannot be negative")
    return compile_time_seconds / wall_clock_seconds
