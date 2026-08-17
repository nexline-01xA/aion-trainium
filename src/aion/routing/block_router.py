"""Block-level conditional-compute routing.

Deliberately NOT token-level MoE-style routing. Decisions are made per
fixed contiguous block of tokens so that every path a real model executes
still operates on large, regular tensor tiles — irregular per-token
gather/scatter is expected to hurt Trainium2 TensorEngine utilization more
than it saves in FLOPs (see docs/research_hypotheses.md, H4, and
docs/trainium_notes.md).

This module decides *which blocks* get the full compute path. It does not
implement the "full path" vs "reduced path" itself — that lives in the
real model, once we have it.

TODO: integrate official AWS Trainium Frontier starter pipeline
  Call `BlockRouter.route(num_blocks, progress)` from the real model's
  forward pass to get a per-block boolean mask, then dispatch each block
  to the full or reduced computation path accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class RoutingMode(str, Enum):
    ALWAYS_FULL = "always_full"
    FIXED_RATIO = "fixed_ratio"
    SCHEDULED_RATIO = "scheduled_ratio"
    SCORE_BASED = "score_based"


@dataclass(frozen=True)
class RoutingConfig:
    mode: RoutingMode
    block_size_tokens: int = 128

    # fixed_ratio
    full_compute_fraction: float = 1.0

    # scheduled_ratio
    scheduled_start_fraction: float = 0.25
    scheduled_end_fraction: float = 0.9

    # score_based — interface only, see ScoreFn below. Not implemented.
    score_signal: Optional[str] = None
    score_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if self.block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive")
        if self.mode == RoutingMode.FIXED_RATIO and not (0.0 <= self.full_compute_fraction <= 1.0):
            raise ValueError("full_compute_fraction must be in [0, 1]")
        if self.mode == RoutingMode.SCHEDULED_RATIO:
            for name in ("scheduled_start_fraction", "scheduled_end_fraction"):
                v = getattr(self, name)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"{name} must be in [0, 1]")
        if self.mode == RoutingMode.SCORE_BASED:
            if self.score_signal is None or self.score_threshold is None:
                raise ValueError(
                    "score_based mode requires score_signal and score_threshold to be "
                    "set, but this mode is an interface only — see ScoreFn docstring; "
                    "it will raise NotImplementedError until a real scoring function "
                    "and a real model exist to validate it against."
                )


# A ScoreFn takes per-block features (whatever the real model exposes —
# e.g. hidden-state norms) and returns a per-block score used to decide
# routing. Left as a type alias with no implementation: the actual scoring
# function depends on signals only a real model's forward pass can produce,
# and per docs/research_hypotheses.md (H4) this mode is not attempted until
# fixed_ratio/scheduled_ratio show a validated win.
ScoreFn = Callable[[list[float]], list[float]]


@dataclass
class BlockRouter:
    config: RoutingConfig

    def route(self, num_blocks: int, progress: float = 0.0,
              scores: Optional[list[float]] = None) -> list[bool]:
        """Return a per-block boolean mask: True = full compute path.

        progress is training progress in [0, 1], used by scheduled_ratio.
        scores is only used (and required) by score_based mode.
        """
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if not (0.0 <= progress <= 1.0):
            raise ValueError(f"progress must be in [0, 1], got {progress}")

        if self.config.mode == RoutingMode.ALWAYS_FULL:
            return [True] * num_blocks

        if self.config.mode == RoutingMode.FIXED_RATIO:
            return self._ratio_mask(num_blocks, self.config.full_compute_fraction)

        if self.config.mode == RoutingMode.SCHEDULED_RATIO:
            start = self.config.scheduled_start_fraction
            end = self.config.scheduled_end_fraction
            fraction = start + (end - start) * progress
            return self._ratio_mask(num_blocks, fraction)

        if self.config.mode == RoutingMode.SCORE_BASED:
            raise NotImplementedError(
                "score_based routing is an interface only. It requires a real "
                "scoring signal computed from an actual model's forward pass "
                "(e.g. hidden-state norm, prediction entropy) and profiler "
                "validation that it doesn't break Trainium tile regularity. "
                "See docs/research_hypotheses.md H4 and "
                "docs/architecture.md for the gating criteria before this is "
                "implemented."
            )

        raise ValueError(f"unknown routing mode: {self.config.mode}")

    @staticmethod
    def _ratio_mask(num_blocks: int, fraction: float) -> list[bool]:
        """Deterministic mask with exactly round(num_blocks * fraction) True
        values, evenly spread across the sequence rather than clustered at
        one end (so downstream tiling stays as regular as possible)."""
        if not (0.0 <= fraction <= 1.0):
            raise ValueError("fraction must be in [0, 1]")
        num_full = round(num_blocks * fraction)
        if num_full <= 0:
            return [False] * num_blocks
        if num_full >= num_blocks:
            return [True] * num_blocks
        mask = [False] * num_blocks
        step = num_blocks / num_full
        for i in range(num_full):
            idx = min(int(i * step), num_blocks - 1)
            mask[idx] = True
        return mask
