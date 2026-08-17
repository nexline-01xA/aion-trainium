# Reproducibility & AWS Starter Integration Plan

## Current state

This repository is infrastructure-only. It contains no AWS Trainium
Frontier starter code, no real model weights, and no benchmark results.
Every experiment record in `experiments/experiments.csv` will be tagged
with a real git commit hash so any KEEP verdict is exactly reproducible
from source.

## Files that will be replaced or wired in once the starter repo arrives

| This repo | Action when starter repo arrives |
|---|---|
| `configs/baseline.yaml` | Fill in real model/training/optimizer/runtime values from the starter repo's default config. |
| `docs/architecture.md` | Replace the "expected baseline" section with the confirmed real architecture. |
| `src/aion/schedules/compute_annealing.py` | Wire the existing progress→capacity-fraction schedule into the real training loop's per-step hook. |
| `src/aion/routing/block_router.py` | Attach `always_full` / `fixed_ratio` / `scheduled_ratio` modes to the real model's forward pass, at whatever granularity (block/microbatch) the real architecture supports cleanly. |
| `kernels/nki/` | Populate only after profiling the real pipeline and confirming a genuine bottleneck (see H5 in `docs/research_hypotheses.md`). |
| `scripts/run_experiment.py` | Point its training invocation at the starter repo's actual entry point instead of the current placeholder stub. |

## Reproducibility rules for every experiment

1. Every run is tagged with a git commit hash (`git rev-parse HEAD`) at the
   moment it was launched — no uncommitted-diff runs get recorded as KEEP.
2. Every run's exact config (the resolved YAML, not just the filename) is
   saved alongside its result in `experiments/results/`.
3. `val_bpb` comparisons are only valid between runs using the same
   evaluation harness and held-out data — if the starter repo's eval
   harness changes, old results are marked stale, not silently reused.
4. No experiment is marked KEEP off a single run; see the per-hypothesis
   "required measurement" sections in `docs/research_hypotheses.md` for
   seed/repeat requirements.
5. AWS SDK / Neuron SDK version is recorded per run (`docs/trainium_notes.md`
   environment section) since kernel and precision behavior can change
   between SDK versions.
