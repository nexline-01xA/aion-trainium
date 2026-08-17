# Architecture Notes

**Status: no official starter model code available yet.** This document
records what we currently *expect* the baseline architecture to look like,
based on the public nanochat / modded-nanogpt lineage referenced by the
competition, plus the modifications AION intends to layer on top. Nothing
here is confirmed until we read the actual AWS starter repository.

## Expected baseline (unconfirmed prior)

- Decoder-only transformer, RoPE positional encoding.
- RMSNorm, pre-norm placement.
- ReLU² (or similar cheap gated) MLP rather than SwiGLU, for FLOP efficiency
  at small scale.
- Muon optimizer for hidden-layer matrices, AdamW for embeddings/head/norms.
- No dropout.
- Sequence-packed batches with document-boundary masking.

`TODO: integrate official AWS Trainium Frontier starter pipeline` — replace
this section with the real architecture once `configs/baseline.yaml` is
filled in from the starter repo.

## AION additions (framework only, not yet attached to a real model)

### 1. Compute annealing

A schedule that maps *training progress* (0.0 → 1.0) to a set of capacity
fractions (`active_layer_fraction`, `mlp_width_fraction`,
`sequence_length_fraction`). Implemented in
`src/aion/schedules/compute_annealing.py` as a pure function of progress —
it has no dependency on any specific model class yet, so it can be wired
into whatever transformer implementation the starter repo provides.

### 2. Block-level conditional computation

A routing abstraction (`src/aion/routing/block_router.py`) that decides,
per fixed block of tokens (not per token), whether that block receives the
full computation path or a reduced one. Four modes are defined:

- `always_full` — routing disabled, useful as a control condition.
- `fixed_ratio` — a constant fraction of blocks get the full path.
- `scheduled_ratio` — the fraction changes over training progress (can be
  driven by the same schedule as compute annealing).
- `score_based` — **interface only**. The actual scoring function (hidden
  state norm, prediction entropy, etc.) is deliberately unimplemented until
  we have a real model to compute those signals from and a profiler to
  confirm it doesn't destroy Trainium tile regularity.

### 3. Trainium/NKI systems layer

No kernels exist yet. See `kernels/nki/README.md` — kernels are written
only after profiler evidence identifies a genuine bottleneck in the real
pipeline.

## Open questions (to resolve once starter repo is available)

- Exact hidden dimension, depth, head count/dimension of the baseline model.
- Whether attention uses a fused kernel already, or a naive implementation.
- What precision (BF16/FP8/FP32) the baseline runs in by default.
- Where in the training loop `val_bpb` is computed and how much of the
  30-minute budget that eval costs.
