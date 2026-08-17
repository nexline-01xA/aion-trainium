# Competition Strategy — Phase 1 Experiment Plan

Primary metric: **val_bpb (lower is better)**, measured after a strict
30-minute wall-clock training run on a single Trainium2 chip.

Rule: **Tier 1 must be exhausted (and its winners locked in) before Tier 2
begins; Tier 2 winners before Tier 3; combinations (Tier 4) only after each
individual component has an independently validated KEEP verdict** in
`experiments/experiments.csv`. No experiment below has been run yet — this
is a plan, not a result.

---

## Tier 1 — Baseline & hyperparameters (cheap, likely wins)

| # | Hypothesis | Code area | Expected Δval_bpb | Expected Δthroughput | Difficulty | Risk |
|---|---|---|---|---|---|---|
| 1 | Run baseline unmodified, log everything | none (measurement only) | — | — | trivial | none |
| 2 | Verify/enable graph caching so compile cost is paid once | runtime config | 0 | high+ | low | low |
| 3 | Sweep global batch size | training loop | med | med | low | low |
| 4 | Coarse LR sweep around default | optimizer config | med-high | 0 | low | low |
| 5 | Tune warmup length | schedule | low-med | 0 | low | low |
| 6 | Trapezoidal vs cosine decay | schedule | med | 0 | low | low |
| 7 | Confirm BF16 everywhere, no accidental FP32 upcast | precision | 0 | med-high | low | low |
| 8 | Try FP8 matmuls if Neuron SDK version supports it | precision | risk both ways | high | med | med |
| 9 | Sequence length sweep | data pipeline | med | high | low | med |
| 10 | Remove/overlap dataloader stalls | dataloader | 0 | med-high | med | low |
| 11 | Weight decay sweep | optimizer | low-med | 0 | low | low |
| 12 | Gradient clipping on/off/threshold | optimizer | low | 0 | low | low |
| 13 | Init scale sweep | model init | med | 0 | low | low |

## Tier 2 — Architecture

| # | Hypothesis | Code area | Expected Δval_bpb | Expected Δthroughput | Difficulty | Risk |
|---|---|---|---|---|---|---|
| 14 | Depth vs width reallocation at fixed param count | model config | med-high | med | med | med |
| 15 | MLP expansion ratio sweep | model | med | med | low | low |
| 16 | Head dimension/count sweep for tile-friendliness | attention | low-med | med | med | low |
| 17 | Norm placement variant | model | low | low | med | med |

## Tier 3 — Systems / NKI

| # | Hypothesis | Code area | Expected Δval_bpb | Expected Δthroughput | Difficulty | Risk |
|---|---|---|---|---|---|---|
| 18 | Fuse RMSNorm+residual via NKI, if profiler shows it's hot | kernels | 0 (indirect) | med-high | high | med |
| 19 | Fuse activation+projection via NKI, if profiler shows it's hot | kernels | 0 (indirect) | med | high | med |

## Tier 4 — AION-specific (combinations, gated on Tier 1–3 results)

| # | Hypothesis | Code area | Expected Δval_bpb | Expected Δthroughput | Difficulty | Risk |
|---|---|---|---|---|---|---|
| 20 | Compute annealing: minimal two-stage width ramp | schedules + training loop | speculative | speculative | high | high |
| 21 | Compute annealing: progressive layer activation | schedules + training loop | speculative | speculative | high | high |
| 22 | Block-level conditional compute, `fixed_ratio` mode | routing + model | speculative | speculative | high | high |
| 23 | Block-level conditional compute, `scheduled_ratio` mode | routing + model | speculative | speculative | high | high |
| 24 | Combined: best Tier 1 recipe + best Tier 2 architecture + validated Tier 4 mechanism | all | speculative | speculative | high | high |

Tier 4 rows are deliberately not numbered 1–20 from the original plan — they
extend it, since Tier 1–3 already covers the first 19 items and this file
tracks the full plan including the AION-specific work.

## Ablation requirement (final configuration)

Once a final configuration is selected, measure the incremental
contribution of each kept component in isolation, in this order:
baseline → +optimizer/schedule wins → +architecture wins → +compute
annealing → +conditional compute → +NKI kernels. Record each step's
val_bpb delta in `experiments/experiments.csv` with a shared `git_commit`
lineage so the ablation chain is reconstructible.
