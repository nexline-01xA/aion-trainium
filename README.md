# AION — Adaptive Intelligence-per-Operation Network

AION is a hardware-aware language-model training research project built for the
**AWS Trainium Frontier Competition**. It investigates whether compute allocation
and effective model capacity can be changed *dynamically*, during a single
training run, to improve learning under a strict Trainium2 wall-clock budget.

## Research question

> Can a language model learn more, under a fixed compute budget, by dynamically
> deciding where computation is worth spending — across tokens, blocks, and
> training time?

## Three pillars

1. **Compute Annealing** — the effective architecture (depth / MLP width /
   sequence length) changes over the course of a run instead of staying static.
2. **Hardware-Aware Conditional Computation** — computation is allocated
   unevenly across the input, but only at granularities (fixed blocks,
   sequence chunks, microbatches) that keep Trainium2's TensorEngine fed with
   large, regular matmul tiles. No per-token irregular routing.
3. **Trainium / NKI Systems Optimization** — kernel-level fusion decisions are
   made from profiler evidence, not intuition, and only kept if they measurably
   improve wall-clock throughput.

## Competition objective

Phase 1 of the competition trains a model from scratch on a single Trainium2
chip within a strict 30-minute wall-clock budget. The primary metric is:

```
validation bits-per-byte (val_bpb)   — lower is better
```

## Status

**This repository does not yet contain the official AWS competition starter
pipeline.** It is infrastructure — experiment tracking, config schema, a
compute-annealing schedule abstraction, and a block-level conditional-compute
routing interface — built ahead of time so that the moment the starter repo is
released, we can plug it in and start running controlled experiments
immediately instead of building tooling under time pressure.

Every placeholder integration point is marked:

```python
# TODO: integrate official AWS Trainium Frontier starter pipeline
```

**No benchmark numbers, val_bpb figures, or throughput claims appear anywhere
in this repository.** Numbers only get added after a real run on Trainium2
hardware, logged through `experiments/experiments.csv`.

## Methodology

Every change follows the same loop:

```
hypothesis → smallest testable implementation → controlled run
  → measurement against baseline → keep or revert → log to experiments.csv
```

One variable changes at a time until individually-validated wins are combined.
See `docs/research_hypotheses.md` for the standing hypothesis list and
`docs/competition_strategy.md` for the tiered experiment plan.

## Repository layout

```
docs/            research notes, hypotheses, strategy, Trainium/NKI notes
experiments/     experiment tracker (experiments.csv), per-experiment configs, results
configs/         YAML configs (baseline / compute annealing / conditional compute)
src/aion/        core library: metrics, experiment records, schedules, routing
scripts/         run_experiment.py, compare_runs.py, summarize_results.py
kernels/nki/     placeholder for custom NKI kernels (empty until profiler evidence exists)
tests/           unit tests that run without any Trainium hardware
```

## Getting started (development, no hardware required)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## When the AWS starter repository arrives

See `docs/reproducibility.md` for the exact list of files this repo expects
to replace or wire into the official pipeline.
