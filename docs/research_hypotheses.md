# Research Hypotheses

Standing hypothesis list for AION. Each hypothesis is tracked independently
in `experiments/experiments.csv` and must be individually validated (KEEP)
before being combined with others (see `docs/competition_strategy.md`,
Tier 4).

---

## H1 — Better short-run LR/optimizer recipe can lower val_bpb

**Motivation.** Short, fixed-budget runs are known to be highly sensitive to
learning-rate schedule shape and optimizer configuration; the "obvious"
default recipe is rarely optimal for a 30-minute budget.

**Expected benefit.** Lower val_bpb at equal wall-clock time, no throughput
cost.

**Failure mode.** Overfitting the LR search to noise from a single seed;
apparent wins that don't reproduce.

**Required measurement.** val_bpb, tokens/sec, and step count across at
least 2 seeds per candidate configuration.

**Go/no-go criterion.** KEEP only if the improvement holds across seeds and
exceeds run-to-run noise (established from repeated baseline runs).

---

## H2 — Trainium-friendly model dimensions can improve wall-clock throughput

**Motivation.** Matmul shapes that don't align with Trainium2's TensorEngine
tile sizes waste cycles regardless of theoretical FLOP count.

**Expected benefit.** Higher tokens/sec at equal or better val_bpb, purely
from better hardware utilization.

**Failure mode.** Reshaping dimensions to "nicer" numbers changes effective
model capacity and hurts val_bpb even though throughput improves — a net
loss.

**Required measurement.** Profiler utilization numbers (TensorEngine
occupancy) before/after, plus val_bpb and tokens/sec.

**Go/no-go criterion.** KEEP only if val_bpb does not regress; throughput
gains alone are not sufficient (see `compare_runs.py` ranking rule).

---

## H3 — Progressive compute allocation can outperform static capacity

**Motivation.** Early in training, gradients are large and coarse; full
model capacity may be under-utilized relative to its cost. Ramping capacity
up over the run could let more optimizer steps happen early, then use full
capacity once the model is closer to convergence.

**Expected benefit.** Lower val_bpb at fixed wall-clock time versus a static
architecture of equivalent final capacity.

**Failure mode.** The step-time savings from reduced early capacity are
outweighed by loss of representational capacity when it's actually needed;
or the transition points introduce instability.

**Required measurement.** val_bpb curve over training, wall-clock time,
comparison against static-capacity baseline of matched final parameter
count.

**Go/no-go criterion.** KEEP only if final val_bpb improves versus the
static baseline; RETEST if results are schedule-shape-sensitive before
committing to one ramp.

---

## H4 — Block-level conditional computation can improve useful learning per second

**Motivation.** Not all token blocks carry equal information; spending full
MLP computation on low-information blocks may be wasted FLOPs that could go
toward more optimizer steps elsewhere.

**Expected benefit.** Lower val_bpb per unit wall-clock time versus a
uniform-computation model of the same peak capacity.

**Failure mode.** Routing overhead, irregular tile shapes, or DMA overhead
from block selection erase or exceed the FLOP savings; routing signal is
too noisy to be useful at 50M scale.

**Required measurement.** val_bpb, tokens/sec, and TensorEngine utilization
for `fixed_ratio` and `scheduled_ratio` modes versus `always_full` control.

**Go/no-go criterion.** KEEP only if wall-clock val_bpb improves over the
`always_full` control at matched parameter count. `score_based` routing is
not attempted until `fixed_ratio`/`scheduled_ratio` show a validated win.

---

## H5 — Profiler-guided fusion can reduce memory/launch overhead

**Motivation.** Small, frequently-called ops (norm+residual, activation+
projection) can be memory-bandwidth-bound rather than compute-bound;
fusing them into custom NKI kernels can reduce launch and memory-traffic
overhead.

**Expected benefit.** Higher tokens/sec at unchanged val_bpb (this is a
systems optimization, not a learning-quality change — it should be
loss-neutral by construction).

**Failure mode.** Kernel is correct but not actually faster than the
compiler-generated fused op; or introduces numerical differences that shift
val_bpb.

**Required measurement.** Isolated micro-benchmark of the kernel versus the
baseline op, plus an end-to-end run confirming val_bpb is unchanged
(within noise) and wall-clock improves.

**Go/no-go criterion.** KEEP only if end-to-end wall-clock improves and
val_bpb does not regress. A kernel that is "elegant" but not measurably
faster is REJECTed.
