# Experiments

- `experiments.csv` — the single source of truth for every run: hypothesis,
  exact change, config, measured results, and verdict. Append-only; never
  edit a past row's measured values, only append corrections as new rows.
- `experiment_template.yaml` — copy this to create a new experiment config.
- `results/` — one subdirectory per experiment ID, containing the resolved
  config actually used and raw logs/profiler output for that run. Not
  checked into git by default (see `.gitignore`); keep locally or push to
  external storage as the result set grows.

## Workflow

1. Copy `experiment_template.yaml`, fill it in, save under `configs/` or
   `experiments/results/<experiment_id>/config.yaml`.
2. Run it via `scripts/run_experiment.py` (currently a stub — see
   `TODO: integrate official AWS Trainium Frontier starter pipeline`).
3. Append the outcome as a new row in `experiments.csv`.
4. Run `scripts/compare_runs.py` to see how it ranks against prior runs.
5. Update the verdict (`KEEP` / `REJECT` / `RETEST` / `BASELINE`) once
   you've decided based on `docs/research_hypotheses.md`'s go/no-go
   criteria for that hypothesis.
