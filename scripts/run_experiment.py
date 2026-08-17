#!/usr/bin/env python3
"""Launch one experiment and append its result to experiments/experiments.csv.

TODO: integrate official AWS Trainium Frontier starter pipeline
  This script currently cannot train anything — there is no training
  pipeline in this repository yet. Once the AWS starter repo is available,
  replace `_run_training_stub()` with a call into its real entry point
  (e.g. subprocess.run([...]) or a direct import), capturing:
    wall_clock_seconds, steps, tokens_processed, val_bpb, compile_time_seconds
  from its real output instead of raising NotImplementedError.

Usage (once wired up):
    python scripts/run_experiment.py \
        --experiment-id E001 \
        --config configs/baseline.yaml \
        --hypothesis "BASELINE" \
        --change "none" \
        --verdict BASELINE
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aion.experiment import ExperimentRecord, append_experiment  # noqa: E402
from aion.logging_utils import get_logger  # noqa: E402

log = get_logger(__name__)

EXPERIMENTS_CSV = REPO_ROOT / "experiments" / "experiments.csv"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _run_training_stub(config_path: Path) -> dict:
    """Placeholder for the real training invocation.

    Intentionally raises rather than returning fabricated numbers — this
    repo does not manufacture benchmark results.
    """
    raise NotImplementedError(
        "No training pipeline is wired in yet. "
        "TODO: integrate official AWS Trainium Frontier starter pipeline. "
        f"(would have used config: {config_path})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--change", required=True)
    parser.add_argument("--verdict", default="RETEST",
                         choices=["KEEP", "REJECT", "RETEST", "BASELINE"])
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and config path without attempting to train.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        parser.error(f"config not found: {args.config}")

    if args.dry_run:
        log.info("Dry run OK. Config found: %s", args.config)
        log.info("No training was attempted (--dry-run).")
        return

    log.info("Launching experiment %s with config %s", args.experiment_id, args.config)
    results = _run_training_stub(args.config)  # will raise until integrated

    record = ExperimentRecord(
        experiment_id=args.experiment_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit(),
        hypothesis=args.hypothesis,
        change=args.change,
        config=str(args.config),
        verdict=args.verdict,
        notes=args.notes,
        **results,
    )
    append_experiment(EXPERIMENTS_CSV, record)
    log.info("Recorded experiment %s -> %s", args.experiment_id, EXPERIMENTS_CSV)


if __name__ == "__main__":
    main()
