#!/usr/bin/env python3
"""Rank recorded experiments primarily by lowest val_bpb.

Secondary diagnostics (shown, not used for ranking): tokens_per_second,
compile_time_seconds. Throughput alone never determines rank — see
docs/competition_strategy.md and research_hypotheses.md go/no-go criteria.

Usage:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --verdict KEEP
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aion.experiment import load_experiments  # noqa: E402

EXPERIMENTS_CSV = REPO_ROOT / "experiments" / "experiments.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=EXPERIMENTS_CSV)
    parser.add_argument("--verdict", default=None,
                         choices=["KEEP", "REJECT", "RETEST", "BASELINE"])
    args = parser.parse_args()

    records = load_experiments(args.csv)
    if args.verdict:
        records = [r for r in records if r.verdict == args.verdict]

    scored = [r for r in records if r.val_bpb is not None]
    unscored = [r for r in records if r.val_bpb is None]

    scored.sort(key=lambda r: r.val_bpb)

    if not scored and not unscored:
        print("No experiments recorded yet in", args.csv)
        return

    if scored:
        header = f"{'rank':<5}{'experiment_id':<16}{'val_bpb':<12}{'tok/s':<12}{'compile_s':<11}{'verdict':<10}hypothesis"
        print(header)
        print("-" * len(header))
        for i, r in enumerate(scored, start=1):
            tok_s = f"{r.tokens_per_second:.1f}" if r.tokens_per_second is not None else "n/a"
            compile_s = f"{r.compile_time_seconds:.1f}" if r.compile_time_seconds is not None else "n/a"
            print(
                f"{i:<5}{r.experiment_id:<16}{r.val_bpb:<12.5f}{tok_s:<12}{compile_s:<11}{r.verdict:<10}{r.hypothesis}"
            )
    else:
        print("No scored experiments (val_bpb) yet.")

    if unscored:
        print(f"\n{len(unscored)} experiment(s) with no val_bpb recorded yet:")
        for r in unscored:
            print(f"  - {r.experiment_id}: {r.hypothesis} [{r.verdict}]")


if __name__ == "__main__":
    main()
