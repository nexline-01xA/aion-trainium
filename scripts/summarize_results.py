#!/usr/bin/env python3
"""Print summary statistics over recorded experiments: counts by verdict,
best KEEP result so far, and how many hypotheses (from
docs/research_hypotheses.md naming, H1-H5) have at least one KEEP.

Usage:
    python scripts/summarize_results.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aion.experiment import load_experiments  # noqa: E402

EXPERIMENTS_CSV = REPO_ROOT / "experiments" / "experiments.csv"


def main() -> None:
    records = load_experiments(EXPERIMENTS_CSV)

    if not records:
        print("No experiments recorded yet in", EXPERIMENTS_CSV)
        print("Run scripts/run_experiment.py once the AWS starter pipeline is integrated.")
        return

    verdict_counts = Counter(r.verdict for r in records)
    print("Experiments by verdict:")
    for verdict in ("BASELINE", "KEEP", "REJECT", "RETEST"):
        print(f"  {verdict:<10} {verdict_counts.get(verdict, 0)}")

    scored_keeps = [r for r in records if r.verdict == "KEEP" and r.val_bpb is not None]
    if scored_keeps:
        best = min(scored_keeps, key=lambda r: r.val_bpb)
        print(f"\nBest KEEP so far: {best.experiment_id} — val_bpb={best.val_bpb:.5f} "
              f"({best.hypothesis})")
    else:
        print("\nNo KEEP experiments with a recorded val_bpb yet.")

    total = len(records)
    print(f"\nTotal experiments recorded: {total}")


if __name__ == "__main__":
    main()
