"""Experiment record schema and CSV I/O.

This models the columns in experiments/experiments.csv. It validates
structure only — it does not run training. The actual training invocation
is: `TODO: integrate official AWS Trainium Frontier starter pipeline`
(see scripts/run_experiment.py).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

VALID_VERDICTS = {"KEEP", "REJECT", "RETEST", "BASELINE"}

# Column order mirrors experiments/experiments.csv exactly.
COLUMNS = [
    "experiment_id",
    "timestamp",
    "git_commit",
    "hypothesis",
    "change",
    "config",
    "parameter_count",
    "sequence_length",
    "batch_size",
    "learning_rate",
    "optimizer",
    "precision",
    "wall_clock_seconds",
    "steps",
    "tokens_processed",
    "tokens_per_second",
    "val_bpb",
    "delta_vs_baseline",
    "compile_time_seconds",
    "notes",
    "verdict",
]


@dataclass
class ExperimentRecord:
    experiment_id: str
    timestamp: str
    git_commit: str
    hypothesis: str
    change: str
    config: str
    verdict: str
    parameter_count: Optional[int] = None
    sequence_length: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    optimizer: Optional[str] = None
    precision: Optional[str] = None
    wall_clock_seconds: Optional[float] = None
    steps: Optional[int] = None
    tokens_processed: Optional[int] = None
    tokens_per_second: Optional[float] = None
    val_bpb: Optional[float] = None
    delta_vs_baseline: Optional[float] = None
    compile_time_seconds: Optional[float] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if self.verdict not in VALID_VERDICTS:
            raise ValueError(
                f"verdict must be one of {sorted(VALID_VERDICTS)}, got {self.verdict!r}"
            )

    def to_row(self) -> dict:
        row = {}
        for f in fields(self):
            value = getattr(self, f.name)
            row[f.name] = "" if value is None else value
        return row


def load_experiments(csv_path: Path) -> list[ExperimentRecord]:
    """Load and validate every row in experiments.csv. Raises on malformed
    rows rather than silently skipping them."""
    records: list[ExperimentRecord] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"experiments.csv is missing columns: {sorted(missing)}")
        for raw in reader:
            records.append(_row_to_record(raw))
    return records


def _row_to_record(raw: dict) -> ExperimentRecord:
    def _opt(key: str, cast):
        v = raw.get(key, "")
        if v is None or v == "":
            return None
        return cast(v)

    return ExperimentRecord(
        experiment_id=raw["experiment_id"],
        timestamp=raw["timestamp"],
        git_commit=raw["git_commit"],
        hypothesis=raw["hypothesis"],
        change=raw["change"],
        config=raw["config"],
        verdict=raw["verdict"] or "RETEST",
        parameter_count=_opt("parameter_count", int),
        sequence_length=_opt("sequence_length", int),
        batch_size=_opt("batch_size", int),
        learning_rate=_opt("learning_rate", float),
        optimizer=raw.get("optimizer") or None,
        precision=raw.get("precision") or None,
        wall_clock_seconds=_opt("wall_clock_seconds", float),
        steps=_opt("steps", int),
        tokens_processed=_opt("tokens_processed", int),
        tokens_per_second=_opt("tokens_per_second", float),
        val_bpb=_opt("val_bpb", float),
        delta_vs_baseline=_opt("delta_vs_baseline", float),
        compile_time_seconds=_opt("compile_time_seconds", float),
        notes=raw.get("notes") or "",
    )


def append_experiment(csv_path: Path, record: ExperimentRecord) -> None:
    """Append one experiment record to experiments.csv. Never rewrites or
    edits existing rows."""
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record.to_row())
