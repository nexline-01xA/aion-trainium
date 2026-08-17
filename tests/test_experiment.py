from pathlib import Path

import pytest

from aion.experiment import COLUMNS, ExperimentRecord, append_experiment, load_experiments


def _minimal_record(**overrides) -> ExperimentRecord:
    base = dict(
        experiment_id="E001",
        timestamp="2026-01-01T00:00:00+00:00",
        git_commit="deadbeef",
        hypothesis="BASELINE run",
        change="none",
        config="configs/baseline.yaml",
        verdict="BASELINE",
    )
    base.update(overrides)
    return ExperimentRecord(**base)


def test_valid_record_roundtrip_via_to_row():
    r = _minimal_record(val_bpb=1.234, wall_clock_seconds=1800.0)
    row = r.to_row()
    assert row["experiment_id"] == "E001"
    assert row["val_bpb"] == 1.234
    assert row["parameter_count"] == ""  # unset optional field -> empty string


def test_requires_experiment_id():
    with pytest.raises(ValueError):
        _minimal_record(experiment_id="")


def test_rejects_invalid_verdict():
    with pytest.raises(ValueError):
        _minimal_record(verdict="MAYBE")


def test_append_and_load_roundtrip(tmp_path: Path):
    csv_path = tmp_path / "experiments.csv"
    r1 = _minimal_record(experiment_id="E001", val_bpb=1.2)
    r2 = _minimal_record(experiment_id="E002", verdict="KEEP", val_bpb=1.05)

    append_experiment(csv_path, r1)
    append_experiment(csv_path, r2)

    loaded = load_experiments(csv_path)
    assert len(loaded) == 2
    assert loaded[0].experiment_id == "E001"
    assert loaded[0].val_bpb == pytest.approx(1.2)
    assert loaded[1].verdict == "KEEP"


def test_load_experiments_rejects_missing_columns(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("experiment_id,timestamp\nE001,2026-01-01\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiments(csv_path)


def test_columns_match_expected_schema():
    expected = {
        "experiment_id", "timestamp", "git_commit", "hypothesis", "change", "config",
        "parameter_count", "sequence_length", "batch_size", "learning_rate", "optimizer",
        "precision", "wall_clock_seconds", "steps", "tokens_processed", "tokens_per_second",
        "val_bpb", "delta_vs_baseline", "compile_time_seconds", "notes", "verdict",
    }
    assert set(COLUMNS) == expected
