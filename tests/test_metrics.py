import pytest

from aion.metrics import (
    RunMetrics,
    compile_overhead_fraction,
    relative_improvement,
    steps_per_second,
    tokens_per_second,
    val_bpb_delta,
)


def test_run_metrics_valid():
    m = RunMetrics(wall_clock_seconds=1800, steps=1000, tokens_processed=2_000_000, val_bpb=1.05)
    assert m.wall_clock_seconds == 1800


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(wall_clock_seconds=0, steps=1, tokens_processed=1, val_bpb=1.0),
        dict(wall_clock_seconds=1, steps=0, tokens_processed=1, val_bpb=1.0),
        dict(wall_clock_seconds=1, steps=1, tokens_processed=0, val_bpb=1.0),
        dict(wall_clock_seconds=1, steps=1, tokens_processed=1, val_bpb=1.0, compile_time_seconds=-1),
    ],
)
def test_run_metrics_invalid(kwargs):
    with pytest.raises(ValueError):
        RunMetrics(**kwargs)


def test_tokens_per_second():
    assert tokens_per_second(1000, 10) == 100.0


def test_tokens_per_second_rejects_nonpositive_time():
    with pytest.raises(ValueError):
        tokens_per_second(1000, 0)


def test_steps_per_second():
    assert steps_per_second(50, 10) == 5.0


def test_val_bpb_delta_worse():
    assert val_bpb_delta(1.1, 1.0) == pytest.approx(0.1)


def test_val_bpb_delta_better():
    assert val_bpb_delta(0.9, 1.0) == pytest.approx(-0.1)


def test_relative_improvement():
    assert relative_improvement(0.95, 1.0) == pytest.approx(0.05)


def test_relative_improvement_rejects_nonpositive_baseline():
    with pytest.raises(ValueError):
        relative_improvement(0.9, 0)


def test_compile_overhead_fraction():
    assert compile_overhead_fraction(60, 1800) == pytest.approx(60 / 1800)


def test_compile_overhead_fraction_rejects_negative_compile_time():
    with pytest.raises(ValueError):
        compile_overhead_fraction(-1, 1800)
