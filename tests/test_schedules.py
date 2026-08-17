import pytest

from aion.routing.block_router import BlockRouter, RoutingConfig, RoutingMode
from aion.schedules.compute_annealing import ComputeAnnealingSchedule, Stage


# --- Compute annealing schedule ---

def _two_stage_schedule():
    return ComputeAnnealingSchedule(
        [
            Stage(progress=0.0, active_layer_fraction=0.5, mlp_width_fraction=0.5,
                  sequence_length_fraction=0.5),
            Stage(progress=1.0, active_layer_fraction=1.0, mlp_width_fraction=1.0,
                  sequence_length_fraction=1.0),
        ]
    )


def test_capacity_at_endpoints():
    sched = _two_stage_schedule()
    start = sched.capacity_at(0.0)
    end = sched.capacity_at(1.0)
    assert start.active_layer_fraction == pytest.approx(0.5)
    assert end.active_layer_fraction == pytest.approx(1.0)


def test_capacity_at_midpoint_interpolates_linearly():
    sched = _two_stage_schedule()
    mid = sched.capacity_at(0.5)
    assert mid.active_layer_fraction == pytest.approx(0.75)
    assert mid.mlp_width_fraction == pytest.approx(0.75)
    assert mid.sequence_length_fraction == pytest.approx(0.75)


def test_schedule_requires_start_and_end_at_0_and_1():
    with pytest.raises(ValueError):
        ComputeAnnealingSchedule(
            [
                Stage(progress=0.1, active_layer_fraction=0.5, mlp_width_fraction=0.5,
                      sequence_length_fraction=0.5),
                Stage(progress=1.0, active_layer_fraction=1.0, mlp_width_fraction=1.0,
                      sequence_length_fraction=1.0),
            ]
        )


def test_schedule_requires_at_least_two_stages():
    with pytest.raises(ValueError):
        ComputeAnnealingSchedule(
            [Stage(progress=0.0, active_layer_fraction=1.0, mlp_width_fraction=1.0,
                   sequence_length_fraction=1.0)]
        )


def test_stage_rejects_out_of_range_fraction():
    with pytest.raises(ValueError):
        Stage(progress=0.0, active_layer_fraction=1.5, mlp_width_fraction=0.5,
              sequence_length_fraction=0.5)


def test_schedule_from_config():
    sched = ComputeAnnealingSchedule.from_config(
        [
            {"progress": 0.0, "active_layer_fraction": 0.5, "mlp_width_fraction": 0.5,
             "sequence_length_fraction": 0.5},
            {"progress": 1.0, "active_layer_fraction": 1.0, "mlp_width_fraction": 1.0,
             "sequence_length_fraction": 1.0},
        ]
    )
    assert sched.capacity_at(0.0).active_layer_fraction == pytest.approx(0.5)


# --- Block router ---

def test_always_full_routes_everything():
    router = BlockRouter(RoutingConfig(mode=RoutingMode.ALWAYS_FULL))
    mask = router.route(num_blocks=8)
    assert mask == [True] * 8


def test_fixed_ratio_routes_correct_count():
    router = BlockRouter(RoutingConfig(mode=RoutingMode.FIXED_RATIO, full_compute_fraction=0.5))
    mask = router.route(num_blocks=10)
    assert sum(mask) == 5


def test_fixed_ratio_zero_fraction():
    router = BlockRouter(RoutingConfig(mode=RoutingMode.FIXED_RATIO, full_compute_fraction=0.0))
    mask = router.route(num_blocks=10)
    assert sum(mask) == 0


def test_fixed_ratio_full_fraction():
    router = BlockRouter(RoutingConfig(mode=RoutingMode.FIXED_RATIO, full_compute_fraction=1.0))
    mask = router.route(num_blocks=10)
    assert sum(mask) == 10


def test_scheduled_ratio_increases_with_progress():
    router = BlockRouter(
        RoutingConfig(
            mode=RoutingMode.SCHEDULED_RATIO,
            scheduled_start_fraction=0.2,
            scheduled_end_fraction=0.8,
        )
    )
    early = sum(router.route(num_blocks=100, progress=0.0))
    late = sum(router.route(num_blocks=100, progress=1.0))
    assert early < late
    assert early == pytest.approx(20, abs=1)
    assert late == pytest.approx(80, abs=1)


def test_score_based_is_interface_only():
    config = RoutingConfig(mode=RoutingMode.SCORE_BASED, score_signal="hidden_state_norm",
                            score_threshold=0.5)
    router = BlockRouter(config)
    with pytest.raises(NotImplementedError):
        router.route(num_blocks=4, scores=[0.1, 0.9, 0.3, 0.7])


def test_score_based_config_requires_signal_and_threshold():
    with pytest.raises(ValueError):
        RoutingConfig(mode=RoutingMode.SCORE_BASED)


def test_router_rejects_nonpositive_num_blocks():
    router = BlockRouter(RoutingConfig(mode=RoutingMode.ALWAYS_FULL))
    with pytest.raises(ValueError):
        router.route(num_blocks=0)
