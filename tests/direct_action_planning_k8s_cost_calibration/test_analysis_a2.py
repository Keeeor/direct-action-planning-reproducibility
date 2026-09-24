from __future__ import annotations

import math

import numpy as np
import pytest

from stage2_dynamic_budget.direct_action_planning_k8s_cost_calibration.analysis_a2 import (
    action_horizon_valid,
    bootstrap_ci,
    count_nonempty_lines,
    decide_claims,
    exact_sign_flip_p,
    finite_mean_or_none,
    holm_adjust,
    paired_effect,
    relative_cost_bootstrap,
)


def test_a2_statistics_have_hand_checkable_boundaries() -> None:
    low, high = bootstrap_ci([1.0] * 20, seed=7, draws=200)
    assert low == high == 1.0
    assert exact_sign_flip_p([1.0] * 20) == pytest.approx(2 / 2**20)
    assert holm_adjust([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert math.isinf(paired_effect([1.0] * 20))


def test_relative_cost_is_ratio_of_paired_means_not_mean_seed_ratio() -> None:
    point, low, high, undefined = relative_cost_bootstrap(
        [2.0] * 20,
        [0.0] + [1.0] * 19,
        seed=11,
        draws=2_000,
    )
    assert point == pytest.approx(21.0 / 19.0)
    assert np.isfinite([low, high]).all()
    assert high >= low
    assert undefined == 0


def test_registered_a2_decisions_use_all_bounds_and_integrity() -> None:
    favorable = {
        "completion_gain": {"ci_low": 0.002, "ci_high": 0.02},
        "completion_loss": {"ci_low": -0.02, "ci_high": -0.002},
        "slo_increase": {"ci_low": -0.04, "ci_high": 0.01},
        "ready_cost_difference": {"ci_low": -4.0, "ci_high": 8.0},
        "relative_cost_increase": {"ci_low": -0.05, "ci_high": 0.04},
    }
    decision = decide_claims(favorable, integrity_pass=True)
    assert decision["bounded_service_gain_pass"] is True
    assert decision["strict_pareto_pass"] is False
    assert decision["original_service_gain_pass"] is True
    assert decision["service_noninferiority_pass"] is True

    cost_crossing = {key: dict(value) for key, value in favorable.items()}
    cost_crossing["ready_cost_difference"]["ci_high"] = 10.0001
    assert decide_claims(cost_crossing, integrity_pass=True)[
        "bounded_service_gain_pass"
    ] is False
    assert decide_claims(favorable, integrity_pass=False)[
        "bounded_service_gain_pass"
    ] is False


def test_action_horizon_accepts_absent_optional_controller_steps_only() -> None:
    actions = [{"step": index} for index in range(32)]
    assert action_horizon_valid(actions, {"steps": 32}) is True
    assert action_horizon_valid(actions, {"steps": None}) is True
    assert action_horizon_valid(actions, {}) is True
    assert action_horizon_valid(actions[:-1], {}) is False
    assert action_horizon_valid(actions, {"steps": 31}) is False


def test_optional_mechanism_mean_is_missing_instead_of_nan() -> None:
    assert finite_mean_or_none([1.0, 3.0]) == pytest.approx(2.0)
    assert finite_mean_or_none([math.nan, math.nan]) is None


def test_request_line_counter_ignores_empty_lines(tmp_path) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text('{"id": 1}\n\n  \n{"id": 2}\n', encoding="utf-8")
    assert count_nonempty_lines(path) == 2
