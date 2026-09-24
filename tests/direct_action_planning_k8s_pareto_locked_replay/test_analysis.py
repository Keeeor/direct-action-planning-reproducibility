from __future__ import annotations

import pytest

from stage2_dynamic_budget.direct_action_planning_k8s_pareto_locked_replay.analysis import (
    decide_locked_replay,
    paired_rows,
)


ANALYSIS = {
    "completion_loss_margin": 0.01,
    "slo_increase_margin": 0.02,
    "ready_cost_increase_seconds_margin": 10.0,
}


def _statistics(
    *, completion_low: float = 0.001, slo_high: float = 0.01, cost_high: float = 9.0
) -> dict[str, dict[str, float]]:
    return {
        "completion_difference": {"ci_low": completion_low, "ci_high": 0.03},
        "slo_violation_difference": {"ci_low": -0.04, "ci_high": slo_high},
        "ready_cost_difference_seconds": {"ci_low": -2.0, "ci_high": cost_high},
    }


def test_locked_decision_requires_all_guards_and_one_strict_interval() -> None:
    result = decide_locked_replay(
        _statistics(), analysis_config=ANALYSIS, integrity_pass=True
    )
    assert result["all_primary_guards_pass"] is True
    assert result["strict_favorable_checks"] == {
        "completion_lcl_gt_zero": True,
        "slo_ucl_lt_zero": False,
        "ready_cost_ucl_lt_zero": False,
    }
    assert result["locked_replay_success"] is True

    no_strict = decide_locked_replay(
        _statistics(completion_low=0.0),
        analysis_config=ANALYSIS,
        integrity_pass=True,
    )
    assert no_strict["all_primary_guards_pass"] is True
    assert no_strict["at_least_one_strict_favorable_ci"] is False
    assert no_strict["locked_replay_success"] is False


@pytest.mark.parametrize(
    ("statistics", "integrity"),
    [
        (_statistics(completion_low=-0.010001), True),
        (_statistics(slo_high=0.020001), True),
        (_statistics(cost_high=10.000001), True),
        (_statistics(), False),
    ],
)
def test_locked_decision_cannot_trade_away_a_failed_guard(
    statistics: dict[str, dict[str, float]], integrity: bool
) -> None:
    result = decide_locked_replay(
        statistics, analysis_config=ANALYSIS, integrity_pass=integrity
    )
    assert result["locked_replay_success"] is False


def test_locked_pairs_require_identical_request_plan_hashes() -> None:
    rows = [
        {"method": "dap_cont_0p05", "seed": 7, "plan_sha256": "sha256:a"},
        {"method": "threshold", "seed": 7, "plan_sha256": "sha256:a"},
    ]
    assert paired_rows(rows, seeds=[7])[0][0] == 7
    rows[1]["plan_sha256"] = "sha256:b"
    with pytest.raises(ValueError, match="unpaired"):
        paired_rows(rows, seeds=[7])
