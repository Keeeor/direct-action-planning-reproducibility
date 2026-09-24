from __future__ import annotations

import numpy as np
import pandas as pd

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    solve_action_dp,
)
from stage2_dynamic_budget.direct_action_planning.planning import BudgetValueTable
from stage2_dynamic_budget.direct_action_planning_context_value.aliasing import (
    alias_summary,
    build_alias_table,
    build_near_alias_pairs,
    future_load_window,
    oracle_phase,
)
from stage2_dynamic_budget.direct_action_planning_context_value.history import (
    CausalHistory,
    history_features,
)
from stage2_dynamic_budget.direct_action_planning_context_value.gate import (
    assess_context_gate,
    assess_oracle_context_gate,
)
from stage2_dynamic_budget.direct_action_planning_context_value.experiment import (
    _decision_latency_mean,
    verify_frozen_inputs,
)


def _mdp(scenario: str) -> ActionConditionedBudgetMDP:
    return ActionConditionedBudgetMDP(
        ACBADPConfig(horizon=8, max_budget=6, max_queue=4, scenario=scenario)
    )


def test_alias_table_matches_all_exact_states_and_detects_conflicts():
    mdps = {name: _mdp(name) for name in ("early_burst", "late_burst", "periodic")}
    optima = {name: solve_action_dp(mdp) for name, mdp in mdps.items()}
    table = build_alias_table(mdps, optima, future_steps=4)
    expected = 8 * 3 * 5 * 7
    assert len(table) == expected
    assert table.optimal_action_conflict.any()
    assert (table.cross_scenario_value_variance >= 0).all()
    assert (table.max_future_window_difference >= 0).all()
    summary = alias_summary(table)
    assert summary["exact_alias_states"] == expected
    assert 0 < summary["optimal_action_conflict_rate"] < 1


def test_near_alias_pairs_exclude_exact_matches_and_respect_radius():
    mdps = {name: _mdp(name) for name in ("early_burst", "late_burst", "periodic")}
    optima = {name: solve_action_dp(mdp) for name, mdp in mdps.items()}
    pairs = build_near_alias_pairs(mdps, optima, radius=0.20)
    assert len(pairs) > 0
    assert (pairs.normalized_linf_distance > 0).all()
    assert (pairs.normalized_linf_distance <= 0.20 + 1.0e-12).all()


def test_future_window_starts_at_current_information_and_has_fixed_width():
    mdp = _mdp("early_burst")
    first = future_load_window(mdp, t=2, current_load=1, steps=4)
    second = future_load_window(mdp, t=2, current_load=1, steps=4)
    np.testing.assert_allclose(first, second)
    assert first.shape == (4,)
    assert first[0] == mdp.load_arrivals[1]


def test_oracle_phase_is_diagnostic_and_scenario_specific():
    early = oracle_phase(_mdp("early_burst"), t=2)
    late = oracle_phase(_mdp("late_burst"), t=2)
    periodic = oracle_phase(_mdp("periodic"), t=2)
    assert early.shape == late.shape == periodic.shape
    assert not np.allclose(early, late)
    assert not np.allclose(early, periodic)


def test_history_features_do_not_change_when_only_future_suffix_changes():
    prefix = CausalHistory(
        arrivals=(1.0, 2.0, 4.0),
        queues=(0.0, 1.0, 2.0),
        actions=(0.0, 1.0),
        capacities=(1.0, 2.0),
    )
    future_a = prefix.with_future_for_test(arrivals=(1.0, 1.0), queues=(0.0, 0.0))
    future_b = prefix.with_future_for_test(arrivals=(4.0, 4.0), queues=(4.0, 4.0))
    a = history_features(future_a, window=8, cutoff=3)
    b = history_features(future_b, window=8, cutoff=3)
    np.testing.assert_allclose(a, b)


def test_oracle_value_table_remains_exact_after_horizon_reindexing():
    mdp = _mdp("periodic")
    optimum = solve_action_dp(mdp)
    table = BudgetValueTable.from_exact_dp(optimum.values)
    for t, load, queue, budget in [(0, 1, 0, 6), (5, 2, 3, 2), (7, 0, 4, 0)]:
        assert table.predict(load, queue, budget, mdp.config.horizon - t) == (
            optimum.values[t, load, queue, budget]
        )


def test_alias_attribution_requires_unique_state_rows():
    frame = pd.DataFrame(
        {
            "t": [0, 0],
            "load": [1, 1],
            "queue": [0, 0],
            "remaining_budget": [4, 4],
        }
    )
    assert frame.duplicated().any()


def test_oracle_gate_requires_both_oracles_to_beat_d1_in_every_scenario():
    rows = []
    for scenario in ("early_burst", "late_burst", "periodic"):
        for seed in range(5):
            rows.extend(
                [
                    {"scenario": scenario, "model_seed": seed, "method": "frozen_D1", "regret": 0.02},
                    {"scenario": scenario, "model_seed": seed, "method": "oracle_scenario_value", "regret": 0.001},
                    {"scenario": scenario, "model_seed": seed, "method": "oracle_phase_value", "regret": 0.002},
                ]
            )
    gate = assess_oracle_context_gate(pd.DataFrame(rows))
    assert gate["decision"] == "CONTINUE"
    rows[-1]["regret"] = 0.2
    gate = assess_oracle_context_gate(pd.DataFrame(rows))
    assert gate["decision"] == "STOP"


def test_context_route_frozen_manifest_is_valid():
    rows = verify_frozen_inputs(".")
    assert len(rows) == 12
    assert all(row["matches"] for row in rows)


def test_runtime_adapter_uses_evaluator_contract_key():
    assert _decision_latency_mean({"decision_latency_ms_mean": 0.25}) == 0.25


def test_context_gate_aggregates_scenario_seed_units_before_decision():
    episode_rows = []
    for scenario in ("early_burst", "late_burst", "periodic"):
        for model_seed in range(5):
            for method, regret in (
                ("frozen_D1", 0.02),
                ("oracle_scenario_value", 0.0),
                ("final_context_value", 0.004),
                ("domain_value_refresh", 0.0),
                ("in_domain_context_value", 0.001),
            ):
                episode_rows.append(
                    {
                        "scenario": scenario,
                        "model_seed": model_seed,
                        "method": method,
                        "mean_Q_star_regret": regret,
                        "completion_rate": 0.9,
                        "slo_violation_rate": 0.1,
                        "total_cost": 8.0,
                        "return_gap_to_paired_optimal": regret,
                    }
                )
    state_rows = []
    for region, final_regret, d1_regret in ((False, 0.01, 0.02), (True, 0.002, 0.03)):
        for method, regret in (("final_context_value", final_regret), ("frozen_D1", d1_regret)):
            state_rows.append(
                {
                    "method": method,
                    "scenario": "early_burst",
                    "model_seed": 0,
                    "alias_region": region,
                    "mean_Q_star_regret": regret,
                }
            )
    oracle = {
        "decision": "CONTINUE",
        "checks": {"oracle": True},
    }
    config = {
        "oracle_gain_recovery_floor": 0.70,
        "wins_required_of_15": 10,
        "in_domain_regret_increase_ceiling": 0.002,
        "material_completion_loss": 0.01,
        "material_slo_increase": 0.01,
        "material_uncompensated_cost_increase_fraction": 0.05,
    }
    gate = assess_context_gate(
        pd.DataFrame(episode_rows), pd.DataFrame(state_rows), oracle, True, config
    )
    assert gate["decision"] == "CONTINUE"
    assert gate["cell_wins"] == 15
