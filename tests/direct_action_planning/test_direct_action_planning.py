from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dap.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    solve_action_dp,
)
from dap.direct_action_planning.learning import (
    collect_transition_samples,
    fit_empirical_action_model,
    solve_empirical_value,
)
from dap.direct_action_planning.planning import (
    BudgetValueTable,
    DirectPlanningAgent,
    one_step_plan,
)
from dap.direct_action_planning.gate import assess_continuation
from dap.direct_action_planning.analysis import (
    _evidence_grade,
    paired_comparisons,
)


def test_oracle_one_step_plan_matches_every_exact_q_value() -> None:
    cfg = ACBADPConfig(
        horizon=4, max_budget=3, max_queue=2, scenario="late_burst", gamma=0.93
    )
    mdp = ActionConditionedBudgetMDP(cfg)
    optimum = solve_action_dp(mdp)
    value = BudgetValueTable.from_exact_dp(optimum.values)

    for t, load, queue, budget in np.ndindex(optimum.actions.shape):
        plan = one_step_plan(mdp, value, t, load, queue, budget)
        assert np.allclose(
            plan.q_values,
            optimum.q_values[t, load, queue, budget],
            equal_nan=True,
            rtol=1e-12,
            atol=1e-12,
        )
        assert plan.action == int(optimum.actions[t, load, queue, budget])


def test_empirical_value_uses_budget_and_remaining_horizon() -> None:
    cfg = ACBADPConfig(
        horizon=3, max_budget=2, max_queue=2, scenario="stable", gamma=0.9
    )
    mdp = ActionConditionedBudgetMDP(cfg)
    samples = collect_transition_samples(mdp, samples_per_state_action=32, seed=7)
    model = fit_empirical_action_model(mdp, samples, smoothing=0.25)
    value = solve_empirical_value(mdp, model)

    assert value.values.shape == (4, 3, 3, 3)
    assert value.predict(load=1, queue=1, budget=0, remaining_horizon=0) == 0.0
    assert np.isfinite(value.predict(load=1, queue=1, budget=2, remaining_horizon=3))
    assert not np.array_equal(value.values[1], value.values[3])
    assert np.all(value.values[:, :, :, 1:] + 1e-12 >= value.values[:, :, :, :-1])


def test_empirical_model_predictions_are_normalized_and_budget_safe() -> None:
    cfg = ACBADPConfig(
        horizon=4, max_budget=3, max_queue=2, scenario="periodic", gamma=0.95
    )
    mdp = ActionConditionedBudgetMDP(cfg)
    samples = collect_transition_samples(mdp, samples_per_state_action=16, seed=13)
    model = fit_empirical_action_model(mdp, samples, smoothing=0.5)
    value = solve_empirical_value(mdp, model)

    prediction = model.predict(t=1, load=2, queue=1, action=3)
    assert np.isclose(prediction.next_state_probabilities.sum(), 1.0)
    assert np.isfinite(prediction.reward)
    assert np.isfinite(prediction.cost)

    for budget in range(cfg.max_budget + 1):
        plan = one_step_plan(
            mdp, value, t=1, load=2, queue=1, budget=budget, learned_model=model
        )
        assert mdp.action_costs[plan.action] <= budget
        assert np.isfinite(plan.q_values[plan.action])


def test_transition_sampling_is_seed_reproducible() -> None:
    cfg = ACBADPConfig(horizon=3, max_budget=2, max_queue=2, scenario="early_burst")
    mdp = ActionConditionedBudgetMDP(cfg)
    first = collect_transition_samples(mdp, samples_per_state_action=8, seed=101)
    second = collect_transition_samples(mdp, samples_per_state_action=8, seed=101)
    assert first.equals(second)


def test_direct_planning_agent_decodes_canonical_observation() -> None:
    cfg = ACBADPConfig(horizon=4, max_budget=3, max_queue=2, scenario="early_burst")
    mdp = ActionConditionedBudgetMDP(cfg)
    optimum = solve_action_dp(mdp)
    agent = DirectPlanningAgent(mdp, BudgetValueTable.from_exact_dp(optimum.values))
    observation = torch.tensor(
        [[2.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 2 / 3, 3 / 4]],
        dtype=torch.float32,
    )
    output = agent.act(observation, deterministic=True)
    expected = optimum.actions[1, 1, 1, 2]
    assert int(output.action.item()) == int(expected)
    assert output.q_values.shape == (1, 4)


def test_continuation_gate_requires_both_burst_scenarios_and_model_fidelity() -> None:
    rows = []
    methods = ["dsp_b", "oracle_branch", "learned_value_branch", "learned_model_branch"]
    for method in methods:
        for scenario in ("early_burst", "late_burst", "periodic"):
            for budget in (4, 8, 12):
                for seed in range(5):
                    agreement = 0.90
                    regret = 0.20
                    if method == "oracle_branch":
                        agreement, regret = 1.0, 0.0
                    elif method == "learned_value_branch":
                        agreement, regret = 0.94, 0.14
                    elif method == "learned_model_branch":
                        agreement, regret = 0.92, 0.17
                    rows.append(
                        {
                            "method": method,
                            "scenario": scenario,
                            "budget": budget,
                            "seed": seed,
                            "action_consistency_rate": agreement,
                            "mean_Q_star_regret": regret,
                            "completion_rate": 0.90,
                            "slo_violation_rate": 0.10,
                            "total_cost": 8.0,
                        }
                    )
    episodes = pd.DataFrame(rows)
    state = pd.DataFrame(
        [
            {"method": "oracle_branch", "action_consistency_rate": 1.0, "mean_Q_star_regret": 0.0}
        ]
    )
    gate = assess_continuation(episodes, state, {})
    assert gate["decision"] == "CONTINUE"

    late = (episodes.method == "learned_value_branch") & (episodes.scenario == "late_burst")
    episodes.loc[late, "action_consistency_rate"] = 0.80
    episodes.loc[late, "mean_Q_star_regret"] = 0.30
    stopped = assess_continuation(episodes, state, {})
    assert stopped["decision"] == "STOP"
    assert not stopped["checks"]["late_burst_regret_improves"]


def test_continuation_gate_handles_a_smoke_matrix_without_late_burst() -> None:
    rows = []
    for method in ("dsp_b", "oracle_branch", "learned_value_branch", "learned_model_branch"):
        rows.append(
            {
                "method": method,
                "scenario": "early_burst",
                "budget": 4,
                "seed": 0,
                "action_consistency_rate": 1.0 if method == "oracle_branch" else 0.9,
                "mean_Q_star_regret": 0.0 if method == "oracle_branch" else 0.2,
                "completion_rate": 0.9,
                "slo_violation_rate": 0.1,
                "total_cost": 4.0,
            }
        )
    state = pd.DataFrame(
        [{"method": "oracle_branch", "action_consistency_rate": 1.0, "mean_Q_star_regret": 0.0}]
    )
    gate = assess_continuation(pd.DataFrame(rows), state, {})
    assert gate["decision"] == "STOP"
    assert gate["learned_value_vs_dsp_b"]["burst_seed_regret_wins"]["late_burst"] == 0


def test_pooled_analysis_uses_frozen_budget_seed_units() -> None:
    rows = []
    methods = (
        "optimal",
        "b4_budget_state",
        "dsp_b",
        "acba_a",
        "oracle_branch",
        "learned_value_branch",
        "learned_model_branch",
    )
    for method in methods:
        for scenario in ("early_burst", "late_burst", "periodic"):
            for budget in (4, 8, 12):
                for seed in range(5):
                    rows.append(
                        {
                            "method": method,
                            "scenario": scenario,
                            "budget": budget,
                            "seed": seed,
                            "action_consistency_rate": 0.9,
                            "mean_Q_star_regret": 0.1,
                            "return_gap_to_paired_optimal": 1.0,
                            "high_risk_low_cost_balanced_accuracy": 0.8,
                            "budget_trajectory_mae": 0.2,
                            "completion_rate": 0.7,
                            "slo_violation_rate": 0.3,
                            "total_cost": float(budget),
                        }
                    )
    comparisons = paired_comparisons(pd.DataFrame(rows))
    assert set(comparisons.paired_cells) == {15}


def test_small_paired_sample_caps_evidence_at_weak() -> None:
    row = pd.Series(
        {
            "bootstrap_95ci_lower": 0.1,
            "bootstrap_95ci_upper": 0.3,
            "bh_q": 0.001,
            "paired_cohens_dz": 1.2,
            "paired_cells": 15,
        }
    )
    assert _evidence_grade(row) == "weak"
