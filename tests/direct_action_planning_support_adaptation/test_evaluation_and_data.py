from __future__ import annotations

import numpy as np

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import solve_action_dp
from stage2_dynamic_budget.action_conditioned_budget_advantage.evaluation import ExactDPAgent
from stage2_dynamic_budget.direct_action_planning_support_adaptation.data import (
    collect_value_trajectories,
)
from stage2_dynamic_budget.direct_action_planning_support_adaptation.evaluation import (
    evaluate_rollouts,
    make_branchable_env,
)
from stage2_dynamic_budget.direct_action_planning_support_adaptation.models import (
    PooledValueNetwork,
)
from stage2_dynamic_budget.direct_action_planning_support_adaptation.scenario import (
    ContinuousScenario,
    build_continuous_mdp,
)
from stage2_dynamic_budget.direct_action_planning_support_adaptation.support import (
    build_support_features,
)


def _tiny():
    scenario = ContinuousScenario("tiny", 1.8, 0.25, 0.5, 0.2, 0.35, 0.2)
    mdp = build_continuous_mdp(scenario, horizon=5, max_budget=4, max_queue=3)
    return scenario, mdp, solve_action_dp(mdp)


def test_branchable_adapter_preserves_continuous_transition():
    _, mdp, _ = _tiny()
    env = make_branchable_env(mdp, initial_budget=4)
    env.reset(seed=3)
    np.testing.assert_allclose(
        env.mdp.load_probabilities(1, 1), mdp.load_probabilities(1, 1)
    )
    assert env.mdp is mdp


def test_trajectory_collection_has_unique_causal_rows_and_exact_targets():
    scenario, mdp, optimum = _tiny()
    frame = collect_value_trajectories(
        mdp,
        optimum,
        scenario_id=scenario.scenario_id,
        region="train_interpolation",
        budgets=[2, 4],
        seed=11,
        episodes=4,
    )
    assert len(frame) == 20
    assert frame.row_id.is_unique
    assert (frame.remaining_horizon == mdp.config.horizon - frame.t).all()
    first = frame.iloc[0]
    state = (int(first.t), int(first.load), int(first.queue), int(first.remaining_budget))
    assert first.target_value == optimum.values[state]


def test_exact_dp_has_zero_suffix_regret_in_continuous_environment():
    scenario, mdp, optimum = _tiny()
    episodes, steps, _ = evaluate_rollouts(
        ExactDPAgent(mdp, optimum),
        "exact_dp",
        mdp,
        optimum,
        budgets=[2, 4],
        scenario_id=scenario.scenario_id,
        seed=17,
        episodes=3,
        metric_start_t=1,
    )
    assert episodes.mean_Q_star_regret.max() <= 1.0e-12
    assert steps.Q_star_regret.max() <= 1.0e-12


def test_support_features_exclude_scenario_parameters_and_future_values():
    scenario, mdp, optimum = _tiny()
    frame = collect_value_trajectories(
        mdp, optimum, scenario_id=scenario.scenario_id, region="train", budgets=[4], seed=5, episodes=2
    )
    model = PooledValueNetwork(
        horizon=5, n_loads=3, max_queue=3, max_budget=4, hidden_dim=8
    )
    features = build_support_features(frame, mdp, optimum, model)
    assert any(column.startswith("raw_") for column in features)
    assert any(column.startswith("structured_") for column in features)
    assert any(column.startswith("hidden_") for column in features)
    assert any(column.startswith("q_") for column in features)
    forbidden = {
        "base_load", "burst_start", "burst_amplitude", "burst_duration",
        "period", "periodic_amplitude", "future_load",
    }
    assert forbidden.isdisjoint(features.columns)
