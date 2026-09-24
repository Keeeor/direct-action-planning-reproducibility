from __future__ import annotations

import numpy as np
import torch

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    solve_action_dp,
)
from stage2_dynamic_budget.direct_action_planning.learning import (
    collect_transition_samples,
    fit_empirical_action_model,
    solve_empirical_value,
)
from stage2_dynamic_budget.direct_action_planning_repair.data import (
    attach_priority_weights,
    collect_common_random_branch_data,
)
from stage2_dynamic_budget.direct_action_planning_repair.model import (
    LossWeights,
    StructuredActionEffectModel,
    TrainingConfig,
    train_structured_model,
)
from stage2_dynamic_budget.direct_action_planning_repair.planning import (
    ensemble_one_step_plan,
    structured_one_step_plan,
)


def _fixture():
    mdp = ActionConditionedBudgetMDP(
        ACBADPConfig(horizon=3, max_budget=3, max_queue=2, scenario="early_burst")
    )
    optimum = solve_action_dp(mdp)
    old_samples = collect_transition_samples(mdp, 24, seed=5)
    old_model = fit_empirical_action_model(mdp, old_samples, smoothing=0.25)
    value = solve_empirical_value(mdp, old_model)
    return mdp, optimum, old_model, value


def test_common_random_branch_labels_share_exogenous_state_and_exact_known_effects() -> None:
    mdp, optimum, _, value = _fixture()
    data = collect_common_random_branch_data(
        mdp, value, optimum, "early_burst", seed=7, train_samples=4, validation_samples=2
    )
    group = data.samples.groupby("branch_group_id")
    assert group.true_next_load.nunique().max() == 1
    assert group.transition_uniform.nunique().max() == 1
    assert np.all(data.samples.action_effect_load == 0)
    expected_costs = data.samples.action.map(dict(enumerate(mdp.action_costs)))
    assert np.all(data.samples.cost == expected_costs)
    assert set(["q_lv", "q_star", "base_next_queue", "action_effect_queue"]).issubset(
        data.labels.columns
    )


def test_structured_planner_uses_exact_budget_queue_reward_and_masks_actions() -> None:
    mdp, _, _, value = _fixture()
    model = StructuredActionEffectModel(
        mdp.config.horizon, mdp.n_loads, mdp.n_actions, hidden_dim=8
    )
    for budget in range(mdp.config.max_budget + 1):
        plan = structured_one_step_plan(mdp, value, model, 1, 2, 1, budget)
        assert mdp.action_costs[plan.action] <= budget
        assert np.all(np.isnan(plan.q_values[mdp.action_costs > budget]))
        assert np.isclose(plan.next_load_probabilities[plan.action].sum(), 1.0)


def test_training_records_all_losses_and_uses_order_first_selection() -> None:
    torch.set_num_threads(1)
    mdp, optimum, old_model, value = _fixture()
    data = collect_common_random_branch_data(
        mdp, value, optimum, "early_burst", seed=9, train_samples=8, validation_samples=4
    )
    labels = attach_priority_weights(
        data.labels, mdp, old_model, value, optimum, "early_burst", 0.05, 6.0
    )
    trained = train_structured_model(
        labels,
        mdp.config.horizon,
        mdp.n_loads,
        mdp.n_actions,
        mdp.config.gamma,
        LossWeights(state=1.0, effect=2.0, q=4.0, rank=2.0),
        TrainingConfig(
            hidden_dim=8,
            max_epochs=8,
            patience=4,
            seed=2,
            use_priority_weights=True,
        ),
    )
    assert trained.best_epoch >= 0
    assert set(
        ["state_loss", "effect_loss", "q_loss", "rank_loss", "selection_score"]
    ).issubset(trained.history.columns)
    assert np.isfinite(list(trained.selection_metrics.values())).all()


def test_ensemble_reports_variance_disagreement_and_budget_safe_fallback() -> None:
    mdp, _, _, value = _fixture()
    models = [
        StructuredActionEffectModel(mdp.config.horizon, mdp.n_loads, mdp.n_actions, 8)
        for _ in range(3)
    ]
    plan = ensemble_one_step_plan(
        mdp,
        value,
        models,
        t=0,
        load=1,
        queue=0,
        budget=1,
        uncertainty_multiplier=1.0,
        minimum_margin=1.0e6,
        fallback_action=0,
    )
    assert plan.action == 0
    assert plan.fallback_used
    assert plan.model_q_variance is not None
    assert np.all(np.isnan(plan.q_values[mdp.action_costs > 1]))
