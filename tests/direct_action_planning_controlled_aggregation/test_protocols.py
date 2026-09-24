from __future__ import annotations

import numpy as np
import pandas as pd
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
from stage2_dynamic_budget.direct_action_planning.planning import DirectPlanningAgent
from stage2_dynamic_budget.direct_action_planning_controlled_aggregation.collection import (
    collect_mixed_policy_branches,
    selective_samples,
)
from stage2_dynamic_budget.direct_action_planning_controlled_aggregation.replay import (
    balanced_replay_labels,
)
from stage2_dynamic_budget.direct_action_planning_repair.data import (
    attach_priority_weights,
    collect_common_random_branch_data,
)
from stage2_dynamic_budget.direct_action_planning_repair.model import (
    LossWeights,
    StructuredActionEffectModel,
    TrainingConfig,
)
from stage2_dynamic_budget.direct_action_planning_repair.planning import (
    StructuredPlanningAgent,
)
from stage2_dynamic_budget.direct_action_planning_controlled_aggregation.training import (
    train_controlled_model,
)


def _fixture():
    mdp = ActionConditionedBudgetMDP(
        ACBADPConfig(horizon=3, max_budget=3, max_queue=2, scenario="late_burst")
    )
    optimum = solve_action_dp(mdp)
    old = fit_empirical_action_model(
        mdp, collect_transition_samples(mdp, 12, seed=4), smoothing=0.25
    )
    value = solve_empirical_value(mdp, old)
    data = collect_common_random_branch_data(
        mdp, value, optimum, "late_burst", seed=7, train_samples=4, validation_samples=2
    )
    labels = attach_priority_weights(data.labels, mdp, old, value, optimum, "late_burst", 0.05, 6.0)
    return mdp, optimum, old, value, data, labels


def test_balanced_replay_keeps_full_d0_support_and_normalized_targets() -> None:
    mdp, optimum, old, value, data, labels = _fixture()
    model = StructuredActionEffectModel(3, mdp.n_loads, mdp.n_actions, 8)
    repair = StructuredPlanningAgent(mdp, value, model)
    reference = DirectPlanningAgent(mdp, value, learned_model=old)
    samples, _ = collect_mixed_policy_branches(
        repair, reference, mdp, value, optimum, [2], "late_burst", 9, 2, 1, 0.5
    )
    selected = selective_samples(samples)
    result = balanced_replay_labels(
        labels,
        [],
        selected,
        selected[selected.high_regret | selected.ranking_error],
        {"D0": 0.4, "historical_best": 0.3, "current": 0.2, "hard": 0.1},
    )
    train = result.labels[result.labels.split == "train"]
    assert len(train) == len(labels[labels.split == "train"])
    assert np.allclose(
        train[["next_load_prob_0", "next_load_prob_1", "next_load_prob_2"]].sum(axis=1),
        1.0,
    )
    assert set(result.source_mix.source) == {"D0", "historical_best", "current", "hard"}
    assert np.isclose(result.source_mix.realized_full_grid_loss_mass.sum(), 1.0)
    assert np.isclose(result.state_targets.state_loss_weight.sum(), 1.0)


def test_collection_uses_one_transition_tape_for_all_actions_and_reports_mix() -> None:
    mdp, optimum, old, value, _, _ = _fixture()
    model = StructuredActionEffectModel(3, mdp.n_loads, mdp.n_actions, 8)
    samples, visits = collect_mixed_policy_branches(
        StructuredPlanningAgent(mdp, value, model),
        DirectPlanningAgent(mdp, value, learned_model=old),
        mdp,
        value,
        optimum,
        [2],
        "late_burst",
        11,
        8,
        1,
        0.5,
    )
    grouped = samples.groupby("branch_group_id")
    assert grouped.transition_uniform.nunique().max() == 1
    assert grouped.true_next_load.nunique().max() == 1
    assert set(visits.collector_component) == {"repair", "reference"}
    assert 0 < len(selective_samples(samples)) <= len(samples)


def test_retention_training_records_constraint_without_changing_architecture() -> None:
    torch.set_num_threads(1)
    mdp, _, _, _, _, labels = _fixture()
    reference = StructuredActionEffectModel(3, mdp.n_loads, mdp.n_actions, 8)
    result = train_controlled_model(
        labels,
        3,
        mdp.n_loads,
        mdp.n_actions,
        mdp.config.gamma,
        LossWeights(1.0, 2.0, 4.0, 2.0, 0.02),
        TrainingConfig(hidden_dim=8, max_epochs=4, patience=2, seed=3, use_priority_weights=True),
        retention_reference=reference,
        retention_anchor=labels[labels.split == "validation"],
        retention_weight=1.0,
    )
    assert "retention_loss" in result.history
    assert result.selection_metrics["retention_enabled"] == 1.0
    assert result.model.base[0].out_features == reference.base[0].out_features
