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
from stage2_dynamic_budget.direct_action_planning_repair.aggregation import (
    aggregate_branch_labels,
    collect_closed_loop_branch_labels,
    visitation_distribution_distance,
)
from stage2_dynamic_budget.direct_action_planning_repair.data import (
    collect_common_random_branch_data,
)
from stage2_dynamic_budget.direct_action_planning_repair.model import (
    StructuredActionEffectModel,
)
from stage2_dynamic_budget.direct_action_planning_repair.planning import (
    StructuredPlanningAgent,
)


def test_closed_loop_aggregation_labels_all_feasible_actions_with_one_uniform() -> None:
    torch.set_num_threads(1)
    mdp = ActionConditionedBudgetMDP(
        ACBADPConfig(horizon=3, max_budget=3, max_queue=2, scenario="periodic")
    )
    optimum = solve_action_dp(mdp)
    old = fit_empirical_action_model(
        mdp, collect_transition_samples(mdp, 16, 4), smoothing=0.25
    )
    value = solve_empirical_value(mdp, old)
    base = collect_common_random_branch_data(
        mdp, value, optimum, "periodic", 50, train_samples=4, validation_samples=2
    )
    model = StructuredActionEffectModel(3, mdp.n_loads, mdp.n_actions, 8)
    agent = StructuredPlanningAgent(mdp, value, model)
    samples, visits = collect_closed_loop_branch_labels(
        agent,
        mdp,
        value,
        optimum,
        budgets=[2],
        scenario="periodic",
        seed=5,
        episodes=2,
        round_index=1,
    )
    assert samples.groupby("branch_group_id").transition_uniform.nunique().max() == 1
    assert samples.groupby("branch_group_id").true_next_load.nunique().max() == 1
    assert len(visits) == 6
    updated = aggregate_branch_labels(base.labels, base.samples, [samples])
    assert set(updated.aggregation_round) == {1}
    train = updated[updated.split == "train"]
    assert np.allclose(
        train[["next_load_prob_0", "next_load_prob_1", "next_load_prob_2"]].sum(axis=1),
        1.0,
    )
    distance = visitation_distribution_distance(base.samples, visits)
    assert 0.0 <= distance <= 1.0
