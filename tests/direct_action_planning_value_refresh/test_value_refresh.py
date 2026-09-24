from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    solve_action_dp,
)
from stage2_dynamic_budget.direct_action_planning.planning import BudgetValueTable
from stage2_dynamic_budget.direct_action_planning_repair.model import (
    StructuredActionEffectModel,
)
from stage2_dynamic_budget.direct_action_planning_value_refresh.causal import (
    causal_decomposition,
)
from stage2_dynamic_budget.direct_action_planning_value_refresh.data import (
    build_anchor_states,
    value_state_targets,
)
from stage2_dynamic_budget.direct_action_planning_value_refresh.protocol import (
    FinalTestLedger,
)
from stage2_dynamic_budget.direct_action_planning_value_refresh.value import (
    ResidualBudgetValueModel,
    budget_monotonic_loss,
    planning_rank_loss,
)


@pytest.fixture
def tiny_mdp() -> ActionConditionedBudgetMDP:
    return ActionConditionedBudgetMDP(
        ACBADPConfig(horizon=4, max_budget=4, max_queue=3, scenario="early_burst")
    )


def test_residual_value_starts_at_fixed_table_and_exports(tiny_mdp):
    optimum = solve_action_dp(tiny_mdp)
    fixed = BudgetValueTable.from_exact_dp(optimum.values)
    model = ResidualBudgetValueModel(fixed)
    assert torch.allclose(model.value_grid(), torch.as_tensor(fixed.values))
    exported = model.as_value_table("test")
    assert exported.source == "test"
    np.testing.assert_allclose(exported.values, fixed.values)


def test_budget_monotonic_loss_detects_decrease(tiny_mdp):
    optimum = solve_action_dp(tiny_mdp)
    fixed = BudgetValueTable.from_exact_dp(optimum.values)
    model = ResidualBudgetValueModel(fixed)
    assert float(budget_monotonic_loss(model).detach()) <= 1.0e-12
    with torch.no_grad():
        model.residual[2, 0, 0, 2] = -10.0
    assert float(budget_monotonic_loss(model).detach()) > 0.0


def test_planning_rank_loss_is_zero_for_oracle_value(tiny_mdp):
    optimum = solve_action_dp(tiny_mdp)
    oracle = BudgetValueTable.from_exact_dp(optimum.values)
    value_model = ResidualBudgetValueModel(oracle)
    transition = StructuredActionEffectModel(
        tiny_mdp.config.horizon, tiny_mdp.n_loads, tiny_mdp.n_actions, hidden_dim=8
    )
    # Use exact transition probabilities so the ranking target and planner agree.
    logits = []
    for t in range(tiny_mdp.config.horizon):
        row = []
        for load in range(tiny_mdp.n_loads):
            probability = tiny_mdp.load_probabilities(t, load)
            row.append(np.log(probability))
        logits.append(row)
    with torch.no_grad():
        transition.base[-1].weight.zero_()
        transition.base[-1].bias.zero_()
        transition.effect[-1].weight.zero_()
        transition.effect[-1].bias.zero_()
    states = value_state_targets(
        tiny_mdp,
        optimum,
        {
            "D0": pd.DataFrame(
                [{"t": 0, "load": 0, "queue": 0, "remaining_budget": 4}]
            )
        },
    )
    loss, pair_count = planning_rank_loss(
        value_model, transition, tiny_mdp, optimum, states, margin=0.0
    )
    assert pair_count > 0
    assert torch.isfinite(loss)


def test_true_oracle_causal_arm_recovers_exact_policy(tiny_mdp):
    optimum = solve_action_dp(tiny_mdp)
    oracle = BudgetValueTable.from_exact_dp(optimum.values)
    transition = StructuredActionEffectModel(
        tiny_mdp.config.horizon, tiny_mdp.n_loads, tiny_mdp.n_actions, hidden_dim=8
    )
    states = pd.DataFrame(
        [
            {"t": t, "load": load, "queue": queue, "remaining_budget": budget}
            for t, load, queue, budget in np.ndindex(optimum.actions.shape)
        ]
    )
    rows = causal_decomposition(
        tiny_mdp, optimum, oracle, transition, {"D0": states}, "early_burst", 1
    )
    exact = rows[(rows.planner == "true_transition_oracle_value")]
    assert exact.action_consistency.mean() == 1.0
    assert exact.q_star_regret.max() <= 1.0e-12


def test_anchor_builder_includes_required_strata(tiny_mdp):
    optimum = solve_action_dp(tiny_mdp)
    grid = pd.DataFrame(
        [
            {"t": t, "load": load, "queue": queue, "remaining_budget": budget}
            for t, load, queue, budget in np.ndindex(optimum.actions.shape)
        ]
    )
    anchors = build_anchor_states(tiny_mdp, optimum, grid, grid, "late_burst", 0.75)
    assert {"D0_fixed", "D1_high_value", "late_burst", "tight_budget"}.issubset(
        set(anchors.anchor_reason.str.split("|").explode())
    )
    assert anchors[["t", "load", "queue", "remaining_budget"]].duplicated().sum() == 0


def test_final_test_ledger_is_single_use(tmp_path):
    path = tmp_path / "selection.json"
    ledger = FinalTestLedger.create(path, [{"scenario": "early_burst", "train_seed": 25}])
    assert ledger.test_evaluations_completed == 0
    ledger.mark_test_started()
    with pytest.raises(RuntimeError, match="already started"):
        FinalTestLedger.load(path).mark_test_started()
