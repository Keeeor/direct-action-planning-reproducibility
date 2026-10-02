from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from dap.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    solve_action_dp,
)
from dap.direct_action_planning.planning import BudgetValueTable
from dap.direct_action_planning_repair.model import StructuredActionEffectModel
from dap.direct_action_planning_context_value.history import (
    CausalHistory,
    leakage_audit,
    serialize_history,
)
from dap.direct_action_planning_context_value.model import (
    ContextResidualValueModel,
    ContextValuePredictor,
)
from dap.direct_action_planning_context_value.planning import (
    ContextPlanningAgent,
    context_one_step_plan,
)
from dap.direct_action_planning_context_value.protocol import FinalTestLedger
from dap.direct_action_planning_context_value.data import (
    collect_context_trajectories,
)
from dap.direct_action_planning_context_value.training import (
    ContextLossWeights,
    ContextTrainingConfig,
    train_context_value,
)
from dap.direct_action_planning_context_value.evaluation import (
    evaluate_context_state_grid,
)


@pytest.fixture
def tiny_mdp() -> ActionConditionedBudgetMDP:
    return ActionConditionedBudgetMDP(
        ACBADPConfig(horizon=4, max_budget=4, max_queue=3, scenario="early_burst")
    )


def test_causal_history_advances_without_future_suffix():
    history = CausalHistory(arrivals=(1.0,), queues=(0.0,))
    advanced = history.advance(
        action=1, capacity=2, next_arrival=4.0, next_queue=1.0
    )
    assert advanced.arrivals == (1.0, 4.0)
    assert advanced.queues == (0.0, 1.0)
    assert advanced.actions == (1.0,)
    assert advanced.capacities == (2.0,)


@pytest.mark.parametrize("mode", ["current", "feature", "gru"])
def test_context_value_model_has_finite_batched_output(mode: str):
    model = ContextResidualValueModel(
        mode=mode,
        horizon=4,
        n_loads=3,
        max_queue=3,
        max_budget=4,
        feature_dim=24,
        hidden_dim=8,
        gru_hidden_dim=6,
    )
    base = torch.tensor([2.0, 3.0], dtype=torch.float64)
    kwargs = {}
    if mode == "feature":
        kwargs["features"] = torch.zeros((2, 24), dtype=torch.float64)
    if mode == "gru":
        kwargs["sequence"] = torch.zeros((2, 4, 4), dtype=torch.float64)
        kwargs["mask"] = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 1]], dtype=torch.bool)
    output = model(
        torch.tensor([4, 3]),
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        torch.tensor([4, 2]),
        base,
        **kwargs,
    )
    assert output.shape == (2,)
    assert torch.isfinite(output).all()


def test_context_planner_masks_actions_above_true_budget(tiny_mdp):
    optimum = solve_action_dp(tiny_mdp)
    fixed = BudgetValueTable.from_exact_dp(optimum.values)
    model = ContextResidualValueModel(
        mode="feature",
        horizon=4,
        n_loads=3,
        max_queue=3,
        max_budget=4,
        feature_dim=24,
        hidden_dim=8,
        gru_hidden_dim=6,
    )
    predictor = ContextValuePredictor(model, fixed, window=4)
    transition = StructuredActionEffectModel(4, 3, 4, hidden_dim=8)
    history = CausalHistory(arrivals=(2.0,), queues=(0.0,))
    plan = context_one_step_plan(
        tiny_mdp, predictor, transition, history, t=0, load=1, queue=0, budget=1
    )
    assert np.isfinite(plan.q_values[:2]).all()
    assert np.isnan(plan.q_values[2:]).all()
    assert plan.action in (0, 1)


def test_batched_context_prediction_matches_scalar_calls(tiny_mdp):
    optimum = solve_action_dp(tiny_mdp)
    fixed = BudgetValueTable.from_exact_dp(optimum.values)
    model = ContextResidualValueModel(
        mode="feature",
        horizon=4,
        n_loads=3,
        max_queue=3,
        max_budget=4,
        feature_dim=24,
        hidden_dim=8,
        gru_hidden_dim=6,
    )
    predictor = ContextValuePredictor(model, fixed, window=4)
    histories = [
        CausalHistory((1.0,), (0.0,)),
        CausalHistory((1.0, 2.0), (0.0, 1.0), (0.0,), (1.0,)),
    ]
    batched = predictor.predict_many(
        loads=np.array([0, 1]),
        queues=np.array([0, 1]),
        budgets=np.array([4, 3]),
        remaining_horizons=np.array([4, 3]),
        histories=histories,
    )
    scalar = np.array(
        [
            predictor.predict(0, 0, 4, 4, histories[0]),
            predictor.predict(1, 1, 3, 3, histories[1]),
        ]
    )
    np.testing.assert_allclose(batched, scalar, rtol=1e-12, atol=1e-12)


def test_context_agent_advances_and_resets_history(tiny_mdp):
    optimum = solve_action_dp(tiny_mdp)
    fixed = BudgetValueTable.from_exact_dp(optimum.values)
    model = ContextResidualValueModel(
        mode="current",
        horizon=4,
        n_loads=3,
        max_queue=3,
        max_budget=4,
        feature_dim=24,
        hidden_dim=8,
        gru_hidden_dim=6,
    )
    transition = StructuredActionEffectModel(4, 3, 4, hidden_dim=8)
    agent = ContextPlanningAgent(
        tiny_mdp, ContextValuePredictor(model, fixed, 4), transition
    )
    from dap.action_conditioned_budget_advantage.branching import (
        BranchableDiscreteEnv,
    )

    env = BranchableDiscreteEnv(tiny_mdp.config, 4, budget_scale=4)
    observation, _ = env.reset(seed=9)
    first = agent.act(torch.as_tensor(observation).float().unsqueeze(0))
    observation, _, _, _, _ = env.step_with_uniform(int(first.action.item()), 0.5)
    agent.act(torch.as_tensor(observation).float().unsqueeze(0))
    assert agent.history is not None
    assert len(agent.history.arrivals) == 2
    assert len(agent.history.actions) == 1
    agent.reset_budget_controller()
    assert agent.history is None


def test_history_serialization_and_leakage_audit_are_causal():
    history = CausalHistory(
        arrivals=(1.0, 2.0, 4.0),
        queues=(0.0, 1.0, 2.0),
        actions=(0.0, 1.0),
        capacities=(1.0, 2.0),
    )
    row = {"t": 2, **serialize_history(history)}
    report = leakage_audit(pd.DataFrame([row]))
    assert report["passed"] is True
    bad = dict(row)
    bad["arrivals_history"] += "|4"
    report = leakage_audit(pd.DataFrame([bad]))
    assert report["passed"] is False


def test_final_test_ledger_opens_exactly_once(tmp_path):
    path = tmp_path / "selection.json"
    ledger = FinalTestLedger.create(
        path,
        selection={"family": "history_feature", "window": 8},
        validation_sha256="a" * 64,
    )
    ledger.mark_test_started()
    with pytest.raises(RuntimeError, match="already"):
        FinalTestLedger.load(path).mark_test_started()
    ledger.mark_test_completed()
    loaded = FinalTestLedger.load(path)
    assert loaded.test_evaluations_completed == 1
    assert loaded.test_status == "completed"


def test_context_collection_saves_all_sources_without_future_rows(tiny_mdp):
    optimum = solve_action_dp(tiny_mdp)
    fixed = BudgetValueTable.from_exact_dp(optimum.values)
    frame = collect_context_trajectories(
        tiny_mdp,
        optimum,
        fixed,
        policies={"D0": None, "D1": None},
        budgets=[2, 4],
        seed=11,
        episodes_per_source=2,
        device=torch.device("cpu"),
    )
    assert set(frame.source) == {"D0", "D1"}
    assert len(frame) == 2 * 2 * tiny_mdp.config.horizon
    assert leakage_audit(frame)["passed"] is True
    assert np.isfinite(frame.target_residual).all()


def test_tiny_context_training_reduces_validation_value_error(tiny_mdp):
    optimum = solve_action_dp(tiny_mdp)
    fixed = BudgetValueTable.from_exact_dp(optimum.values)
    train = collect_context_trajectories(
        tiny_mdp,
        optimum,
        fixed,
        policies={"D0": None, "D1": None},
        budgets=[2, 4],
        seed=21,
        episodes_per_source=4,
        device=torch.device("cpu"),
    )
    validation = collect_context_trajectories(
        tiny_mdp,
        optimum,
        fixed,
        policies={"D0": None},
        budgets=[2, 4],
        seed=22,
        episodes_per_source=2,
        device=torch.device("cpu"),
    )
    transition = StructuredActionEffectModel(4, 3, 4, hidden_dim=8)
    result = train_context_value(
        mode="feature",
        window=4,
        training=train,
        validation=validation,
        mdps={"early_burst": tiny_mdp},
        optima={"early_burst": optimum},
        base_tables={"early_burst": fixed},
        transitions={"early_burst": transition},
        loss_weights=ContextLossWeights(1.0, 0.1, 0.2, 1.0),
        config=ContextTrainingConfig(
            hidden_dim=8,
            gru_hidden_dim=6,
            learning_rate=0.01,
            max_epochs=20,
            patience=6,
            batch_size=16,
            rank_states=8,
            validation_rank_states=12,
            validation_interval=2,
            seed=3,
        ),
        source_weights={"D0": 0.5, "D1": 0.5},
        device=torch.device("cpu"),
    )
    assert np.isfinite(result.selection_metrics["value_mae"])
    assert result.selection_metrics["value_mae"] <= result.initial_metrics["value_mae"] + 1e-9
    assert {"value_loss", "anchor_loss", "mono_loss", "rank_loss"}.issubset(
        result.history.columns
    )


def test_context_state_grid_reports_alias_and_pair_metrics(tiny_mdp):
    optimum = solve_action_dp(tiny_mdp)
    fixed = BudgetValueTable.from_exact_dp(optimum.values)
    contexts = collect_context_trajectories(
        tiny_mdp,
        optimum,
        fixed,
        policies={"D0": None},
        budgets=[2, 4],
        seed=30,
        episodes_per_source=2,
        device=torch.device("cpu"),
    )
    model = ContextResidualValueModel(
        mode="current",
        horizon=4,
        n_loads=3,
        max_queue=3,
        max_budget=4,
        feature_dim=24,
        hidden_dim=8,
        gru_hidden_dim=6,
    )
    transition = StructuredActionEffectModel(4, 3, 4, hidden_dim=8)
    aliases = pd.DataFrame(
        [
            {
                "t": t,
                "load": load,
                "queue": queue,
                "remaining_budget": budget,
                "material_action_conflict": bool((t + load + queue + budget) % 2),
            }
            for t, load, queue, budget in np.ndindex(optimum.actions.shape)
        ]
    )
    states, summary, pairs = evaluate_context_state_grid(
        ContextValuePredictor(model, fixed, 4),
        transition,
        tiny_mdp,
        optimum,
        contexts,
        aliases,
        method="test",
        model_seed=1,
    )
    assert len(states) == np.prod(optimum.actions.shape)
    assert set(summary.alias_region) == {False, True, "all"}
    assert len(pairs) > 0
    assert pairs.pair_ranking_accuracy.between(0, 1).all()
