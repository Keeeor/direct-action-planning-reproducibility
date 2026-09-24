from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import solve_action_dp
from stage2_dynamic_budget.direct_action_planning_support_adaptation.data import (
    full_state_value_frame,
)
from stage2_dynamic_budget.direct_action_planning_support_adaptation.models import (
    FrozenValueWithAdapter,
    LinearValueAdapter,
    PooledValueNetwork,
)
from stage2_dynamic_budget.direct_action_planning_support_adaptation.scenario import (
    ContinuousScenario,
    build_continuous_mdp,
)
from stage2_dynamic_budget.direct_action_planning_support_adaptation.support import (
    compute_support_distances,
    support_relationship,
)
from stage2_dynamic_budget.direct_action_planning_support_adaptation.training import (
    AdapterTrainingConfig,
    PooledTrainingConfig,
    ValueLossWeights,
    dense_value_table,
    train_pooled_value,
    train_value_adapter,
)


def test_continuous_scenario_probabilities_are_valid_and_parameter_sensitive():
    first = ContinuousScenario("a", 1.5, 0.2, 0.3, 0.2, 0.3, 0.1)
    second = ContinuousScenario("b", 2.3, 0.7, 0.7, 0.3, 0.5, 0.4)
    mdp_a = build_continuous_mdp(first, horizon=8, max_budget=4, max_queue=3)
    mdp_b = build_continuous_mdp(second, horizon=8, max_budget=4, max_queue=3)
    pa = mdp_a.load_probabilities(2, 1)
    pb = mdp_b.load_probabilities(2, 1)
    np.testing.assert_allclose(pa.sum(), 1.0)
    np.testing.assert_allclose(pb.sum(), 1.0)
    assert np.all(pa > 0) and np.all(pb > 0)
    assert not np.allclose(pa, pb)


def test_adapter_freezes_base_and_is_low_capacity():
    base = PooledValueNetwork(horizon=8, n_loads=3, max_queue=4, max_budget=6, hidden_dim=12)
    adapter = LinearValueAdapter(input_dim=4)
    combined = FrozenValueWithAdapter(base, adapter)
    assert all(not parameter.requires_grad for parameter in combined.base.parameters())
    assert sum(parameter.numel() for parameter in combined.adapter.parameters()) <= 256
    state = torch.tensor([[0.5, 0.5, 0.25, 0.75]], dtype=torch.float64)
    assert torch.isfinite(combined(state)).all()


def test_support_distances_cover_all_registered_spaces():
    train = pd.DataFrame(
        {
            "raw_0": [0.0, 1.0], "raw_1": [0.0, 1.0],
            "structured_0": [0.0, 1.0], "structured_1": [0.0, 1.0],
            "hidden_0": [0.0, 1.0], "hidden_1": [0.0, 1.0],
            "q_vector_0": [0.0, 1.0], "q_vector_1": [0.0, 1.0],
        }
    )
    test = pd.DataFrame(
        {
            "raw_0": [0.0, 2.0], "raw_1": [0.0, 2.0],
            "structured_0": [0.0, 2.0], "structured_1": [0.0, 2.0],
            "hidden_0": [0.0, 2.0], "hidden_1": [0.0, 2.0],
            "q_vector_0": [0.0, 2.0], "q_vector_1": [0.0, 2.0],
            "Q_star_regret": [0.0, 1.0], "action_error": [0.0, 1.0],
        }
    )
    distances = compute_support_distances(train, test)
    assert {"raw", "structured", "hidden", "q_vector"}.issubset(distances.columns)
    assert distances.iloc[0]["raw"] == 0.0
    assert distances.iloc[1]["raw"] > 0.0
    relation = support_relationship(distances, spearman_floor=0.2, regret_share_floor=0.35)
    assert relation["deployable_support_clear"]


def test_pooled_training_reduces_validation_error_and_exports_terminal_zero():
    scenario = ContinuousScenario("tiny", 1.8, 0.3, 0.4, 0.2, 0.35, 0.2)
    mdp = build_continuous_mdp(scenario, horizon=4, max_budget=4, max_queue=3)
    optimum = solve_action_dp(mdp)
    frame = full_state_value_frame(mdp, optimum, scenario.scenario_id)
    train = frame[frame.t % 2 == 0].reset_index(drop=True)
    validation = frame[frame.t % 2 == 1].reset_index(drop=True)
    trained = train_pooled_value(
        train,
        validation,
        rank_frames={},
        anchor_states=train.head(20),
        model=PooledValueNetwork(
            horizon=4, n_loads=3, max_queue=3, max_budget=4, hidden_dim=12
        ),
        weights=ValueLossWeights(value=1.0),
        config=PooledTrainingConfig(
            learning_rate=0.02, max_epochs=80, patience=20, seed=3
        ),
    )
    assert trained.history.iloc[-1].validation_value_mae < trained.history.iloc[0].validation_value_mae
    table = dense_value_table(trained.model, mdp, source="test")
    assert table.values.shape == (5, 3, 4, 5)
    np.testing.assert_allclose(table.values[0], 0.0)


def test_target_adapter_reduces_prefix_mae_without_updating_base():
    scenario = ContinuousScenario("tiny", 2.0, 0.5, 0.5, 0.2, 0.4, 0.2)
    mdp = build_continuous_mdp(scenario, horizon=4, max_budget=4, max_queue=3)
    optimum = solve_action_dp(mdp)
    frame = full_state_value_frame(mdp, optimum, scenario.scenario_id).head(80)
    base = PooledValueNetwork(
        horizon=4, n_loads=3, max_queue=3, max_budget=4, hidden_dim=8
    )
    before = {name: value.detach().clone() for name, value in base.state_dict().items()}
    trained = train_value_adapter(
        base,
        calibration=frame,
        anchors=frame.head(10),
        rank_frame=None,
        variant="regression_anchor",
        config=AdapterTrainingConfig(
            learning_rate=0.05, max_epochs=80, patience=20, anchor_weight=0.1, seed=9
        ),
    )
    assert trained.history.iloc[-1].calibration_value_mae < trained.history.iloc[0].calibration_value_mae
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, before[name])
