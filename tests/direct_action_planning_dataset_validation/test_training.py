from __future__ import annotations

import numpy as np

from stage2_dynamic_budget.direct_action_planning_dataset_validation.data import TraceDataset
from stage2_dynamic_budget.direct_action_planning_dataset_validation.experiment import (
    select_refresh_candidate,
)
from stage2_dynamic_budget.direct_action_planning_dataset_validation.training import (
    collect_branch_dataset,
    train_value_fvi,
)


def _dataset() -> TraceDataset:
    values = np.tile(np.array([2.0, 5.0, 8.0, 4.0]), 20)
    return TraceDataset(
        name="test",
        domains={
            "a": {"train": values, "validation": values, "test": values},
            "b": {"train": values[::-1], "validation": values[::-1], "test": values[::-1]},
        },
        split_contract="test",
    )


def test_branch_collection_evaluates_every_affordable_action() -> None:
    data = collect_branch_dataset(
        _dataset(),
        split="train",
        horizon=8,
        budget=4.0,
        episodes_per_domain=1,
        seed=7,
    )
    assert data.n_states == 16
    assert data.feasible[:, 0].all()
    assert np.isfinite(data.rewards[data.feasible]).all()
    np.testing.assert_allclose(data.next_observations[:, 0, -1], np.maximum(data.observations[:, -1] - 1 / 8, 0))


def test_small_fvi_run_is_finite() -> None:
    training = collect_branch_dataset(
        _dataset(), split="train", horizon=8, budget=8.0, episodes_per_domain=2, seed=1
    )
    validation = collect_branch_dataset(
        _dataset(), split="validation", horizon=8, budget=8.0, episodes_per_domain=1, seed=2
    )
    model, history = train_value_fvi(
        training,
        validation,
        seed=3,
        gamma=0.99,
        iterations=2,
        epochs_per_iteration=1,
        learning_rate=1.0e-3,
        hidden_dim=8,
    )
    assert len(history) == 2
    assert np.isfinite(model.predict(validation.observations)).all()


def test_refresh_selection_uses_return_and_service_guardrails() -> None:
    rows = [
        {
            "method": "structured_dap",
            "discounted_return": 1.0,
            "completion_ratio": 0.9,
            "slo_violation_rate": 0.1,
            "total_cost": 10.0,
        },
        {
            "method": "structured_dap_refresh",
            "discounted_return": 1.1,
            "completion_ratio": 0.9,
            "slo_violation_rate": 0.1,
            "total_cost": 10.4,
        },
    ]
    assert select_refresh_candidate(rows) == "refreshed_value"
    rows[1]["completion_ratio"] = 0.85
    assert select_refresh_candidate(rows) == "base_value"
