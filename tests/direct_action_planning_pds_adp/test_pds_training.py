from __future__ import annotations

import numpy as np

from stage2_dynamic_budget.direct_action_planning_dataset_validation.training import (
    BranchDataset,
)
from stage2_dynamic_budget.direct_action_planning_pds_adp.training import (
    build_postdecision_regression,
    train_postdecision_candidates,
)


class SumValue:
    def predict(self, observations: np.ndarray) -> np.ndarray:
        return np.asarray(observations, dtype=np.float64).sum(axis=1)


def branch_fixture() -> BranchDataset:
    observations = np.arange(42, dtype=np.float32).reshape(3, 14) / 10.0
    next_observations = np.stack(
        [observations + action for action in range(4)], axis=1
    ).astype(np.float32)
    rewards = np.tile(np.arange(4, dtype=np.float32), (3, 1))
    feasible = np.asarray(
        [[True, True, False, False], [True, True, True, False], [True, False, False, False]]
    )
    return BranchDataset(
        observations=observations,
        next_observations=next_observations,
        rewards=rewards,
        feasible=feasible,
        done=np.asarray([False, False, True]),
        next_load=next_observations[:, 0, 0],
        domain=np.asarray(["d", "d", "d"]),
        window_start=np.asarray([0, 1, 2]),
    )


def test_regression_uses_only_feasible_pairs_and_zero_terminal_target():
    data = branch_fixture()
    regression = build_postdecision_regression(data, {0: SumValue()})
    assert regression.states.shape == (6, 14)
    assert regression.targets[0].shape == (6,)
    assert np.all(regression.states[:, [0, 10, 11]] == 0.0)
    assert regression.targets[0][-1] == 0.0


def test_candidate_training_is_finite_and_reproducible():
    data = branch_fixture()
    first, history_a = train_postdecision_candidates(
        data,
        data,
        source_values={0: SumValue()},
        seed=11,
        epochs=2,
        learning_rate=1.0e-3,
        hidden_dim=8,
        batch_size=4,
        target_scale=10.0,
    )
    second, history_b = train_postdecision_candidates(
        data,
        data,
        source_values={0: SumValue()},
        seed=11,
        epochs=2,
        learning_rate=1.0e-3,
        hidden_dim=8,
        batch_size=4,
        target_scale=10.0,
    )
    probe = build_postdecision_regression(data, {0: SumValue()}).states
    np.testing.assert_allclose(first[0].predict(probe), second[0].predict(probe))
    assert history_a == history_b
    assert all(np.isfinite(row["training_loss"]) for row in history_a)
