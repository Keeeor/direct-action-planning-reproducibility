from __future__ import annotations

import numpy as np
import pytest
import torch

from dap.direct_action_planning_recent_sota.lcpo import (
    LCPOConfig,
    LCPOPolicy,
    ReservoirOODBuffer,
    feasible_action_mask,
    lcpo_policy_step,
    mahalanobis_log_likelihood,
)


def test_reservoir_retains_recent_window_and_is_seed_reproducible() -> None:
    states = np.arange(48, dtype=np.float32).reshape(12, 4)
    first = ReservoirOODBuffer(4, recent_window=4, capacity=6, seed=17)
    second = ReservoirOODBuffer(4, recent_window=4, capacity=6, seed=17)
    first.add_many(states)
    second.add_many(states)

    np.testing.assert_array_equal(first.recent_states, states[-4:])
    np.testing.assert_array_equal(first.reservoir_states, second.reservoir_states)
    assert first.seen == len(states)
    assert len(first.reservoir_states) == 6


def test_mahalanobis_uses_regularized_covariance_and_detects_distant_states() -> None:
    recent = np.zeros((20, 3), dtype=np.float64)
    candidates = np.asarray([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    score = mahalanobis_log_likelihood(candidates, recent, ridge=1.0e-3)

    assert np.isfinite(score).all()
    assert score[0] == pytest.approx(0.0)
    assert score[1] < -6.0


def test_budget_mask_never_exposes_unaffordable_action() -> None:
    observations = torch.zeros((2, 14), dtype=torch.float32)
    observations[:, -2] = torch.tensor([0.0, 0.5])
    mask = feasible_action_mask(
        observations,
        budget=4.0,
        action_costs=torch.tensor([0.0, 1.0, 2.0, 4.0]),
    )

    assert mask.tolist() == [[True, False, False, False], [True, True, True, False]]


def test_lcpo_step_respects_recent_and_anchor_kl_bounds() -> None:
    torch.manual_seed(7)
    policy = LCPOPolicy(observation_dim=14, action_dim=4, hidden_dim=16)
    observations = torch.randn(32, 14)
    observations[:, -2] = 1.0
    anchors = torch.randn(32, 14)
    anchors[:, -2] = 1.0
    actions = torch.randint(0, 4, (32,))
    advantages = torch.linspace(-1.0, 1.0, 32)
    mask = torch.ones((32, 4), dtype=torch.bool)
    anchor_mask = torch.ones((32, 4), dtype=torch.bool)

    diagnostics = lcpo_policy_step(
        policy,
        observations,
        actions,
        advantages,
        mask,
        anchors,
        anchor_mask,
        entropy_coef=0.0,
        recent_kl_limit=0.1,
        anchor_kl_limit=1.0e-4,
        damping=0.1,
        cg_steps=15,
        max_backtracks=10,
    )

    assert diagnostics["recent_kl"] <= 0.1 + 1.0e-7
    assert diagnostics["anchor_kl"] <= 1.0e-4 + 1.0e-7
    assert np.isfinite(list(diagnostics.values())).all()


def test_config_preserves_registered_official_components() -> None:
    config = LCPOConfig()
    assert config.batch_size == 200
    assert config.total_steps == 102_400
    assert config.recent_window == 200
    assert config.reservoir_capacity == 1_024
    assert config.recent_kl_limit == pytest.approx(0.1)
    assert config.anchor_kl_limit == pytest.approx(1.0e-4)
    assert config.solve_dual is False
    assert config.context_indices == (0, 1, 10)
