from __future__ import annotations

import numpy as np
import torch

from stage2_dynamic_budget.direct_action_planning_dataset_validation.models import (
    FeatureNormalizer,
    FullTransitionNetwork,
    LoadForecaster,
    MaskedBudgetStatePolicy,
    ValueNetwork,
)
from stage2_dynamic_budget.direct_action_planning_dataset_validation.planning import (
    assert_structured_planner_prefix_invariant,
    make_planner,
)
from stage2_dynamic_budget.envs.synthetic_queue_env import SyntheticQueueConfig
from stage2_dynamic_budget.envs.trace_driven_env import TraceDrivenQueueEnv
from stage2_dynamic_budget.models.policy import PolicyConfig


def _models():
    normalizer = FeatureNormalizer.fit(np.vstack([np.zeros(14), np.ones(14)]))
    return (
        ValueNetwork(normalizer, hidden_dim=8).eval(),
        LoadForecaster(normalizer, hidden_dim=8).eval(),
        FullTransitionNetwork(normalizer, hidden_dim=8).eval(),
    )


def test_structured_planner_does_not_read_unseen_trace_suffix() -> None:
    value, forecaster, transition = _models()
    planner = make_planner(
        "structured_dap", value, forecaster, transition, gamma=0.99
    )
    config = SyntheticQueueConfig(horizon=4, budget=4.0)
    env_a = TraceDrivenQueueEnv(np.array([5.0, 0.0, 0.0, 0.0]), config)
    env_b = TraceDrivenQueueEnv(np.array([5.0, 100.0, 100.0, 100.0]), config)
    assert_structured_planner_prefix_invariant(planner, env_a, env_b)


def test_planner_masks_unaffordable_actions() -> None:
    value, forecaster, transition = _models()
    planner = make_planner(
        "structured_dap", value, forecaster, transition, gamma=0.99
    )
    env = TraceDrivenQueueEnv(
        np.array([10.0, 10.0]), SyntheticQueueConfig(horizon=2, budget=0.5)
    )
    observation, _ = env.reset(seed=0)
    action, q_values = planner(env, observation)
    assert action == 0
    assert np.isneginf(q_values[1:]).all()


def test_b4_hard_mask_applies_during_sampling_and_evaluation() -> None:
    policy = MaskedBudgetStatePolicy(
        PolicyConfig(
            method="budget_state",
            action_dim=4,
            episode_budget=10.0,
            horizon=4,
            hidden_dim=8,
        ),
        np.array([0.0, 1.0, 2.0, 4.0]),
    )
    observation = torch.zeros((1, 14), dtype=torch.float32)
    observation[0, -2] = 0.05
    observation[0, -1] = 1.0
    output = policy.act(observation, deterministic=True)
    assert int(output.action.item()) == 0
    assert torch.all(output.logits[0, 1:] < -1.0e8)
