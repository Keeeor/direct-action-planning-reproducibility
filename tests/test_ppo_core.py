import numpy as np
import torch

from stage2_dynamic_budget.agents.ppo import PPOConfig, compute_gae
from stage2_dynamic_budget.models.policy import ConstrainedSchedulingPolicy, PolicyConfig


def test_gae_matches_hand_calculation_without_bootstrap() -> None:
    rewards = np.asarray([1.0, 1.0])
    values = np.asarray([0.5, 0.25])
    dones = np.asarray([0.0, 1.0])
    advantages, returns = compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=1.0)
    np.testing.assert_allclose(advantages, [1.5, 0.75])
    np.testing.assert_allclose(returns, [2.0, 1.0])


def test_policy_outputs_finite_action_logprob_and_two_values() -> None:
    policy = ConstrainedSchedulingPolicy(PolicyConfig(method="ppo", action_dim=4))
    obs = torch.zeros((5, 14), dtype=torch.float32)
    output = policy.act(obs, deterministic=False)
    assert output.action.shape == (5,)
    assert output.log_prob.shape == (5,)
    assert output.reward_value.shape == (5,)
    assert output.cost_value.shape == (5,)
    assert torch.isfinite(output.log_prob).all()


def test_ppo_config_rejects_invalid_clip_range() -> None:
    try:
        PPOConfig(clip_coef=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid clip_coef should fail")
