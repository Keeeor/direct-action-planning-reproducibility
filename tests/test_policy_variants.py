import pytest
import torch

from dap.models.policy import ConstrainedSchedulingPolicy, PolicyConfig


@pytest.mark.parametrize(
    "method",
    ["ppo", "lagrangian", "budget_state", "fixed_local", "cdba", "cdba_discrete"],
)
def test_all_learning_variants_share_action_space_and_finite_outputs(method: str) -> None:
    policy = ConstrainedSchedulingPolicy(
        PolicyConfig(method=method, action_dim=4, episode_budget=64.0, horizon=64)
    )
    obs = torch.zeros((3, 14), dtype=torch.float32)
    obs[:, -2:] = 1.0
    output = policy.act(obs)
    assert output.action.min() >= 0 and output.action.max() < 4
    assert torch.isfinite(output.reward_value).all()
    assert torch.isfinite(output.cost_value).all()
    if method in {"fixed_local", "cdba", "cdba_discrete"}:
        assert output.local_budget is not None
        assert torch.all(output.local_budget >= 0)


def test_cdba_continuous_budget_bounds_follow_anchor() -> None:
    policy = ConstrainedSchedulingPolicy(
        PolicyConfig(
            method="cdba",
            action_dim=4,
            episode_budget=64.0,
            horizon=64,
            eta=2.0,
            min_multiplier=0.25,
            max_multiplier=4.0,
        )
    )
    obs = torch.zeros((7, 14), dtype=torch.float32)
    obs[:, -2:] = 1.0
    output = policy.act(obs, deterministic=True)
    assert torch.all(output.local_budget >= 0.25)
    assert torch.all(output.local_budget <= 4.0)


def test_budget_state_ablation_zeroes_only_registered_feature() -> None:
    raw = torch.arange(28, dtype=torch.float32).reshape(2, 14)
    no_budget = ConstrainedSchedulingPolicy(
        PolicyConfig(method="budget_state", action_dim=4, use_remaining_budget=False)
    ).prepare_observation(raw)
    no_horizon = ConstrainedSchedulingPolicy(
        PolicyConfig(method="budget_state", action_dim=4, use_remaining_horizon=False)
    ).prepare_observation(raw)
    assert torch.all(no_budget[:, -2] == 0)
    assert torch.all(no_horizon[:, -1] == 0)
    assert torch.equal(no_budget[:, :-2], no_horizon[:, :-2])


def test_fixed_local_quota_is_constant_episode_budget_per_horizon() -> None:
    policy = ConstrainedSchedulingPolicy(
        PolicyConfig(method="fixed_local", action_dim=4, episode_budget=128.0, horizon=128)
    )
    observations = torch.zeros((2, 14), dtype=torch.float32)
    observations[:, -2] = torch.tensor([1.0, 0.25])
    observations[:, -1] = torch.tensor([1.0, 0.50])
    output = policy.act(observations, deterministic=True)
    assert torch.allclose(output.local_budget, torch.ones(2))


def test_budget_update_period_holds_local_quota_until_next_boundary() -> None:
    torch.manual_seed(3)
    policy = ConstrainedSchedulingPolicy(
        PolicyConfig(
            method="cdba",
            action_dim=4,
            episode_budget=128.0,
            horizon=128,
            budget_update_period=3,
        )
    )
    first = torch.zeros((1, 14), dtype=torch.float32)
    first[:, -2:] = 1.0
    changed = first.clone()
    changed[:, :12] = 5.0
    values = [
        policy.act(first, deterministic=True).local_budget,
        policy.act(changed, deterministic=True).local_budget,
        policy.act(changed, deterministic=True).local_budget,
    ]
    assert torch.allclose(values[0], values[1])
    assert torch.allclose(values[0], values[2])
    policy.reset_budget_controller()
    reset_value = policy.act(changed, deterministic=True).local_budget
    assert not torch.allclose(values[0], reset_value)


def test_hidden_budget_feature_does_not_destroy_internal_accounting_anchor() -> None:
    policy = ConstrainedSchedulingPolicy(
        PolicyConfig(
            method="cdba",
            action_dim=4,
            episode_budget=128.0,
            horizon=128,
            use_remaining_budget=False,
        )
    )
    observations = torch.zeros((2, 14), dtype=torch.float32)
    observations[:, -2] = torch.tensor([1.0, 0.5])
    observations[:, -1] = 1.0
    output = policy.act(observations, deterministic=True)
    assert torch.all(output.local_budget > 0)
    assert torch.allclose(output.local_budget[1], output.local_budget[0] * 0.5)
    assert torch.all(policy.prepare_observation(observations)[:, -2] == 0)


def test_hidden_horizon_feature_does_not_destroy_internal_accounting_anchor() -> None:
    policy = ConstrainedSchedulingPolicy(
        PolicyConfig(
            method="cdba",
            action_dim=4,
            episode_budget=128.0,
            horizon=128,
            use_remaining_horizon=False,
        )
    )
    observations = torch.zeros((2, 14), dtype=torch.float32)
    observations[:, -2] = 1.0
    observations[:, -1] = torch.tensor([1.0, 0.5])
    output = policy.act(observations, deterministic=True)
    assert torch.allclose(output.local_budget[1], output.local_budget[0] * 2.0)
    assert torch.all(policy.prepare_observation(observations)[:, -1] == 0)
