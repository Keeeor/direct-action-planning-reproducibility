import torch

from dap.dynamic_shadow_price.shadow_policy import (
    DSPPolicyConfig,
    DynamicShadowPricePolicy,
    dsp_b_td_target,
    finite_difference_shadow_price,
    monotonicity_loss,
    price_adjust_logits,
)


def test_finite_difference_shadow_price_has_expected_scale_and_clamps_negative():
    current = torch.tensor([10.0, 3.0])
    lower = torch.tensor([8.0, 4.0])
    raw, mu = finite_difference_shadow_price(current, lower, delta_budget=0.5)
    assert torch.allclose(raw, torch.tensor([4.0, -2.0]))
    assert torch.allclose(mu, torch.tensor([4.0, 0.0]))


def test_price_adjustment_directly_reduces_expensive_action_probability():
    logits = torch.zeros((1, 4))
    costs = torch.tensor([0.0, 1.0, 2.0, 4.0])
    adjusted, penalty, kl = price_adjust_logits(
        logits, torch.tensor([1.5]), costs, alpha=1.0
    )
    before = torch.softmax(logits, dim=-1)
    after = torch.softmax(adjusted, dim=-1)
    assert torch.allclose(penalty, torch.tensor([[0.0, 1.5, 3.0, 6.0]]))
    assert after[0, 3] < before[0, 3]
    assert after[0, 0] > before[0, 0]
    assert kl.item() > 0


def test_monotonicity_loss_penalizes_value_decrease_with_more_budget():
    low_budget_value = torch.tensor([1.0, 4.0])
    high_budget_value = torch.tensor([2.0, 3.0])
    loss, violation_rate = monotonicity_loss(low_budget_value, high_budget_value)
    assert torch.isclose(loss, torch.tensor(0.5))
    assert torch.isclose(violation_rate, torch.tensor(0.5))


def test_dsp_b_target_uses_sampled_post_action_transition_and_done_mask():
    target = dsp_b_td_target(
        rewards=torch.tensor([1.0, 2.0]),
        next_values=torch.tensor([5.0, 7.0]),
        dones=torch.tensor([0.0, 1.0]),
        gamma=0.9,
    )
    assert torch.allclose(target, torch.tensor([5.5, 2.0]))


def test_dsp_policy_outputs_finite_price_and_enforces_global_budget_mask():
    policy = DynamicShadowPricePolicy(
        DSPPolicyConfig(
            variant="dsp_a",
            episode_budget=8.0,
            horizon=16,
            action_costs=(0.0, 1.0, 2.0, 3.0),
            hard_global_budget=True,
        )
    )
    observation = torch.zeros((1, 14))
    observation[:, -2] = 0.125  # one absolute budget unit remains
    observation[:, -1] = 0.5
    output = policy.act(observation, deterministic=True)
    probabilities = torch.softmax(output.adjusted_logits, dim=-1)
    assert torch.isfinite(output.shadow_price).all()
    assert probabilities[0, 2].item() == 0.0
    assert probabilities[0, 3].item() == 0.0


def test_dsp_monotonic_regularizer_is_differentiable():
    policy = DynamicShadowPricePolicy(
        DSPPolicyConfig(
            variant="dsp_b",
            episode_budget=8.0,
            horizon=16,
            action_costs=(0.0, 1.0, 2.0, 3.0),
        )
    )
    observation = torch.rand((6, 14))
    observation[:, -2:] = torch.rand((6, 2))
    loss, rate = policy.monotonic_regularization(observation)
    loss.backward()
    gradients = [p.grad for p in policy.value_network.parameters() if p.grad is not None]
    assert gradients
    assert torch.isfinite(loss)
    assert 0 <= rate.item() <= 1
