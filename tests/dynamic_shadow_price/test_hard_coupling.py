import torch

from dap.dynamic_shadow_price.hard_coupling import (
    GlobalBudgetMaskedPolicy,
    HardCoupledCDBAPolicy,
    apply_local_budget_mask,
    local_budget_action_mask,
)
from dap.models.policy import ConstrainedSchedulingPolicy, PolicyConfig


def test_local_quota_masks_actions_above_cost_and_keeps_cheapest_valid():
    costs = torch.tensor([0.0, 1.0, 2.0, 4.0])
    quotas = torch.tensor([1.5, -0.5, 4.0])
    mask = local_budget_action_mask(quotas, costs)
    assert mask.tolist() == [
        [True, True, False, False],
        [True, False, False, False],
        [True, True, True, True],
    ]


def test_masked_logits_assign_zero_probability_to_invalid_actions():
    logits = torch.tensor([[0.0, 1.0, 8.0, 12.0]])
    masked, mask = apply_local_budget_mask(
        logits, torch.tensor([1.5]), torch.tensor([0.0, 1.0, 2.0, 4.0])
    )
    probs = torch.softmax(masked, dim=-1)
    assert mask.tolist() == [[True, True, False, False]]
    assert torch.all(probs[~mask] == 0)
    assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_mask_is_batched_and_preserves_gradient_on_valid_logits():
    logits = torch.zeros((2, 4), requires_grad=True)
    masked, _ = apply_local_budget_mask(
        logits, torch.tensor([0.0, 2.0]), torch.tensor([0.0, 1.0, 2.0, 4.0])
    )
    loss = torch.log_softmax(masked, dim=-1)[1, 2]
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1, 2] != 0


def test_hard_coupled_policy_overrides_an_expensive_argmax():
    base = ConstrainedSchedulingPolicy(
        PolicyConfig(method="cdba", action_dim=4, episode_budget=128, horizon=128)
    )
    with torch.no_grad():
        for parameter in base.actor_critic.actor.parameters():
            parameter.zero_()
        base.actor_critic.actor[-1].bias.copy_(torch.tensor([0.0, 1.0, 2.0, 20.0]))
    wrapped = HardCoupledCDBAPolicy(base, [0.0, 1.0, 2.0, 4.0])
    observation = torch.zeros((1, 14))
    observation[:, -2:] = 1.0
    output = wrapped.act(
        observation,
        deterministic=True,
        local_budget_override=torch.tensor([0.5]),
    )
    assert output.action.item() == 0
    assert torch.softmax(output.logits, dim=-1)[0, 3].item() == 0.0
    assert torch.isfinite(wrapped.last_policy_kl).all()


def test_global_budget_wrapper_makes_b4_feasible_in_dp_reference():
    base = ConstrainedSchedulingPolicy(
        PolicyConfig(method="budget_state", action_dim=4, episode_budget=8, horizon=16)
    )
    with torch.no_grad():
        for parameter in base.actor_critic.actor.parameters():
            parameter.zero_()
        base.actor_critic.actor[-1].bias.copy_(torch.tensor([0.0, 1.0, 2.0, 20.0]))
    wrapped = GlobalBudgetMaskedPolicy(base, [0.0, 1.0, 2.0, 3.0], episode_budget=8)
    observation = torch.zeros((1, 14))
    observation[:, -2] = 0.125
    observation[:, -1] = 0.5
    output = wrapped.act(observation, deterministic=True)
    assert output.action.item() == 1
    assert torch.softmax(output.logits, dim=-1)[0, 2].item() == 0.0
