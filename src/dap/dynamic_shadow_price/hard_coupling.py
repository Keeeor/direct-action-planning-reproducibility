from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from dap.models.policy import PolicyOutput


def local_budget_action_mask(
    local_budget: torch.Tensor, action_costs: torch.Tensor
) -> torch.Tensor:
    """Return a batch-by-action feasibility mask for a local quota.

    The cheapest action is forced valid even for a negative or non-finite quota.
    This is a diagnostic local-quota coupling, distinct from the environment's
    global hard-budget feasibility rule.
    """

    quota = torch.as_tensor(local_budget)
    costs = torch.as_tensor(action_costs, device=quota.device, dtype=quota.dtype)
    if costs.ndim != 1 or costs.numel() == 0 or torch.any(costs < 0):
        raise ValueError("action_costs must be a non-empty non-negative vector")
    if quota.ndim == 0:
        quota = quota.unsqueeze(0)
    if quota.ndim != 1:
        raise ValueError("local_budget must be scalar or one-dimensional")
    safe_quota = torch.where(torch.isfinite(quota), quota, torch.zeros_like(quota))
    mask = costs.unsqueeze(0) <= safe_quota.clamp(min=0).unsqueeze(-1) + 1e-12
    mask[:, int(torch.argmin(costs).item())] = True
    return mask


def apply_local_budget_mask(
    logits: torch.Tensor,
    local_budget: torch.Tensor,
    action_costs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, actions]")
    mask = local_budget_action_mask(local_budget, action_costs)
    if mask.shape != logits.shape:
        raise ValueError("batch size or action dimension does not match logits")
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    return masked_logits, mask


class HardCoupledCDBAPolicy(nn.Module):
    """Add a differentiable valid-action mask to a frozen-compatible CDBA.

    D1 loads a trained base policy and evaluates this wrapper. D2 trains the
    wrapper normally. ``freeze_allocator`` controls only allocator gradients;
    evaluation callers can freeze the complete module with ``requires_grad_``.
    """

    def __init__(self, base_policy: nn.Module, action_costs, freeze_allocator: bool = False):
        super().__init__()
        self.base_policy = base_policy
        self.config = base_policy.config
        costs = torch.as_tensor(action_costs, dtype=torch.float32)
        self.register_buffer("action_costs", costs)
        self.last_invalid_probability_mass: torch.Tensor | None = None
        self.last_policy_kl: torch.Tensor | None = None
        if freeze_allocator and getattr(base_policy, "allocator", None) is not None:
            for parameter in base_policy.allocator.parameters():
                parameter.requires_grad_(False)

    def reset_budget_controller(self) -> None:
        self.base_policy.reset_budget_controller()

    def act(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        **kwargs,
    ) -> PolicyOutput:
        base = self.base_policy.act(
            observation, action=action, deterministic=deterministic, **kwargs
        )
        if base.local_budget is None:
            raise ValueError("hard coupling requires a policy with a local budget")
        logits, mask = apply_local_budget_mask(
            base.logits, base.local_budget, self.action_costs
        )
        base_log_probs = torch.log_softmax(base.logits, dim=-1)
        masked_log_probs = torch.log_softmax(logits, dim=-1)
        masked_probs = masked_log_probs.exp()
        self.last_invalid_probability_mass = (
            base_log_probs.exp() * (~mask).to(base.logits.dtype)
        ).sum(dim=-1)
        kl_terms = torch.where(
            mask,
            masked_probs * (masked_log_probs - base_log_probs),
            torch.zeros_like(masked_probs),
        )
        self.last_policy_kl = kl_terms.sum(dim=-1)
        distribution = Categorical(logits=logits)
        selected = action
        if selected is None:
            selected = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return PolicyOutput(
            action=selected,
            log_prob=distribution.log_prob(selected),
            entropy=distribution.entropy(),
            reward_value=base.reward_value,
            cost_value=base.cost_value,
            logits=logits,
            local_budget=base.local_budget,
            budget_multiplier=base.budget_multiplier,
            allocator_output=base.allocator_output,
            allocator_probabilities=base.allocator_probabilities,
            local_lambda=base.local_lambda,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class GlobalBudgetMaskedPolicy(nn.Module):
    """Give a legacy policy the same hard feasibility set as the exact DP."""

    def __init__(self, base_policy: nn.Module, action_costs, episode_budget: float):
        super().__init__()
        self.base_policy = base_policy
        self.config = base_policy.config
        self.episode_budget = float(episode_budget)
        self.register_buffer(
            "action_costs", torch.as_tensor(action_costs, dtype=torch.float32)
        )

    def reset_budget_controller(self) -> None:
        self.base_policy.reset_budget_controller()

    def act(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        **kwargs,
    ) -> PolicyOutput:
        base = self.base_policy.act(
            observation, action=action, deterministic=deterministic, **kwargs
        )
        remaining = observation[..., -2].clamp(min=0.0) * self.episode_budget
        valid = self.action_costs.to(remaining.device).unsqueeze(0) <= remaining.unsqueeze(-1) + 1e-8
        valid[:, int(torch.argmin(self.action_costs).item())] = True
        logits = base.logits.masked_fill(~valid, -torch.inf)
        distribution = Categorical(logits=logits)
        selected = action
        if selected is None:
            selected = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return PolicyOutput(
            action=selected,
            log_prob=distribution.log_prob(selected),
            entropy=distribution.entropy(),
            reward_value=base.reward_value,
            cost_value=base.cost_value,
            logits=logits,
            local_budget=base.local_budget,
            budget_multiplier=base.budget_multiplier,
            allocator_output=base.allocator_output,
            allocator_probabilities=base.allocator_probabilities,
            local_lambda=base.local_lambda,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
