from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


def finite_difference_shadow_price(
    current_values: torch.Tensor,
    lower_budget_values: torch.Tensor,
    delta_budget: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not delta_budget > 0:
        raise ValueError("delta_budget must be positive")
    raw = (current_values - lower_budget_values) / float(delta_budget)
    return raw, raw.clamp(min=0.0)


def price_adjust_logits(
    logits: torch.Tensor,
    shadow_price: torch.Tensor,
    action_costs: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    costs = torch.as_tensor(action_costs, dtype=logits.dtype, device=logits.device)
    mu = torch.as_tensor(shadow_price, dtype=logits.dtype, device=logits.device)
    if mu.ndim == 0:
        mu = mu.unsqueeze(0)
    penalty = float(alpha) * mu.unsqueeze(-1) * costs.unsqueeze(0)
    adjusted = logits - penalty
    before_log = F.log_softmax(logits, dim=-1)
    after_log = F.log_softmax(adjusted, dim=-1)
    before = before_log.exp()
    kl = (before * (before_log - after_log)).sum(dim=-1)
    return adjusted, penalty, kl


def monotonicity_loss(
    lower_budget_values: torch.Tensor,
    higher_budget_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    violation = (lower_budget_values - higher_budget_values).clamp(min=0.0)
    return violation.mean(), (violation > 0).to(torch.float32).mean()


def dsp_b_td_target(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    return rewards + float(gamma) * (1.0 - dones) * next_values


def _mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass(frozen=True)
class DSPPolicyConfig:
    variant: str
    episode_budget: float
    horizon: int
    action_costs: tuple[float, ...]
    hidden_dim: int = 64
    alpha: float = 1.0
    delta_budget_ratio: float = 0.05
    monotonic_coef: float = 0.1
    actor_use_budget_state: bool = True
    actor_use_horizon: bool = True
    fixed_shadow_price: float | None = None
    hard_global_budget: bool = False

    def __post_init__(self) -> None:
        if self.variant not in {"dsp_a", "dsp_b"}:
            raise ValueError("variant must be dsp_a or dsp_b")
        if self.episode_budget <= 0 or self.horizon <= 0 or self.hidden_dim <= 0:
            raise ValueError("budget, horizon, and hidden dimension must be positive")
        if len(self.action_costs) < 2 or min(self.action_costs) != 0:
            raise ValueError("actions must include a zero-cost action")
        if self.alpha < 0 or not 0 < self.delta_budget_ratio <= 1:
            raise ValueError("invalid alpha or delta-budget ratio")
        if self.monotonic_coef < 0:
            raise ValueError("monotonic coefficient must be non-negative")


@dataclass
class DSPPolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    reward_value: torch.Tensor
    cost_value: torch.Tensor
    base_logits: torch.Tensor
    adjusted_logits: torch.Tensor
    raw_shadow_price: torch.Tensor
    shadow_price: torch.Tensor
    price_penalty: torch.Tensor
    policy_kl: torch.Tensor
    q_scores: torch.Tensor | None
    post_budget_values: torch.Tensor
    monotonic_violation: torch.Tensor


class DynamicShadowPricePolicy(nn.Module):
    def __init__(self, config: DSPPolicyConfig):
        super().__init__()
        self.config = config
        actor_input_dim = 12 + int(config.actor_use_budget_state) + int(
            config.actor_use_horizon
        )
        self.actor = _mlp(actor_input_dim, len(config.action_costs), config.hidden_dim)
        self.value_network = _mlp(14, 1, config.hidden_dim)
        self.cost_value_network = _mlp(14, 1, config.hidden_dim)
        self.q_network = (
            _mlp(14, len(config.action_costs), config.hidden_dim)
            if config.variant == "dsp_b"
            else None
        )
        self.register_buffer(
            "action_costs", torch.as_tensor(config.action_costs, dtype=torch.float32)
        )
        self.register_buffer(
            "observation_scales",
            torch.as_tensor(
                [20, 20, 100, 20, 10, 1, 1, 10, 10, 1, 5, 1, 1, 1],
                dtype=torch.float32,
            ),
        )

    def normalized(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != 14:
            raise ValueError("canonical observation must contain 14 fields")
        scales = self.observation_scales.to(observation.device, observation.dtype)
        return torch.clamp(observation.to(torch.float32) / scales, -10.0, 10.0)

    def actor_features(self, normalized: torch.Tensor) -> torch.Tensor:
        fields = [normalized[..., :12]]
        if self.config.actor_use_budget_state:
            fields.append(normalized[..., 12:13])
        if self.config.actor_use_horizon:
            fields.append(normalized[..., 13:14])
        return torch.cat(fields, dim=-1)

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value_network(self.normalized(observation)).squeeze(-1)

    def _lower_budget_observation(self, observation: torch.Tensor) -> torch.Tensor:
        lower = observation.clone().to(torch.float32)
        lower[..., -2] = (
            lower[..., -2] - self.config.delta_budget_ratio
        ).clamp(min=0.0)
        return lower

    def shadow_price(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.value(observation)
        lower = self.value(self._lower_budget_observation(observation))
        delta_absolute = self.config.delta_budget_ratio * self.config.episode_budget
        raw, mu = finite_difference_shadow_price(current, lower, delta_absolute)
        if self.config.fixed_shadow_price is not None:
            mu = torch.full_like(mu, float(self.config.fixed_shadow_price))
        return raw, mu

    def monotonic_regularization(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lower = observation.clone().to(torch.float32)
        higher = observation.clone().to(torch.float32)
        lower[..., -2] = (
            lower[..., -2] - self.config.delta_budget_ratio
        ).clamp(min=0.0)
        higher[..., -2] = (
            higher[..., -2] + self.config.delta_budget_ratio
        ).clamp(max=1.0)
        return monotonicity_loss(self.value(lower), self.value(higher))

    def _post_budget_values(self, observation: torch.Tensor) -> torch.Tensor:
        batch, actions = observation.shape[0], len(self.config.action_costs)
        expanded = observation[:, None, :].expand(batch, actions, 14).clone().to(torch.float32)
        costs_ratio = self.action_costs.to(expanded.device) / self.config.episode_budget
        expanded[..., -2] = (expanded[..., -2] - costs_ratio).clamp(min=0.0)
        expanded[..., -1] = (
            expanded[..., -1] - 1.0 / self.config.horizon
        ).clamp(min=0.0)
        return self.value(expanded.reshape(batch * actions, 14)).reshape(batch, actions)

    def act(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> DSPPolicyOutput:
        normalized = self.normalized(observation)
        base_logits = self.actor(self.actor_features(normalized))
        reward_value = self.value_network(normalized).squeeze(-1)
        cost_value = self.cost_value_network(normalized).squeeze(-1)
        raw_mu, mu = self.shadow_price(observation)
        post_budget_values = self._post_budget_values(observation)
        q_scores = self.q_network(normalized) if self.q_network is not None else None
        if self.config.variant == "dsp_a":
            adjusted, penalty, kl = price_adjust_logits(
                base_logits, mu, self.action_costs, self.config.alpha
            )
        else:
            assert q_scores is not None
            adjusted = base_logits + self.config.alpha * q_scores
            penalty = base_logits - adjusted
            before_log = F.log_softmax(base_logits, dim=-1)
            after_log = F.log_softmax(adjusted, dim=-1)
            kl = (before_log.exp() * (before_log - after_log)).sum(dim=-1)
        if self.config.hard_global_budget:
            remaining = observation[..., -2].clamp(min=0.0) * self.config.episode_budget
            valid = self.action_costs.to(remaining.device).unsqueeze(0) <= remaining.unsqueeze(-1) + 1e-8
            valid[:, int(torch.argmin(self.action_costs).item())] = True
            adjusted = adjusted.masked_fill(~valid, -torch.inf)
        distribution = Categorical(logits=adjusted)
        selected = action
        if selected is None:
            selected = adjusted.argmax(dim=-1) if deterministic else distribution.sample()
        _, violation_rate = self.monotonic_regularization(observation)
        return DSPPolicyOutput(
            action=selected,
            log_prob=distribution.log_prob(selected),
            entropy=distribution.entropy(),
            reward_value=reward_value,
            cost_value=cost_value,
            base_logits=base_logits,
            adjusted_logits=adjusted,
            raw_shadow_price=raw_mu,
            shadow_price=mu,
            price_penalty=penalty,
            policy_kl=kl,
            q_scores=q_scores,
            post_budget_values=post_budget_values,
            monotonic_violation=violation_rate.expand_as(mu),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
