from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from .actor_critic import ActorCritic
from .budget_allocator import BudgetAllocator
from .local_multiplier import LocalMultiplier


METHODS = {"ppo", "lagrangian", "budget_state", "fixed_local", "cdba", "cdba_discrete"}


@dataclass(frozen=True)
class PolicyConfig:
    method: str
    action_dim: int
    hidden_dim: int = 64
    episode_budget: float = 128.0
    horizon: int = 128
    eta: float = 1.0
    min_multiplier: float = 0.25
    max_multiplier: float = 4.0
    use_remaining_budget: bool = True
    use_remaining_horizon: bool = True
    use_local_lambda: bool = True
    budget_update_period: int = 1

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown method: {self.method}")
        if self.action_dim <= 1 or self.hidden_dim <= 0 or self.horizon <= 0:
            raise ValueError("invalid network or environment dimensions")
        if self.budget_update_period <= 0:
            raise ValueError("budget_update_period must be positive")


@dataclass
class PolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    reward_value: torch.Tensor
    cost_value: torch.Tensor
    logits: torch.Tensor
    local_budget: torch.Tensor | None = None
    budget_multiplier: torch.Tensor | None = None
    allocator_output: torch.Tensor | None = None
    allocator_probabilities: torch.Tensor | None = None
    local_lambda: torch.Tensor | None = None


class ConstrainedSchedulingPolicy(nn.Module):
    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        if config.method in {"ppo", "lagrangian"}:
            actor_input_dim = 12
        elif config.method == "budget_state":
            actor_input_dim = 14
        else:
            actor_input_dim = 15
        self.actor_critic = ActorCritic(actor_input_dim, config.action_dim, config.hidden_dim)
        self.allocator: BudgetAllocator | None = None
        self.local_multiplier: LocalMultiplier | None = None
        if config.method in {"cdba", "cdba_discrete"}:
            self.allocator = BudgetAllocator(
                14, config.hidden_dim, discrete=config.method == "cdba_discrete"
            )
            if config.use_local_lambda:
                self.local_multiplier = LocalMultiplier(15, config.hidden_dim)
        self._held_local_budget: torch.Tensor | None = None
        self._budget_controller_step = 0

    def reset_budget_controller(self) -> None:
        self._held_local_budget = None
        self._budget_controller_step = 0

    def prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != 14:
            raise ValueError("canonical observation must contain 14 fields")
        obs = observation.to(dtype=torch.float32).clone()
        scales = torch.as_tensor(
            [20, 20, 100, 20, 10, 1, 1, 10, 10, 1, 5, 1, 1, 1],
            dtype=obs.dtype,
            device=obs.device,
        )
        obs = torch.clamp(obs / scales, -10.0, 10.0)
        if not self.config.use_remaining_budget:
            obs[..., -2] = 0.0
        if not self.config.use_remaining_horizon:
            obs[..., -1] = 0.0
        return obs

    def _features(
        self,
        observation: torch.Tensor,
        deterministic_allocator: bool = False,
        local_budget_override: torch.Tensor | None = None,
    ):
        normalized = self.prepare_observation(observation)
        method = self.config.method
        if method in {"ppo", "lagrangian"}:
            return normalized[..., :12], {}
        if method == "budget_state":
            return normalized, {}

        # Accounting variables remain available to the internal quota anchor
        # even when an ablation hides them from the learned networks. Otherwise
        # "no input" would inadvertently alter the controller definition.
        accounting = observation.to(dtype=torch.float32)
        remaining_budget_ratio = accounting[..., -2].clamp(min=0.0)
        remaining_horizon_ratio = accounting[..., -1].clamp(min=0.0)
        remaining_steps = torch.clamp(
            remaining_horizon_ratio * float(self.config.horizon), min=1.0
        )
        mean_rate = self.config.episode_budget * remaining_budget_ratio / remaining_steps
        extras: dict[str, torch.Tensor | None] = {}
        if local_budget_override is not None:
            local_budget = local_budget_override.to(device=mean_rate.device, dtype=mean_rate.dtype)
            multiplier = local_budget / mean_rate.clamp(min=1e-8)
            extras["allocator_output"] = None
            extras["allocator_probabilities"] = None
        elif method == "fixed_local":
            multiplier = torch.ones_like(mean_rate)
            # B5 is the genuinely fixed quota baseline: it must not silently
            # adapt to the remaining budget or the remaining horizon.
            local_budget = torch.full_like(
                mean_rate, self.config.episode_budget / float(self.config.horizon)
            )
            extras["allocator_output"] = None
            extras["allocator_probabilities"] = None
        else:
            assert self.allocator is not None
            allocator_output, probabilities = self.allocator(
                normalized, deterministic=deterministic_allocator
            )
            if method == "cdba":
                q = allocator_output
                multiplier = torch.exp(self.config.eta * (2.0 * q - 1.0)).clamp(
                    self.config.min_multiplier, self.config.max_multiplier
                )
            else:
                q = allocator_output
                multiplier = allocator_output.clamp(
                    self.config.min_multiplier, self.config.max_multiplier
                )
            local_budget = mean_rate * multiplier
            extras["allocator_output"] = q
            extras["allocator_probabilities"] = probabilities
        local_scaled = (local_budget / 4.0).unsqueeze(-1)
        actor_features = torch.cat(
            [normalized[..., :12], local_scaled, normalized[..., -2:]], dim=-1
        )
        extras["local_budget"] = local_budget
        extras["budget_multiplier"] = multiplier
        if self.local_multiplier is not None:
            extras["local_lambda"] = self.local_multiplier(actor_features)
        else:
            extras["local_lambda"] = None
        return actor_features, extras

    def act(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        local_budget_override: torch.Tensor | None = None,
        advance_budget_state: bool = True,
    ) -> PolicyOutput:
        stateful_period = (
            self.config.method in {"cdba", "cdba_discrete"}
            and self.config.budget_update_period > 1
            and action is None
            and observation.shape[0] == 1
            and local_budget_override is None
        )
        if stateful_period and self._held_local_budget is not None:
            if self._budget_controller_step % self.config.budget_update_period != 0:
                local_budget_override = self._held_local_budget
        features, extras = self._features(
            observation,
            deterministic_allocator=deterministic,
            local_budget_override=local_budget_override,
        )
        if stateful_period and advance_budget_state:
            if local_budget_override is None and extras.get("local_budget") is not None:
                self._held_local_budget = extras["local_budget"].detach().clone()
            self._budget_controller_step += 1
        logits, reward_value, cost_value = self.actor_critic(features)
        distribution = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return PolicyOutput(
            action=action,
            log_prob=distribution.log_prob(action),
            entropy=distribution.entropy(),
            reward_value=reward_value,
            cost_value=cost_value,
            logits=logits,
            local_budget=extras.get("local_budget"),
            budget_multiplier=extras.get("budget_multiplier"),
            allocator_output=extras.get("allocator_output"),
            allocator_probabilities=extras.get("allocator_probabilities"),
            local_lambda=extras.get("local_lambda"),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
