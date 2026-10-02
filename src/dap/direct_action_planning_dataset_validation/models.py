from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from dap.models.policy import (
    ConstrainedSchedulingPolicy,
    PolicyConfig,
    PolicyOutput,
)


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureNormalizer":
        values = np.asarray(values, dtype=np.float64)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale < 1.0e-6, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(values) - self.mean) / self.scale, -10.0, 10.0)

    def transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        scale = torch.as_tensor(self.scale, dtype=values.dtype, device=values.device)
        return torch.clamp((values - mean) / scale, -10.0, 10.0)


class ValueNetwork(nn.Module):
    def __init__(self, normalizer: FeatureNormalizer, hidden_dim: int = 64):
        super().__init__()
        self.normalizer = normalizer
        self.network = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = self.normalizer.transform_tensor(observation.to(torch.float32))
        return self.network(normalized).squeeze(-1)

    @torch.no_grad()
    def predict(self, observation: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32)
        return self(tensor).cpu().numpy()


class LoadForecaster(nn.Module):
    def __init__(self, normalizer: FeatureNormalizer, hidden_dim: int = 32):
        super().__init__()
        self.normalizer = normalizer
        self.network = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.normalizer.transform_tensor(observation.to(torch.float32))
        return torch.nn.functional.softplus(self.network(x).squeeze(-1))

    @torch.no_grad()
    def predict(self, observation: np.ndarray) -> float:
        tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        return float(self(tensor).item())


class FullTransitionNetwork(nn.Module):
    def __init__(self, normalizer: FeatureNormalizer, hidden_dim: int = 96):
        super().__init__()
        self.normalizer = normalizer
        self.network = nn.Sequential(
            nn.Linear(18, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 15),
        )

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.normalizer.transform_tensor(observation.to(torch.float32))
        one_hot = torch.nn.functional.one_hot(action.long(), num_classes=4).to(torch.float32)
        output = self.network(torch.cat([normalized, one_hot], dim=-1))
        delta = output[..., :14]
        reward = output[..., 14]
        predicted_normalized = normalized + delta
        mean = torch.as_tensor(
            self.normalizer.mean, dtype=observation.dtype, device=observation.device
        )
        scale = torch.as_tensor(
            self.normalizer.scale, dtype=observation.dtype, device=observation.device
        )
        raw_next = predicted_normalized * scale + mean
        next_observation = torch.cat(
            [raw_next[..., :-2], raw_next[..., -2:].clamp(0.0, 1.0)], dim=-1
        )
        return next_observation, reward


class MaskedBudgetStatePolicy(ConstrainedSchedulingPolicy):
    """B4 with the same hard affordability mask used by the planners."""

    def __init__(self, config: PolicyConfig, action_costs: np.ndarray):
        super().__init__(config)
        self.register_buffer("hard_action_costs", torch.as_tensor(action_costs, dtype=torch.float32))

    def act(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        local_budget_override: torch.Tensor | None = None,
        advance_budget_state: bool = True,
    ) -> PolicyOutput:
        del local_budget_override, advance_budget_state
        features, extras = self._features(observation)
        logits, reward_value, cost_value = self.actor_critic(features)
        remaining_budget = observation[..., -2] * float(self.config.episode_budget)
        feasible = self.hard_action_costs.to(logits.device) <= remaining_budget.unsqueeze(-1) + 1e-6
        logits = logits.masked_fill(~feasible, -1.0e9)
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
