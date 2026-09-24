from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


OBS_SCALES = torch.as_tensor(
    [20, 20, 100, 20, 10, 1, 1, 10, 10, 1, 5, 1, 1, 1], dtype=torch.float32
)


def normalize_observation(observation: torch.Tensor) -> torch.Tensor:
    scales = OBS_SCALES.to(device=observation.device, dtype=observation.dtype)
    return torch.clamp(observation.to(torch.float32) / scales, -10.0, 10.0)


def action_mask(observation: torch.Tensor, action_costs: torch.Tensor) -> torch.Tensor:
    remaining = observation[..., -2].clamp(min=0.0)
    # The observation carries a ratio; the caller supplies costs in normalized units.
    budget = action_costs.new_tensor(1.0)
    return action_costs <= remaining.unsqueeze(-1) * budget + 1.0e-6


class ActorCritic(nn.Module):
    def __init__(self, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(14, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.reward_value = nn.Linear(hidden_dim, 1)
        self.cost_value = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor):
        hidden = self.body(normalize_observation(observation))
        return self.actor(hidden), self.reward_value(hidden).squeeze(-1), self.cost_value(hidden).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(self, action_dim: int = 4, hidden_dim: int = 128, dueling: bool = True):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(14, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.dueling = bool(dueling)
        if dueling:
            self.value = nn.Linear(hidden_dim, 1)
            self.advantage = nn.Linear(hidden_dim, action_dim)
        else:
            self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self.body(normalize_observation(observation))
        if self.dueling:
            return self.value(hidden) + self.advantage(hidden) - self.advantage(hidden).mean(dim=-1, keepdim=True)
        return self.head(hidden)


@dataclass(frozen=True)
class AgentDecision:
    action: int
    q_values: np.ndarray
    diagnostics: dict[str, float]

