from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    network = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
    return network


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = mlp(input_dim, action_dim, hidden_dim)
        self.reward_critic = mlp(input_dim, 1, hidden_dim)
        self.cost_critic = mlp(input_dim, 1, hidden_dim)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, features: torch.Tensor):
        return (
            self.actor(features),
            self.reward_critic(features).squeeze(-1),
            self.cost_critic(features).squeeze(-1),
        )
