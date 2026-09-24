from __future__ import annotations

import torch
from torch import nn

from .actor_critic import mlp


class LocalMultiplier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = mlp(input_dim, 1, hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.network(features).squeeze(-1))
