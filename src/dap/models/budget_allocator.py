from __future__ import annotations

import math

import torch
from torch import nn

from .actor_critic import mlp


def local_budget_from_q(
    mean_remaining_rate: float,
    q: float,
    eta: float,
    min_multiplier: float,
    max_multiplier: float,
) -> float:
    if mean_remaining_rate < 0:
        raise ValueError("mean_remaining_rate must be non-negative")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if not 0 < min_multiplier <= max_multiplier:
        raise ValueError("invalid multiplier bounds")
    multiplier = math.exp(float(eta) * (2.0 * float(q) - 1.0))
    multiplier = min(max(multiplier, min_multiplier), max_multiplier)
    return float(mean_remaining_rate * multiplier)


class BudgetAllocator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, discrete: bool = False):
        super().__init__()
        self.discrete = discrete
        self.network = mlp(input_dim, 3 if discrete else 1, hidden_dim)

    def forward(self, features: torch.Tensor, deterministic: bool = False):
        raw = self.network(features)
        if not self.discrete:
            q = torch.sigmoid(raw.squeeze(-1))
            return q, None
        probabilities = torch.softmax(raw, dim=-1)
        modes = torch.as_tensor([0.5, 1.0, 2.0], dtype=features.dtype, device=features.device)
        if deterministic:
            indices = probabilities.argmax(dim=-1)
            multiplier = modes[indices]
        else:
            multiplier = (probabilities * modes).sum(dim=-1)
        return multiplier, probabilities
