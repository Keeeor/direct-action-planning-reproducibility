from __future__ import annotations

import copy

import torch
from torch import nn


class PooledValueNetwork(nn.Module):
    """Scenario-blind value model over normalized current state only."""

    def __init__(
        self,
        *,
        horizon: int,
        n_loads: int,
        max_queue: int,
        max_budget: int,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.n_loads = int(n_loads)
        self.max_queue = int(max_queue)
        self.max_budget = int(max_budget)
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.double()

    def normalize(
        self,
        remaining_horizon: torch.Tensor,
        load: torch.Tensor,
        queue: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                remaining_horizon.to(torch.float64) / max(self.horizon, 1),
                load.to(torch.float64) / max(self.n_loads - 1, 1),
                queue.to(torch.float64) / max(self.max_queue, 1),
                budget.to(torch.float64) / max(self.max_budget, 1),
            ],
            dim=-1,
        )

    def hidden(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return self.encoder(normalized_state.to(torch.float64))

    def forward(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.hidden(normalized_state)).squeeze(-1)

    def predict_state(
        self,
        remaining_horizon: torch.Tensor,
        load: torch.Tensor,
        queue: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        return self(self.normalize(remaining_horizon, load, queue, budget))


class LinearValueAdapter(nn.Module):
    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.layer = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)
        self.double()

    def forward(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return self.layer(normalized_state.to(torch.float64)).squeeze(-1)


class SmallMLPValueAdapter(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)
        self.double()

    def forward(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return self.network(normalized_state.to(torch.float64)).squeeze(-1)


class FrozenValueWithAdapter(nn.Module):
    def __init__(self, base: PooledValueNetwork, adapter: nn.Module):
        super().__init__()
        self.base = copy.deepcopy(base).eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.adapter = adapter

    def forward(self, normalized_state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.base(normalized_state)
        return base + self.adapter(normalized_state)
