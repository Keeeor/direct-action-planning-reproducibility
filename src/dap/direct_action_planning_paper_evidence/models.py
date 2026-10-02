from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from dap.direct_action_planning_dataset_validation.models import (
    FeatureNormalizer,
)


class EvidenceValueNetwork(nn.Module):
    """Value network with an explicit, auditable input-feature mask."""

    def __init__(
        self,
        normalizer: FeatureNormalizer,
        hidden_dim: int = 64,
        feature_mask: np.ndarray | None = None,
    ):
        super().__init__()
        self.normalizer = normalizer
        mask = np.ones(14, dtype=np.float32) if feature_mask is None else feature_mask
        mask = np.asarray(mask, dtype=np.float32)
        if mask.shape != (14,):
            raise ValueError("feature_mask must have shape (14,)")
        self.register_buffer("feature_mask", torch.as_tensor(mask, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = self.normalizer.transform_tensor(observation.to(torch.float32))
        masked = normalized * self.feature_mask.to(normalized.device)
        return self.network(masked).squeeze(-1)

    @torch.no_grad()
    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self(torch.as_tensor(observation, dtype=torch.float32)).cpu().numpy()


class EvidenceLoadForecaster(nn.Module):
    """Action-invariant model for the only unknown next-state component: load."""

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
        normalized = self.normalizer.transform_tensor(observation.to(torch.float32))
        return torch.nn.functional.softplus(self.network(normalized).squeeze(-1))

    @torch.no_grad()
    def predict(self, observation: np.ndarray) -> float:
        values = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        return float(self(values).item())


@dataclass(frozen=True)
class DistillationHistory:
    epoch: int
    training_loss: float
    validation_accuracy: float


class DistilledActionPolicy(nn.Module):
    """Policy-only ablation trained from the same planner action targets."""

    def __init__(
        self,
        normalizer: FeatureNormalizer,
        action_costs: np.ndarray,
        episode_budget: float,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.normalizer = normalizer
        self.episode_budget = float(episode_budget)
        self.register_buffer(
            "action_costs", torch.as_tensor(action_costs, dtype=torch.float32)
        )
        self.network = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(action_costs)),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = self.normalizer.transform_tensor(observation.to(torch.float32))
        return self.network(normalized)

    @torch.no_grad()
    def act(self, observation: torch.Tensor, deterministic: bool = True):
        del deterministic
        logits = self(observation)
        remaining = observation[..., -2] * self.episode_budget
        feasible = self.action_costs.to(logits.device) <= remaining.unsqueeze(-1) + 1.0e-8
        logits = logits.masked_fill(~feasible, -1.0e9)
        return SimpleNamespace(action=torch.argmax(logits, dim=-1), logits=logits)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))

