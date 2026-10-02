from __future__ import annotations

import torch
from torch import nn


OBSERVATION_SCALES = (
    20.0,
    20.0,
    100.0,
    20.0,
    10.0,
    1.0,
    1.0,
    10.0,
    10.0,
    1.0,
    5.0,
    1.0,
    1.0,
    1.0,
)


def advantage_adjust_logits(
    logits: torch.Tensor, advantages: torch.Tensor, alpha: float
) -> torch.Tensor:
    if logits.shape != advantages.shape:
        raise ValueError("logits and advantages must have identical shapes")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    centered = advantages - advantages[..., :1]
    return logits + float(alpha) * centered


def pairwise_ranking_loss(
    logits: torch.Tensor,
    target_advantages: torch.Tensor,
    margin: float = 0.05,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape != target_advantages.shape or logits.ndim != 2:
        raise ValueError("logits and targets must have shape [batch, actions]")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    action_count = logits.shape[-1]
    upper = torch.triu(
        torch.ones(action_count, action_count, device=logits.device, dtype=torch.bool),
        diagonal=1,
    )
    target_delta = target_advantages.unsqueeze(-1) - target_advantages.unsqueeze(-2)
    logit_delta = logits.unsqueeze(-1) - logits.unsqueeze(-2)
    comparable = upper.unsqueeze(0) & (target_delta.abs() > 1e-8)
    if valid_mask is not None:
        if valid_mask.shape != logits.shape:
            raise ValueError("valid mask must match logits")
        comparable = comparable & valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)
    if not torch.any(comparable):
        return logits.sum() * 0.0
    direction = target_delta.sign()
    losses = torch.relu(float(margin) - direction * logit_delta)
    return losses[comparable].mean()


class ActionAdvantageModel(nn.Module):
    """Predict one centered branch advantage per action from the canonical state."""

    def __init__(self, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        if action_dim < 2 or hidden_dim <= 0:
            raise ValueError("invalid model dimensions")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.register_buffer("observation_scales", torch.tensor(OBSERVATION_SCALES))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != 14:
            raise ValueError("canonical observation must contain 14 fields")
        scales = self.observation_scales.to(observation.device, observation.dtype)
        normalized = torch.clamp(observation.to(torch.float32) / scales, -10.0, 10.0)
        markov_state = normalized[..., [0, 2, 12, 13]]
        raw = self.network(markov_state)
        return raw - raw[..., :1]
