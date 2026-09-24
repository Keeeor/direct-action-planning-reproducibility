from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from stage2_dynamic_budget.agents.ppo import PPOConfig, PPOTrainer

from .datasets import training_arrays
from .model import ActionAdvantageModel, pairwise_ranking_loss


@dataclass(frozen=True)
class AdvantageFitConfig:
    branch_horizon: int = 10
    epochs: int = 120
    batch_size: int = 512
    learning_rate: float = 1e-3
    hidden_dim: int = 64


def fit_advantage_model(
    frame: pd.DataFrame,
    config: AdvantageFitConfig,
    seed: int,
    device: torch.device,
) -> tuple[ActionAdvantageModel, list[dict[str, float]]]:
    torch.manual_seed(seed)
    observations, targets, masks = training_arrays(frame, config.branch_horizon)
    model = ActionAdvantageModel(targets.shape[1], config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
    target = torch.as_tensor(targets, dtype=torch.float32, device=device)
    mask = torch.as_tensor(masks, dtype=torch.bool, device=device)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        losses: list[float] = []
        permutation = rng.permutation(len(obs))
        for start in range(0, len(obs), config.batch_size):
            index = torch.as_tensor(
                permutation[start : start + config.batch_size],
                dtype=torch.long,
                device=device,
            )
            prediction = model(obs[index])
            error = (prediction - target[index]) ** 2
            loss = error[mask[index]].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            history.append({"epoch": epoch + 1, "mse": float(np.mean(losses))})
    history.append({"elapsed_seconds": time.perf_counter() - started})
    model.eval()
    return model, history


class ACBABTrainer(PPOTrainer):
    """Budget-state Lagrangian PPO with a branch-informed action ranking update."""

    def __init__(
        self,
        *args,
        teacher: ActionAdvantageModel,
        ranking_coef: float,
        ranking_margin: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if ranking_coef < 0 or ranking_margin < 0:
            raise ValueError("ranking coefficient and margin must be non-negative")
        self.teacher = teacher.to(self.device).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.ranking_coef = float(ranking_coef)
        self.ranking_margin = float(ranking_margin)

    def _update(self, buffer, advantages, reward_returns, cost_returns):
        stats = super()._update(buffer, advantages, reward_returns, cost_returns)
        obs = torch.as_tensor(np.asarray(buffer["obs"]), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            targets = self.teacher(obs)
        base_policy = getattr(self.policy, "base_policy", self.policy)
        output = base_policy.act(obs, deterministic=True)
        remaining = obs[..., -2].clamp(min=0.0) * float(self.budget)
        costs = self.policy.action_costs.to(self.device)
        valid = costs.unsqueeze(0) <= remaining.unsqueeze(-1) + 1e-8
        rank_loss = pairwise_ranking_loss(
            output.logits, targets, margin=self.ranking_margin, valid_mask=valid
        )
        self.optimizer.zero_grad(set_to_none=True)
        (self.ranking_coef * rank_loss).backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        stats["ranking_loss"] = float(rank_loss.detach())
        return stats


def acba_b_ppo_config(base: dict, total_steps: int, rollout_steps: int) -> PPOConfig:
    allowed = set(PPOConfig.__dataclass_fields__)
    values = {key: value for key, value in base.items() if key in allowed}
    values["total_steps"] = int(total_steps)
    values["rollout_steps"] = int(rollout_steps)
    return PPOConfig(**values)
