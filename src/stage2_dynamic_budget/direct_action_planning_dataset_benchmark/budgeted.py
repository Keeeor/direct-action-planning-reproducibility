from __future__ import annotations

from dataclasses import dataclass
import copy
import time

import numpy as np
import torch
from torch import nn

from stage2_dynamic_budget.direct_action_planning_dataset_validation.data import TraceDataset
from stage2_dynamic_budget.direct_action_planning_dataset_validation.training import (
    BranchDataset,
    collect_branch_dataset,
)

from .models import normalize_observation


class BudgetedQNetwork(nn.Module):
    """Vector reward/cost action values for a deterministic BMDP extreme point."""

    def __init__(self, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(14, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.reward_head = nn.Linear(hidden_dim, action_dim)
        self.cost_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(normalize_observation(observation))
        return self.reward_head(hidden), torch.nn.functional.softplus(self.cost_head(hidden))


class BudgetedFittedQAgent:
    """Budgeted Fitted-Q policy adapted from the BMDP Bellman operator.

    The original BFTQ implementation supports convex-hull mixtures. The simulator evaluates a
    deterministic action at every step, so this port selects the highest reward extreme point whose
    predicted continuation cost fits the remaining budget, falling back to the lowest-cost point.
    This distinction is recorded in the method card and is not presented as an exact reproduction
    of randomized BFTQ.
    """

    name = "budgeted_fitted_q"

    def __init__(self, model: BudgetedQNetwork, budget: float, action_costs: np.ndarray):
        self.model = model.eval()
        self.budget = float(budget)
        self.action_costs = np.asarray(action_costs, dtype=np.float64)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def select(self, env, observation: np.ndarray):
        reward_values, cost_values = self.model(torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1))
        reward_values = reward_values[0].cpu().numpy().astype(np.float64)
        cost_values = cost_values[0].cpu().numpy().astype(np.float64)
        remaining = max(self.budget - float(env.cumulative_cost), 0.0)
        feasible = self.action_costs <= remaining + 1.0e-8
        feasible &= cost_values <= remaining + 1.0e-6
        if not np.any(feasible):
            feasible = self.action_costs <= remaining + 1.0e-8
        scores = reward_values.copy()
        scores[~feasible] = -np.inf
        return int(np.argmax(scores)), reward_values - 0.01 * cost_values


def train_budgeted_fitted_q(
    dataset: TraceDataset,
    *,
    horizon: int,
    budget: float,
    seed: int,
    episodes_per_domain: int = 16,
    iterations: int = 8,
    epochs_per_iteration: int = 2,
    hidden_dim: int = 128,
) -> tuple[BudgetedFittedQAgent, list[dict[str, float]], float]:
    """Train a budget-conditioned fitted-Q baseline on all affordable action branches."""
    started = time.perf_counter()
    training = collect_branch_dataset(
        dataset,
        split="train",
        horizon=horizon,
        budget=budget,
        episodes_per_domain=episodes_per_domain,
        seed=seed,
    )
    model = BudgetedQNetwork(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4)
    observations = torch.as_tensor(training.observations, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    action_costs = np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    history: list[dict[str, float]] = []
    valid = np.argwhere(training.feasible)
    if len(valid) == 0:
        raise ValueError("budgeted fitted-Q data has no affordable branches")
    model.eval()
    for iteration in range(iterations):
        with torch.no_grad():
            flat_next = torch.as_tensor(training.next_observations[valid[:, 0], valid[:, 1]], dtype=torch.float32)
            next_reward, next_cost = model(flat_next)
            next_reward = next_reward.numpy()
            next_cost = next_cost.numpy()
        targets_reward = np.zeros(len(valid), dtype=np.float32)
        targets_cost = np.zeros(len(valid), dtype=np.float32)
        for row, (state_index, action) in enumerate(valid):
            state_index = int(state_index)
            action = int(action)
            next_observation = training.next_observations[state_index, action]
            remaining_next = max(float(next_observation[-2]) * float(budget), 0.0)
            feasible_next = action_costs <= remaining_next + 1.0e-8
            if not np.any(feasible_next):
                feasible_next[0] = True
            cost_safe = feasible_next & (next_cost[row] <= remaining_next + 1.0e-6)
            choices = np.flatnonzero(cost_safe if np.any(cost_safe) else feasible_next)
            selected = int(choices[np.argmax(next_reward[row, choices])])
            reward = float(training.rewards[state_index, action])
            done = bool(training.done[state_index])
            targets_reward[row] = reward + (0.99 * float(next_reward[row, selected]) if not done else 0.0)
            targets_cost[row] = float(action_costs[action]) + (0.99 * float(next_cost[row, selected]) if not done else 0.0)
        branch_observations = observations[valid[:, 0]]
        target_reward_tensor = torch.as_tensor(targets_reward, dtype=torch.float32)
        target_cost_tensor = torch.as_tensor(targets_cost, dtype=torch.float32)
        losses = []
        model.train()
        for _epoch in range(epochs_per_iteration):
            permutation = rng.permutation(len(valid))
            for start in range(0, len(valid), 256):
                index = torch.as_tensor(permutation[start : start + 256], dtype=torch.long)
                reward_values, cost_values = model(branch_observations[index])
                actions = torch.as_tensor(valid[index.numpy(), 1], dtype=torch.long)
                prediction_reward = reward_values.gather(1, actions[:, None]).squeeze(1)
                prediction_cost = cost_values.gather(1, actions[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(prediction_reward, target_reward_tensor[index]) + nn.functional.smooth_l1_loss(prediction_cost, target_cost_tensor[index])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        history.append({"iteration": float(iteration), "loss": float(np.mean(losses)) if losses else float("nan"), "branches": float(len(valid))})
    return BudgetedFittedQAgent(model, budget, action_costs), history, time.perf_counter() - started

