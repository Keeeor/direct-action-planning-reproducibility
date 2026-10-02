from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from dap.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
)
from dap.direct_action_planning_repair.model import StructuredActionEffectModel

from .history import CausalHistory
from .model import ContextValuePredictor


@dataclass(frozen=True)
class ContextPlanResult:
    action: int
    q_values: np.ndarray
    next_load_probabilities: np.ndarray


@dataclass(frozen=True)
class ContextPlanningOutput:
    action: torch.Tensor
    q_values: torch.Tensor


def context_one_step_plan(
    mdp: ActionConditionedBudgetMDP,
    predictor: ContextValuePredictor,
    transition: StructuredActionEffectModel,
    history: CausalHistory,
    t: int,
    load: int,
    queue: int,
    budget: int,
) -> ContextPlanResult:
    remaining_horizon = mdp.config.horizon - t
    q_values = np.full(mdp.n_actions, np.nan, dtype=np.float64)
    probabilities = np.full((mdp.n_actions, mdp.n_loads), np.nan, dtype=np.float64)
    candidate_actions: list[int] = []
    candidate_probabilities: list[float] = []
    candidate_loads: list[int] = []
    candidate_queues: list[int] = []
    candidate_budgets: list[int] = []
    candidate_horizons: list[int] = []
    candidate_histories: list[CausalHistory] = []
    rewards = np.full(mdp.n_actions, np.nan, dtype=np.float64)
    for action in range(mdp.n_actions):
        cost = int(mdp.action_costs[action])
        if cost > budget:
            continue
        next_queue, reward, _ = mdp.outcome(queue, load, action)
        predicted = transition.predict_probabilities(t, load, action)
        probabilities[action] = predicted
        rewards[action] = float(reward)
        for next_load, probability in enumerate(predicted):
            next_history = history.advance(
                action,
                int(mdp.action_capacity[action]),
                int(mdp.load_arrivals[next_load]),
                next_queue,
            )
            candidate_actions.append(action)
            candidate_probabilities.append(float(probability))
            candidate_loads.append(next_load)
            candidate_queues.append(next_queue)
            candidate_budgets.append(budget - cost)
            candidate_horizons.append(remaining_horizon - 1)
            candidate_histories.append(next_history)
    values = predictor.predict_many(
        np.asarray(candidate_loads),
        np.asarray(candidate_queues),
        np.asarray(candidate_budgets),
        np.asarray(candidate_horizons),
        candidate_histories,
    )
    continuation = np.zeros(mdp.n_actions, dtype=np.float64)
    np.add.at(
        continuation,
        np.asarray(candidate_actions, dtype=np.int64),
        np.asarray(candidate_probabilities) * values,
    )
    feasible = np.flatnonzero(np.isfinite(rewards))
    q_values[feasible] = rewards[feasible] + mdp.config.gamma * continuation[feasible]
    if not np.isfinite(q_values).any():
        raise RuntimeError("no feasible action was scored")
    return ContextPlanResult(int(np.nanargmax(q_values)), q_values, probabilities)


class ContextPlanningAgent(nn.Module):
    def __init__(
        self,
        mdp: ActionConditionedBudgetMDP,
        predictor: ContextValuePredictor,
        transition: StructuredActionEffectModel,
    ):
        super().__init__()
        self.mdp = mdp
        self.predictor = predictor
        self.transition = transition
        self.history: CausalHistory | None = None
        self._pending_action: int | None = None

    def reset_budget_controller(self) -> None:
        self.history = None
        self._pending_action = None

    def _decode(self, row: torch.Tensor) -> tuple[int, int, int, int]:
        t = int(
            torch.round((1.0 - row[-1]) * self.mdp.config.horizon)
            .clamp(0, self.mdp.config.horizon - 1)
            .item()
        )
        budget = int(
            torch.round(row[-2] * self.mdp.config.max_budget)
            .clamp(0, self.mdp.config.max_budget)
            .item()
        )
        queue = int(torch.round(row[2]).clamp(0, self.mdp.config.max_queue).item())
        arrival_levels = torch.as_tensor(self.mdp.load_arrivals, dtype=row.dtype)
        load = int(torch.argmin((row[0] - arrival_levels).abs()).item())
        return t, load, queue, budget

    def act(self, observation: torch.Tensor, deterministic: bool = True, **kwargs):
        del deterministic, kwargs
        if observation.ndim != 2 or observation.shape != (1, 14):
            raise ValueError("stateful context planning requires one [1, 14] observation")
        row = observation.detach().cpu()[0]
        t, load, queue, budget = self._decode(row)
        arrival = int(self.mdp.load_arrivals[load])
        if t == 0:
            self.history = CausalHistory((float(arrival),), (float(queue),))
            self._pending_action = None
        elif self.history is None or self._pending_action is None:
            raise RuntimeError("context history was not initialized at the episode start")
        elif len(self.history.arrivals) == t:
            action = self._pending_action
            self.history = self.history.advance(
                action,
                int(self.mdp.action_capacity[action]),
                arrival,
                queue,
            )
        if self.history is None or len(self.history.arrivals) != t + 1:
            raise RuntimeError("context history is not aligned with the current decision time")
        plan = context_one_step_plan(
            self.mdp,
            self.predictor,
            self.transition,
            self.history,
            t,
            load,
            queue,
            budget,
        )
        self._pending_action = plan.action
        return ContextPlanningOutput(
            action=torch.as_tensor([plan.action], dtype=torch.long, device=observation.device),
            q_values=torch.as_tensor(
                plan.q_values, dtype=observation.dtype, device=observation.device
            ).unsqueeze(0),
        )
