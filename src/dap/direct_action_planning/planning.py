from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from dap.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
)

if TYPE_CHECKING:
    from .learning import EmpiricalActionModel


@dataclass(frozen=True)
class BudgetValueTable:
    """A value function indexed by horizon, load, queue and remaining budget."""

    values: np.ndarray
    source: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 4 or values.shape[0] < 2:
            raise ValueError("value table must have shape [horizon+1, load, queue, budget]")
        if not np.isfinite(values).all():
            raise ValueError("value table contains non-finite entries")
        object.__setattr__(self, "values", values)

    @classmethod
    def from_exact_dp(cls, time_indexed_values: np.ndarray) -> "BudgetValueTable":
        values = np.asarray(time_indexed_values, dtype=np.float64)
        if values.ndim != 4 or values.shape[0] < 2:
            raise ValueError("exact values must have shape [time+1, load, queue, budget]")
        return cls(values=values[::-1].copy(), source="exact_dp_v_star")

    def predict(
        self,
        load: int,
        queue: int,
        budget: int,
        remaining_horizon: int,
    ) -> float:
        return float(self.values[remaining_horizon, load, queue, budget])


@dataclass(frozen=True)
class PlanResult:
    action: int
    q_values: np.ndarray
    continuation_values: np.ndarray
    planning_costs: np.ndarray
    transition_source: str


@dataclass(frozen=True)
class DirectPlanningOutput:
    action: torch.Tensor
    q_values: torch.Tensor


class DirectPlanningAgent(nn.Module):
    """Canonical-observation adapter for direct one-step action planning."""

    def __init__(
        self,
        mdp: ActionConditionedBudgetMDP,
        value: BudgetValueTable,
        learned_model: "EmpiricalActionModel | None" = None,
    ):
        super().__init__()
        self.mdp = mdp
        self.value = value
        self.learned_model = learned_model

    def reset_budget_controller(self) -> None:
        return None

    def act(
        self,
        observation: torch.Tensor,
        deterministic: bool = True,
        **kwargs,
    ) -> DirectPlanningOutput:
        del deterministic, kwargs
        if observation.ndim != 2 or observation.shape[-1] != 14:
            raise ValueError("canonical observation must have shape [batch, 14]")
        cpu = observation.detach().to(device="cpu", dtype=torch.float64)
        load_values = torch.as_tensor(self.mdp.load_arrivals, dtype=torch.float64)
        actions: list[int] = []
        q_values: list[np.ndarray] = []
        for row in cpu:
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
            queue = int(
                torch.round(row[2]).clamp(0, self.mdp.config.max_queue).item()
            )
            load = int(torch.argmin((row[0] - load_values).abs()).item())
            plan = one_step_plan(
                self.mdp,
                self.value,
                t,
                load,
                queue,
                budget,
                learned_model=self.learned_model,
            )
            actions.append(plan.action)
            q_values.append(plan.q_values)
        return DirectPlanningOutput(
            action=torch.as_tensor(actions, dtype=torch.long, device=observation.device),
            q_values=torch.as_tensor(
                np.stack(q_values), dtype=observation.dtype, device=observation.device
            ),
        )


def _predicted_budget(cost: float, budget: int) -> int:
    if not np.isfinite(cost):
        raise ValueError("predicted cost must be finite")
    discrete_cost = int(np.rint(max(float(cost), 0.0)))
    return max(budget - discrete_cost, 0)


def one_step_plan(
    mdp: ActionConditionedBudgetMDP,
    value: BudgetValueTable,
    t: int,
    load: int,
    queue: int,
    budget: int,
    learned_model: "EmpiricalActionModel | None" = None,
) -> PlanResult:
    """Score all truly feasible actions and directly select the largest one-step value."""

    cfg = mdp.config
    if not 0 <= t < cfg.horizon:
        raise ValueError("time is outside the active horizon")
    if not 0 <= load < mdp.n_loads or not 0 <= queue <= cfg.max_queue:
        raise ValueError("load or queue is outside the state grid")
    if not 0 <= budget <= cfg.max_budget:
        raise ValueError("budget is outside the state grid")
    if value.values.shape != (
        cfg.horizon + 1,
        mdp.n_loads,
        cfg.max_queue + 1,
        cfg.max_budget + 1,
    ):
        raise ValueError("value table does not match the MDP grid")

    remaining_horizon = cfg.horizon - t
    q_values = np.full(mdp.n_actions, np.nan, dtype=np.float64)
    continuations = np.full(mdp.n_actions, np.nan, dtype=np.float64)
    planning_costs = np.full(mdp.n_actions, np.nan, dtype=np.float64)
    for action in range(mdp.n_actions):
        true_cost = int(mdp.action_costs[action])
        if true_cost > budget:
            continue
        if learned_model is None:
            next_queue, reward, _ = mdp.outcome(queue, load, action)
            planning_cost = float(true_cost)
            next_budget = budget - true_cost
            continuation = 0.0
            for next_load, probability in enumerate(mdp.load_probabilities(t, load)):
                continuation += float(probability) * value.predict(
                    next_load,
                    next_queue,
                    next_budget,
                    remaining_horizon - 1,
                )
            source = "true_transition"
        else:
            prediction = learned_model.predict(t, load, queue, action)
            reward = float(prediction.reward)
            planning_cost = float(prediction.cost)
            next_budget = _predicted_budget(planning_cost, budget)
            continuation = 0.0
            for next_load, next_queue in np.ndindex(
                prediction.next_state_probabilities.shape
            ):
                probability = prediction.next_state_probabilities[next_load, next_queue]
                continuation += float(probability) * value.predict(
                    next_load,
                    next_queue,
                    next_budget,
                    remaining_horizon - 1,
                )
            source = "learned_transition"
        continuations[action] = continuation
        planning_costs[action] = planning_cost
        q_values[action] = float(reward) + cfg.gamma * continuation

    if not np.isfinite(q_values).any():
        raise RuntimeError("no truly feasible action was scored")
    return PlanResult(
        action=int(np.nanargmax(q_values)),
        q_values=q_values,
        continuation_values=continuations,
        planning_costs=planning_costs,
        transition_source=source,
    )
