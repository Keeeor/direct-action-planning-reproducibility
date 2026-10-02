from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from dap.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
)
from dap.direct_action_planning.planning import BudgetValueTable

from .model import StructuredActionEffectModel


@dataclass(frozen=True)
class StructuredPlanResult:
    action: int
    q_values: np.ndarray
    next_load_probabilities: np.ndarray
    model_q_variance: np.ndarray | None = None
    ranking_disagreement: float = 0.0
    fallback_used: bool = False


@dataclass(frozen=True)
class StructuredPlanningOutput:
    action: torch.Tensor
    q_values: torch.Tensor


def structured_one_step_plan(
    mdp: ActionConditionedBudgetMDP,
    value: BudgetValueTable,
    model: StructuredActionEffectModel,
    t: int,
    load: int,
    queue: int,
    budget: int,
) -> StructuredPlanResult:
    remaining_horizon = mdp.config.horizon - t
    q_values = np.full(mdp.n_actions, np.nan, dtype=np.float64)
    probabilities = np.full((mdp.n_actions, mdp.n_loads), np.nan, dtype=np.float64)
    for action in range(mdp.n_actions):
        cost = int(mdp.action_costs[action])
        if cost > budget:
            continue
        next_queue, reward, _ = mdp.outcome(queue, load, action)
        predicted = model.predict_probabilities(t, load, action)
        probabilities[action] = predicted
        continuation = sum(
            float(probability)
            * value.predict(
                next_load,
                next_queue,
                budget - cost,
                remaining_horizon - 1,
            )
            for next_load, probability in enumerate(predicted)
        )
        q_values[action] = float(reward) + mdp.config.gamma * continuation
    return StructuredPlanResult(
        action=int(np.nanargmax(q_values)),
        q_values=q_values,
        next_load_probabilities=probabilities,
    )


class StructuredPlanningAgent(nn.Module):
    def __init__(
        self,
        mdp: ActionConditionedBudgetMDP,
        value: BudgetValueTable,
        model: StructuredActionEffectModel,
    ):
        super().__init__()
        self.mdp = mdp
        self.value = value
        self.model = model
        self._plan_cache: dict[tuple[int, int, int, int], StructuredPlanResult] = {}

    def reset_budget_controller(self) -> None:
        return None

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
        arrivals = torch.as_tensor(self.mdp.load_arrivals, dtype=row.dtype)
        load = int(torch.argmin((row[0] - arrivals).abs()).item())
        return t, load, queue, budget

    def act(self, observation: torch.Tensor, deterministic: bool = True, **kwargs):
        del deterministic, kwargs
        if observation.ndim != 2 or observation.shape[-1] != 14:
            raise ValueError("canonical observation must have shape [batch, 14]")
        actions, q_values = [], []
        for row in observation.detach().cpu():
            state = self._decode(row)
            if state not in self._plan_cache:
                self._plan_cache[state] = structured_one_step_plan(
                    self.mdp, self.value, self.model, *state
                )
            plan = self._plan_cache[state]
            actions.append(plan.action)
            q_values.append(plan.q_values)
        return StructuredPlanningOutput(
            action=torch.as_tensor(actions, dtype=torch.long, device=observation.device),
            q_values=torch.as_tensor(
                np.stack(q_values), dtype=observation.dtype, device=observation.device
            ),
        )


def ensemble_one_step_plan(
    mdp: ActionConditionedBudgetMDP,
    value: BudgetValueTable,
    models: list[StructuredActionEffectModel],
    t: int,
    load: int,
    queue: int,
    budget: int,
    uncertainty_multiplier: float = 1.0,
    minimum_margin: float = 0.02,
    fallback_action: int | None = None,
) -> StructuredPlanResult:
    if len(models) < 3:
        raise ValueError("an ensemble requires at least three models")
    member_q = np.stack(
        [
            structured_one_step_plan(mdp, value, model, t, load, queue, budget).q_values
            for model in models
        ]
    )
    feasible = np.flatnonzero(np.isfinite(member_q[0]))
    mean_q = np.full(mdp.n_actions, np.nan, dtype=np.float64)
    variance = np.full(mdp.n_actions, np.nan, dtype=np.float64)
    mean_q[feasible] = np.mean(member_q[:, feasible], axis=0)
    variance[feasible] = np.var(member_q[:, feasible], axis=0)
    order = feasible[np.argsort(-mean_q[feasible], kind="stable")]
    selected = int(order[0])
    fallback_used = False
    if fallback_action is not None and len(order) > 1:
        top, second = int(order[0]), int(order[1])
        margin = float(mean_q[top] - mean_q[second])
        uncertainty = float(np.sqrt(variance[top] + variance[second]))
        confident = (
            margin >= minimum_margin
            and margin > uncertainty_multiplier * uncertainty
        )
        if not confident:
            selected = int(fallback_action)
            fallback_used = True
    member_actions = np.nanargmax(member_q, axis=1)
    disagreement = 1.0 - float(
        np.max(np.bincount(member_actions, minlength=mdp.n_actions)) / len(models)
    )
    return StructuredPlanResult(
        action=selected,
        q_values=mean_q,
        next_load_probabilities=np.empty((0, 0)),
        model_q_variance=variance,
        ranking_disagreement=disagreement,
        fallback_used=fallback_used,
    )


class EnsemblePlanningAgent(StructuredPlanningAgent):
    def __init__(
        self,
        mdp: ActionConditionedBudgetMDP,
        value: BudgetValueTable,
        models: list[StructuredActionEffectModel],
        fallback_agent=None,
        uncertainty_multiplier: float = 1.0,
        minimum_margin: float = 0.02,
    ):
        super().__init__(mdp, value, models[0])
        self.models = nn.ModuleList(models)
        self.fallback_agent = fallback_agent
        self.uncertainty_multiplier = float(uncertainty_multiplier)
        self.minimum_margin = float(minimum_margin)
        self.total_decisions = 0
        self.fallback_decisions = 0

    @property
    def fallback_rate(self) -> float:
        return self.fallback_decisions / max(self.total_decisions, 1)

    def act(self, observation: torch.Tensor, deterministic: bool = True, **kwargs):
        fallback_actions = None
        if self.fallback_agent is not None:
            with torch.no_grad():
                fallback_actions = self.fallback_agent.act(
                    observation, deterministic=True, **kwargs
                ).action.detach().cpu().numpy()
        actions, q_values = [], []
        for index, row in enumerate(observation.detach().cpu()):
            state = self._decode(row)
            fallback = int(fallback_actions[index]) if fallback_actions is not None else None
            cache_key = (*state, fallback if fallback is not None else -1)
            if cache_key not in self._plan_cache:
                self._plan_cache[cache_key] = ensemble_one_step_plan(
                    self.mdp,
                    self.value,
                    list(self.models),
                    *state,
                    uncertainty_multiplier=self.uncertainty_multiplier,
                    minimum_margin=self.minimum_margin,
                    fallback_action=fallback,
                )
            plan = self._plan_cache[cache_key]
            actions.append(plan.action)
            q_values.append(plan.q_values)
            self.total_decisions += 1
            self.fallback_decisions += int(plan.fallback_used)
        return StructuredPlanningOutput(
            action=torch.as_tensor(actions, dtype=torch.long, device=observation.device),
            q_values=torch.as_tensor(
                np.stack(q_values), dtype=observation.dtype, device=observation.device
            ),
        )
