from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)
from stage2_dynamic_budget.direct_action_planning.planning import BudgetValueTable
from stage2_dynamic_budget.direct_action_planning_repair.model import (
    StructuredActionEffectModel,
)


@dataclass(frozen=True)
class ValueLossWeights:
    value: float = 1.0
    anchor: float = 0.0
    mono: float = 0.0
    rank: float = 0.0


@dataclass(frozen=True)
class ValueTrainingConfig:
    learning_rate: float = 0.08
    max_epochs: int = 500
    patience: int = 60
    rank_margin: float = 0.002
    seed: int = 0


@dataclass(frozen=True)
class TrainedValueModel:
    model: "ResidualBudgetValueModel"
    history: pd.DataFrame
    best_epoch: int
    selection_metrics: dict[str, float]
    loss_weights: ValueLossWeights
    training_config: ValueTrainingConfig


class ResidualBudgetValueModel(nn.Module):
    """Finite-grid V(s,b,h), initialized from the frozen Learned Value."""

    def __init__(self, fixed_value: BudgetValueTable):
        super().__init__()
        base = torch.as_tensor(fixed_value.values, dtype=torch.float64)
        self.register_buffer("base_values", base.clone())
        self.residual = nn.Parameter(torch.zeros_like(base))

    def forward(
        self,
        remaining_horizon: torch.Tensor,
        load: torch.Tensor,
        queue: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        return (self.base_values + self.residual)[
            remaining_horizon.long(), load.long(), queue.long(), budget.long()
        ]

    def value_grid(self) -> torch.Tensor:
        return self.base_values + self.residual

    def as_value_table(self, source: str) -> BudgetValueTable:
        values = self.value_grid().detach().cpu().numpy()
        return BudgetValueTable(values=values, source=source)


def budget_monotonic_loss(model: ResidualBudgetValueModel) -> torch.Tensor:
    grid = model.value_grid()
    violations = torch.relu(grid[..., :-1] - grid[..., 1:])
    return torch.mean(violations.square())


def _state_tensors(frame: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.as_tensor(
            frame[column].to_numpy(dtype=np.int64, copy=True),
            dtype=torch.long,
            device=device,
        )
        for column in ("t", "load", "queue", "remaining_budget")
    )


def planning_q_values(
    value_model: ResidualBudgetValueModel,
    transition: StructuredActionEffectModel,
    mdp: ActionConditionedBudgetMDP,
    states: pd.DataFrame,
) -> torch.Tensor:
    device = value_model.residual.device
    t, load, queue, budget = _state_tensors(states, device)
    horizon = mdp.config.horizon - t
    q_values: list[torch.Tensor] = []
    arrivals = torch.as_tensor(mdp.load_arrivals, dtype=torch.long, device=device)
    for action in range(mdp.n_actions):
        cost = int(mdp.action_costs[action])
        feasible = budget >= cost
        action_tensor = torch.full_like(t, action)
        probabilities = transition(t, load, action_tensor).to(dtype=torch.float64)
        available = queue + arrivals[load]
        served = torch.minimum(
            available,
            torch.full_like(available, int(mdp.action_capacity[action])),
        )
        next_queues = torch.clamp(available - served, min=0, max=mdp.config.max_queue)
        violations = next_queues >= max(3, mdp.config.max_queue // 2)
        rewards = (
            served.to(torch.float64)
            - mdp.config.queue_penalty * next_queues.to(torch.float64)
            - mdp.config.severe_queue_penalty * violations.to(torch.float64)
            - float(mdp.config.action_activation_penalty[action])
        )
        next_budget = torch.clamp(budget - cost, min=0)
        continuations = []
        for next_load in range(mdp.n_loads):
            continuations.append(
                value_model(
                    horizon - 1,
                    torch.full_like(load, next_load),
                    next_queues,
                    next_budget,
                )
            )
        continuation = torch.sum(probabilities * torch.stack(continuations, dim=1), dim=1)
        q = rewards + mdp.config.gamma * continuation
        q_values.append(torch.where(feasible, q, torch.full_like(q, -torch.inf)))
    return torch.stack(q_values, dim=1)


def planning_rank_loss(
    value_model: ResidualBudgetValueModel,
    transition: StructuredActionEffectModel,
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    states: pd.DataFrame,
    margin: float,
) -> tuple[torch.Tensor, int]:
    unique = states[["t", "load", "queue", "remaining_budget"]].drop_duplicates()
    predicted = planning_q_values(value_model, transition, mdp, unique)
    index = tuple(
        unique[column].to_numpy(np.int64)
        for column in ("t", "load", "queue", "remaining_budget")
    )
    target = torch.as_tensor(optimum.q_values[index], dtype=torch.float64, device=predicted.device)
    losses = []
    for better in range(mdp.n_actions):
        for worse in range(mdp.n_actions):
            valid = torch.isfinite(target[:, better]) & torch.isfinite(target[:, worse])
            valid &= target[:, better] > target[:, worse] + 1.0e-10
            if torch.any(valid):
                losses.append(
                    torch.relu(
                        float(margin)
                        - (predicted[valid, better] - predicted[valid, worse])
                    )
                )
    if not losses:
        return predicted.sum() * 0.0, 0
    combined = torch.cat(losses)
    return torch.mean(combined), int(combined.numel())


@torch.no_grad()
def value_selection_metrics(
    model: ResidualBudgetValueModel,
    transition: StructuredActionEffectModel,
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    states: pd.DataFrame,
) -> dict[str, float]:
    unique = states[["t", "load", "queue", "remaining_budget", "target_value"]].drop_duplicates(
        ["t", "load", "queue", "remaining_budget"]
    )
    device = model.residual.device
    t, load, queue, budget = _state_tensors(unique, device)
    horizon = mdp.config.horizon - t
    predicted_value = model(horizon, load, queue, budget)
    target_value = torch.as_tensor(
        unique.target_value.to_numpy(dtype=float, copy=True),
        dtype=torch.float64,
        device=device,
    )
    q_values = planning_q_values(model, transition, mdp, unique)
    actions = torch.argmax(q_values, dim=1).cpu().numpy()
    index = tuple(
        unique[column].to_numpy(np.int64)
        for column in ("t", "load", "queue", "remaining_budget")
    )
    optimal_actions = optimum.actions[index]
    selected_q = optimum.q_values[index + (actions,)]
    regrets = optimum.values[index] - selected_q
    return {
        "value_mae": float(torch.mean(torch.abs(predicted_value - target_value)).cpu()),
        "action_consistency": float(np.mean(actions == optimal_actions)),
        "q_star_regret": float(np.mean(regrets)),
        "monotonic_violation_rate": float(
            torch.mean((model.value_grid()[..., :-1] > model.value_grid()[..., 1:]).double()).cpu()
        ),
    }


def train_refreshed_value(
    fixed_value: BudgetValueTable,
    transition: StructuredActionEffectModel,
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    training_states: pd.DataFrame,
    validation_states: pd.DataFrame,
    anchors: pd.DataFrame,
    loss_weights: ValueLossWeights,
    config: ValueTrainingConfig,
    source_weights: dict[str, float] | None = None,
    device: torch.device | str = "cpu",
) -> TrainedValueModel:
    device = torch.device(device)
    torch.manual_seed(config.seed)
    model = ResidualBudgetValueModel(fixed_value).to(device)
    transition = copy.deepcopy(transition).to(device).eval()
    for parameter in transition.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam([model.residual], lr=config.learning_rate)
    t, load, queue, budget = _state_tensors(training_states, device)
    horizon = mdp.config.horizon - t
    target = torch.as_tensor(
        training_states.target_value.to_numpy(dtype=float, copy=True),
        dtype=torch.float64,
        device=device,
    )
    if source_weights:
        source_counts = training_states.source.value_counts().to_dict()
        sample_weight = torch.as_tensor(
            [
                float(source_weights.get(str(source), 1.0))
                / max(int(source_counts[str(source)]), 1)
                for source in training_states.source
            ],
            dtype=torch.float64,
            device=device,
        )
    else:
        sample_weight = torch.ones(len(training_states), dtype=torch.float64, device=device)
    anchor_t, anchor_load, anchor_queue, anchor_budget = _state_tensors(anchors, device)
    anchor_horizon = mdp.config.horizon - anchor_t
    with torch.no_grad():
        anchor_target = torch.as_tensor(
            [
                fixed_value.predict(int(l), int(q), int(b), int(h))
                for l, q, b, h in zip(
                    anchor_load, anchor_queue, anchor_budget, anchor_horizon
                )
            ],
            dtype=torch.float64,
            device=device,
        )

    initial = value_selection_metrics(model, transition, mdp, optimum, validation_states)
    best_score = initial["q_star_regret"]
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(config.max_epochs):
        predicted = model(horizon, load, queue, budget)
        value_loss = torch.sum(sample_weight * (predicted - target).square()) / torch.sum(
            sample_weight
        )
        zero = predicted.sum() * 0.0
        anchor_loss = zero
        if loss_weights.anchor > 0.0:
            anchor_prediction = model(anchor_horizon, anchor_load, anchor_queue, anchor_budget)
            anchor_loss = torch.mean((anchor_prediction - anchor_target).square())
        mono_loss = budget_monotonic_loss(model) if loss_weights.mono > 0.0 else zero
        rank_loss, rank_pairs = zero, 0
        if loss_weights.rank > 0.0:
            rank_loss, rank_pairs = planning_rank_loss(
                model, transition, mdp, optimum, training_states, config.rank_margin
            )
        total = (
            loss_weights.value * value_loss
            + loss_weights.anchor * anchor_loss
            + loss_weights.mono * mono_loss
            + loss_weights.rank * rank_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        with torch.no_grad():
            model.residual[0].zero_()
        metrics = value_selection_metrics(model, transition, mdp, optimum, validation_states)
        history.append(
            {
                "epoch": epoch,
                "total_loss": float(total.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "anchor_loss": float(anchor_loss.detach().cpu()),
                "mono_loss": float(mono_loss.detach().cpu()),
                "rank_loss": float(rank_loss.detach().cpu()),
                "rank_pairs": rank_pairs,
                **metrics,
            }
        )
        score = metrics["q_star_regret"]
        if score < best_score - 1.0e-10 or (
            abs(score - best_score) <= 1.0e-10
            and metrics["value_mae"] < initial["value_mae"]
        ):
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            initial = metrics
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    final = value_selection_metrics(model, transition, mdp, optimum, validation_states)
    return TrainedValueModel(
        model=model,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        selection_metrics=final,
        loss_weights=loss_weights,
        training_config=config,
    )


def value_checkpoint_payload(trained: TrainedValueModel) -> dict[str, object]:
    return {
        "state_dict": trained.model.state_dict(),
        "best_epoch": trained.best_epoch,
        "selection_metrics": trained.selection_metrics,
        "loss_weights": asdict(trained.loss_weights),
        "training_config": asdict(trained.training_config),
    }
