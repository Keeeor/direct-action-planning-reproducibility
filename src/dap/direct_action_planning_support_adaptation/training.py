from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from dap.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
)
from dap.direct_action_planning.planning import BudgetValueTable

from .models import FrozenValueWithAdapter, LinearValueAdapter, PooledValueNetwork


@dataclass(frozen=True)
class ValueLossWeights:
    value: float = 1.0
    anchor: float = 0.0
    mono: float = 0.0
    rank: float = 0.0


@dataclass(frozen=True)
class PooledTrainingConfig:
    learning_rate: float = 0.004
    max_epochs: int = 220
    patience: int = 30
    rank_margin: float = 0.002
    seed: int = 0


@dataclass(frozen=True)
class AdapterTrainingConfig:
    learning_rate: float = 0.02
    max_epochs: int = 160
    patience: int = 25
    anchor_weight: float = 0.25
    rank_weight: float = 1.0
    rank_margin: float = 0.002
    seed: int = 0


@dataclass(frozen=True)
class TrainedPooledValue:
    model: PooledValueNetwork
    history: pd.DataFrame
    best_epoch: int
    validation_metrics: dict[str, float]


@dataclass(frozen=True)
class TrainedAdapter:
    model: FrozenValueWithAdapter
    history: pd.DataFrame
    best_epoch: int
    variant: str


def _model_shape(model: nn.Module) -> tuple[int, int, int, int]:
    base = model.base if isinstance(model, FrozenValueWithAdapter) else model
    if not isinstance(base, PooledValueNetwork):
        raise TypeError("value model must expose the PooledValueNetwork grid")
    return base.horizon, base.n_loads, base.max_queue, base.max_budget


def _normalized(model: nn.Module, frame: pd.DataFrame) -> torch.Tensor:
    horizon, n_loads, max_queue, max_budget = _model_shape(model)
    return torch.as_tensor(
        np.stack(
            [
                frame.remaining_horizon.to_numpy(dtype=np.float64) / max(horizon, 1),
                frame.load.to_numpy(dtype=np.float64) / max(n_loads - 1, 1),
                frame.queue.to_numpy(dtype=np.float64) / max(max_queue, 1),
                frame.remaining_budget.to_numpy(dtype=np.float64) / max(max_budget, 1),
            ],
            axis=1,
        ),
        dtype=torch.float64,
    )


def _targets(frame: pd.DataFrame, column: str = "target_value") -> torch.Tensor:
    return torch.as_tensor(
        frame[column].to_numpy(dtype=np.float64, copy=True), dtype=torch.float64
    )


def _monotonic_loss(model: nn.Module, frame: pd.DataFrame) -> torch.Tensor:
    _, _, _, max_budget = _model_shape(model)
    pairs = frame[frame.remaining_budget < max_budget].drop_duplicates(
        ["remaining_horizon", "load", "queue", "remaining_budget"]
    )
    if pairs.empty:
        return next(model.parameters()).sum() * 0.0
    higher = pairs.copy()
    higher["remaining_budget"] += 1
    lower_value = model(_normalized(model, pairs))
    higher_value = model(_normalized(model, higher))
    return torch.mean(torch.relu(lower_value - higher_value).square())


def _planned_q(
    model: nn.Module,
    mdp: ActionConditionedBudgetMDP,
    states: pd.DataFrame,
) -> torch.Tensor:
    states = states.reset_index(drop=True)
    q_values: list[torch.Tensor] = []
    for action in range(mdp.n_actions):
        costs = int(mdp.action_costs[action])
        feasible = states.remaining_budget.to_numpy(np.int64) >= costs
        rewards = np.zeros(len(states), dtype=np.float64)
        next_queues = np.zeros(len(states), dtype=np.int64)
        probabilities = np.zeros((len(states), mdp.n_loads), dtype=np.float64)
        for index, row in enumerate(states.itertuples(index=False)):
            next_queue, reward, _ = mdp.outcome(int(row.queue), int(row.load), action)
            rewards[index] = reward
            next_queues[index] = next_queue
            probabilities[index] = mdp.load_probabilities(int(row.t), int(row.load))
        continuation = torch.zeros(len(states), dtype=torch.float64)
        for next_load in range(mdp.n_loads):
            next_states = states.copy()
            next_states["remaining_horizon"] = np.maximum(
                next_states.remaining_horizon.to_numpy(np.int64) - 1, 0
            )
            next_states["load"] = next_load
            next_states["queue"] = next_queues
            next_states["remaining_budget"] = np.maximum(
                next_states.remaining_budget.to_numpy(np.int64) - costs, 0
            )
            continuation = continuation + torch.as_tensor(
                probabilities[:, next_load], dtype=torch.float64
            ) * model(_normalized(model, next_states))
        q = torch.as_tensor(rewards, dtype=torch.float64) + mdp.config.gamma * continuation
        q_values.append(
            torch.where(
                torch.as_tensor(feasible), q, torch.full_like(q, -torch.inf)
            )
        )
    return torch.stack(q_values, dim=1)


def _rank_loss(
    model: nn.Module,
    rank_frames: dict[str, tuple[ActionConditionedBudgetMDP, pd.DataFrame]],
    margin: float,
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    pair_count = 0
    for mdp, states in rank_frames.values():
        if states.empty:
            continue
        predicted = _planned_q(model, mdp, states)
        target = torch.as_tensor(
            states[[f"q_star_a{action}" for action in range(mdp.n_actions)]].to_numpy(
                dtype=np.float64, copy=True
            ),
            dtype=torch.float64,
        )
        for better in range(mdp.n_actions):
            for worse in range(mdp.n_actions):
                valid = torch.isfinite(target[:, better]) & torch.isfinite(target[:, worse])
                valid &= target[:, better] > target[:, worse] + 1.0e-10
                if torch.any(valid):
                    current = torch.relu(
                        float(margin) - (predicted[valid, better] - predicted[valid, worse])
                    )
                    losses.append(current)
                    pair_count += int(current.numel())
    if not losses:
        return next(model.parameters()).sum() * 0.0, 0
    return torch.mean(torch.cat(losses)), pair_count


@torch.no_grad()
def _selection_metrics(
    model: nn.Module,
    validation: pd.DataFrame,
    rank_frames: dict[str, tuple[ActionConditionedBudgetMDP, pd.DataFrame]],
) -> dict[str, float]:
    prediction = model(_normalized(model, validation))
    value_mae = float(torch.mean(torch.abs(prediction - _targets(validation))))
    regrets: list[np.ndarray] = []
    agreements: list[np.ndarray] = []
    for mdp, states in rank_frames.values():
        planned = _planned_q(model, mdp, states).cpu().numpy()
        actions = np.argmax(planned, axis=1)
        target = states[[f"q_star_a{i}" for i in range(mdp.n_actions)]].to_numpy(float)
        optimal = np.nanargmax(target, axis=1)
        selected = target[np.arange(len(states)), actions]
        best = target[np.arange(len(states)), optimal]
        regrets.append(best - selected)
        agreements.append(actions == optimal)
    return {
        "validation_value_mae": value_mae,
        "validation_q_star_regret": float(np.mean(np.concatenate(regrets))) if regrets else 0.0,
        "validation_action_agreement": float(np.mean(np.concatenate(agreements)))
        if agreements
        else 1.0,
    }


def train_pooled_value(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    rank_frames: dict[str, tuple[ActionConditionedBudgetMDP, pd.DataFrame]],
    validation_rank_frames: dict[
        str, tuple[ActionConditionedBudgetMDP, pd.DataFrame]
    ] | None = None,
    anchor_states: pd.DataFrame,
    model: PooledValueNetwork,
    weights: ValueLossWeights,
    config: PooledTrainingConfig,
) -> TrainedPooledValue:
    if training.empty or validation.empty:
        raise ValueError("training and validation value rows are required")
    torch.manual_seed(config.seed)
    model = copy.deepcopy(model).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_x, train_y = _normalized(model, training), _targets(training)
    anchor_x = _normalized(model, anchor_states)
    anchor_column = "anchor_value" if "anchor_value" in anchor_states else "target_value"
    anchor_y = _targets(anchor_states, anchor_column)
    selection_rank_frames = validation_rank_frames if validation_rank_frames is not None else rank_frames
    initial = _selection_metrics(model, validation, selection_rank_frames)
    history: list[dict[str, float | int]] = [{"epoch": -1, **initial}]
    best_score = initial["validation_q_star_regret"] + 0.01 * initial["validation_value_mae"]
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(config.max_epochs):
        prediction = model(train_x)
        value_loss = torch.mean((prediction - train_y).square())
        zero = prediction.sum() * 0.0
        anchor_loss = (
            torch.mean((model(anchor_x) - anchor_y).square()) if weights.anchor > 0 else zero
        )
        mono_loss = _monotonic_loss(model, training) if weights.mono > 0 else zero
        rank_loss, pairs = (
            _rank_loss(model, rank_frames, config.rank_margin) if weights.rank > 0 else (zero, 0)
        )
        total = (
            weights.value * value_loss
            + weights.anchor * anchor_loss
            + weights.mono * mono_loss
            + weights.rank * rank_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        metrics = _selection_metrics(model, validation, selection_rank_frames)
        history.append(
            {
                "epoch": epoch,
                "total_loss": float(total.detach()),
                "value_loss": float(value_loss.detach()),
                "anchor_loss": float(anchor_loss.detach()),
                "mono_loss": float(mono_loss.detach()),
                "rank_loss": float(rank_loss.detach()),
                "rank_pairs": pairs,
                **metrics,
            }
        )
        score = metrics["validation_q_star_regret"] + 0.01 * metrics["validation_value_mae"]
        if score < best_score - 1.0e-10:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    final = _selection_metrics(model, validation, selection_rank_frames)
    history.append({"epoch": best_epoch, "selected": 1, **final})
    return TrainedPooledValue(model, pd.DataFrame(history), best_epoch, final)


def train_value_adapter(
    base: PooledValueNetwork,
    *,
    calibration: pd.DataFrame,
    anchors: pd.DataFrame,
    rank_frame: tuple[ActionConditionedBudgetMDP, pd.DataFrame] | None,
    variant: str,
    config: AdapterTrainingConfig,
) -> TrainedAdapter:
    allowed = {"regression", "regression_anchor", "regression_rank", "full"}
    if variant not in allowed:
        raise ValueError(f"unknown adaptation variant: {variant}")
    if calibration.empty:
        raise ValueError("non-zero adaptation requires calibration rows")
    torch.manual_seed(config.seed)
    model = FrozenValueWithAdapter(base, LinearValueAdapter(4)).double()
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=config.learning_rate)
    calibration_x = _normalized(model, calibration)
    calibration_y = _targets(calibration)
    anchor_x = _normalized(model, anchors)
    with torch.no_grad():
        anchor_y = model.base(anchor_x)
    rank_frames = {"target": rank_frame} if rank_frame is not None else {}

    def metrics() -> tuple[float, float]:
        with torch.no_grad():
            mae = float(torch.mean(torch.abs(model(calibration_x) - calibration_y)))
        rank_value = float(_rank_loss(model, rank_frames, config.rank_margin)[0].detach())
        return mae, rank_value

    initial_mae, initial_rank = metrics()
    history: list[dict[str, float | int]] = [
        {"epoch": -1, "calibration_value_mae": initial_mae, "rank_loss": initial_rank}
    ]
    best_score = initial_mae + config.rank_weight * initial_rank
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    use_anchor = variant in {"regression_anchor", "full"}
    use_rank = variant in {"regression_rank", "full"} and rank_frame is not None
    for epoch in range(config.max_epochs):
        prediction = model(calibration_x)
        value_loss = torch.mean((prediction - calibration_y).square())
        anchor_loss = torch.mean((model(anchor_x) - anchor_y).square())
        rank_loss, pairs = (
            _rank_loss(model, rank_frames, config.rank_margin)
            if use_rank
            else (prediction.sum() * 0.0, 0)
        )
        total = value_loss
        if use_anchor:
            total = total + config.anchor_weight * anchor_loss
        if use_rank:
            total = total + config.rank_weight * rank_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        mae, rank_metric = metrics()
        history.append(
            {
                "epoch": epoch,
                "total_loss": float(total.detach()),
                "value_loss": float(value_loss.detach()),
                "anchor_loss": float(anchor_loss.detach()),
                "rank_loss": float(rank_loss.detach()),
                "rank_pairs": pairs,
                "calibration_value_mae": mae,
            }
        )
        score = mae + config.rank_weight * rank_metric
        if score < best_score - 1.0e-10:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    final_mae, final_rank = metrics()
    history.append(
        {
            "epoch": best_epoch,
            "selected": 1,
            "calibration_value_mae": final_mae,
            "rank_loss": final_rank,
        }
    )
    return TrainedAdapter(model, pd.DataFrame(history), best_epoch, variant)


@torch.no_grad()
def dense_value_table(
    model: nn.Module,
    mdp: ActionConditionedBudgetMDP,
    *,
    source: str,
) -> BudgetValueTable:
    shape = (
        mdp.config.horizon + 1,
        mdp.n_loads,
        mdp.config.max_queue + 1,
        mdp.config.max_budget + 1,
    )
    values = np.zeros(shape, dtype=np.float64)
    rows = []
    indices = []
    for h in range(1, mdp.config.horizon + 1):
        for load, queue, budget in np.ndindex(
            mdp.n_loads, mdp.config.max_queue + 1, mdp.config.max_budget + 1
        ):
            rows.append(
                {
                    "remaining_horizon": h,
                    "load": load,
                    "queue": queue,
                    "remaining_budget": budget,
                }
            )
            indices.append((h, load, queue, budget))
    frame = pd.DataFrame(rows)
    predictions = model(_normalized(model, frame)).cpu().numpy()
    for index, prediction in zip(indices, predictions):
        values[index] = prediction
    values[0] = 0.0
    return BudgetValueTable(values=values, source=source)
