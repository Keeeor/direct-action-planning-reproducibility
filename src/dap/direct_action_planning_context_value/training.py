from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from dap.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)
from dap.direct_action_planning.planning import BudgetValueTable
from dap.direct_action_planning_repair.model import StructuredActionEffectModel

from .history import CausalHistory, deserialize_history, history_features, history_sequence
from .model import ContextResidualValueModel


@dataclass(frozen=True)
class ContextLossWeights:
    value: float = 1.0
    anchor: float = 0.1
    mono: float = 0.2
    rank: float = 1.0


@dataclass(frozen=True)
class ContextTrainingConfig:
    hidden_dim: int = 48
    gru_hidden_dim: int = 24
    learning_rate: float = 0.003
    max_epochs: int = 240
    patience: int = 25
    batch_size: int = 512
    rank_states: int = 96
    validation_rank_states: int = 384
    validation_interval: int = 5
    rank_margin: float = 0.002
    seed: int = 0


@dataclass(frozen=True)
class TrainedContextValue:
    model: ContextResidualValueModel
    history: pd.DataFrame
    best_epoch: int
    selection_metrics: dict[str, float]
    initial_metrics: dict[str, float]
    loss_weights: ContextLossWeights
    training_config: ContextTrainingConfig
    mode: str
    window: int


@dataclass
class PreparedData:
    frame: pd.DataFrame
    histories: list[CausalHistory]
    features: np.ndarray | None
    sequences: np.ndarray | None
    masks: np.ndarray | None


@dataclass
class PlanningBatch:
    groups: int
    n_actions: int
    candidate_group: np.ndarray
    candidate_action: np.ndarray
    candidate_probability: np.ndarray
    remaining_horizon: np.ndarray
    load: np.ndarray
    queue: np.ndarray
    budget: np.ndarray
    base_value: np.ndarray
    features: np.ndarray | None
    sequences: np.ndarray | None
    masks: np.ndarray | None
    reward: np.ndarray
    feasible: np.ndarray
    target_q: np.ndarray
    target_value: np.ndarray
    target_action: np.ndarray


def _prepare(frame: pd.DataFrame, mode: str, window: int) -> PreparedData:
    reset = frame.reset_index(drop=True).copy()
    histories = [deserialize_history(row) for row in reset.itertuples(index=False)]
    features = None
    sequences = None
    masks = None
    if mode == "feature":
        features = np.stack([history_features(history, window) for history in histories])
    elif mode == "gru":
        pairs = [history_sequence(history, window) for history in histories]
        sequences = np.stack([pair[0] for pair in pairs])
        masks = np.stack([pair[1] for pair in pairs])
    return PreparedData(reset, histories, features, sequences, masks)


def _forward_prepared(
    model: ContextResidualValueModel,
    data: PreparedData,
    indices: np.ndarray,
    device: torch.device,
    *,
    budgets: np.ndarray | None = None,
    base_values: np.ndarray | None = None,
) -> torch.Tensor:
    frame = data.frame.iloc[indices]
    kwargs: dict[str, torch.Tensor] = {}
    if model.mode == "feature":
        kwargs["features"] = torch.as_tensor(
            data.features[indices], dtype=torch.float64, device=device  # type: ignore[index]
        )
    elif model.mode == "gru":
        kwargs["sequence"] = torch.as_tensor(
            data.sequences[indices], dtype=torch.float64, device=device  # type: ignore[index]
        )
        kwargs["mask"] = torch.as_tensor(
            data.masks[indices], dtype=torch.bool, device=device  # type: ignore[index]
        )
    budget_values = (
        frame.remaining_budget.to_numpy(np.int64, copy=True)
        if budgets is None
        else np.asarray(budgets, dtype=np.int64).copy()
    )
    base = (
        frame.base_value.to_numpy(float, copy=True)
        if base_values is None
        else np.asarray(base_values, dtype=float).copy()
    )
    return model(
        torch.as_tensor(frame.remaining_horizon.to_numpy(np.int64, copy=True), device=device),
        torch.as_tensor(frame.load.to_numpy(np.int64, copy=True), device=device),
        torch.as_tensor(frame.queue.to_numpy(np.int64, copy=True), device=device),
        torch.as_tensor(budget_values, device=device),
        torch.as_tensor(base, dtype=torch.float64, device=device),
        **kwargs,
    )


def _build_planning_batch(
    data: PreparedData,
    indices: np.ndarray,
    mode: str,
    window: int,
    mdps: dict[str, ActionConditionedBudgetMDP],
    optima: dict[str, ActionDPResult],
    base_tables: dict[str, BudgetValueTable],
    transitions: dict[str, StructuredActionEffectModel],
) -> PlanningBatch:
    rows = data.frame.iloc[indices].reset_index(drop=True)
    histories = [data.histories[int(index)] for index in indices]
    n_actions = next(iter(mdps.values())).n_actions
    candidate_group: list[int] = []
    candidate_action: list[int] = []
    candidate_probability: list[float] = []
    horizons: list[int] = []
    loads: list[int] = []
    queues: list[int] = []
    budgets: list[int] = []
    bases: list[float] = []
    candidate_histories: list[CausalHistory] = []
    reward = np.full((len(rows), n_actions), np.nan, dtype=np.float64)
    feasible = np.zeros((len(rows), n_actions), dtype=bool)
    target_q = np.full((len(rows), n_actions), np.nan, dtype=np.float64)
    target_value = np.empty(len(rows), dtype=np.float64)
    target_action = np.empty(len(rows), dtype=np.int64)

    def model_key(row) -> str:
        scenario = str(row.scenario)
        if hasattr(row, "model_seed"):
            candidate = f"{scenario}::{int(row.model_seed)}"
            if candidate in base_tables and candidate in transitions:
                return candidate
        return scenario

    for group, (row, history) in enumerate(zip(rows.itertuples(index=False), histories)):
        scenario = str(row.scenario)
        mdp, optimum = mdps[scenario], optima[scenario]
        key = model_key(row)
        base_table, transition = base_tables[key], transitions[key]
        state = (int(row.t), int(row.load), int(row.queue), int(row.remaining_budget))
        target_q[group] = optimum.q_values[state]
        target_value[group] = optimum.values[state]
        target_action[group] = optimum.actions[state]
        for action in range(mdp.n_actions):
            cost = int(mdp.action_costs[action])
            if cost > state[3]:
                continue
            feasible[group, action] = True
            next_queue, immediate, _ = mdp.outcome(state[2], state[1], action)
            reward[group, action] = immediate
            probabilities = transition.predict_probabilities(state[0], state[1], action)
            for next_load, probability in enumerate(probabilities):
                next_horizon = mdp.config.horizon - state[0] - 1
                next_budget = state[3] - cost
                next_history = history.advance(
                    action,
                    int(mdp.action_capacity[action]),
                    int(mdp.load_arrivals[next_load]),
                    next_queue,
                )
                candidate_group.append(group)
                candidate_action.append(action)
                candidate_probability.append(float(probability))
                horizons.append(next_horizon)
                loads.append(next_load)
                queues.append(next_queue)
                budgets.append(next_budget)
                bases.append(
                    base_table.predict(next_load, next_queue, next_budget, next_horizon)
                )
                candidate_histories.append(next_history)
    features = None
    sequences = None
    masks = None
    if mode == "feature":
        features = np.stack(
            [history_features(history, window) for history in candidate_histories]
        )
    elif mode == "gru":
        pairs = [history_sequence(history, window) for history in candidate_histories]
        sequences = np.stack([pair[0] for pair in pairs])
        masks = np.stack([pair[1] for pair in pairs])
    return PlanningBatch(
        groups=len(rows),
        n_actions=n_actions,
        candidate_group=np.asarray(candidate_group, dtype=np.int64),
        candidate_action=np.asarray(candidate_action, dtype=np.int64),
        candidate_probability=np.asarray(candidate_probability, dtype=np.float64),
        remaining_horizon=np.asarray(horizons, dtype=np.int64),
        load=np.asarray(loads, dtype=np.int64),
        queue=np.asarray(queues, dtype=np.int64),
        budget=np.asarray(budgets, dtype=np.int64),
        base_value=np.asarray(bases, dtype=np.float64),
        features=features,
        sequences=sequences,
        masks=masks,
        reward=reward,
        feasible=feasible,
        target_q=target_q,
        target_value=target_value,
        target_action=target_action,
    )


def _planning_q(
    model: ContextResidualValueModel,
    batch: PlanningBatch,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    kwargs: dict[str, torch.Tensor] = {}
    if model.mode == "feature":
        kwargs["features"] = torch.as_tensor(batch.features, dtype=torch.float64, device=device)
    elif model.mode == "gru":
        kwargs["sequence"] = torch.as_tensor(
            batch.sequences, dtype=torch.float64, device=device
        )
        kwargs["mask"] = torch.as_tensor(batch.masks, dtype=torch.bool, device=device)
    values = model(
        torch.as_tensor(batch.remaining_horizon, device=device),
        torch.as_tensor(batch.load, device=device),
        torch.as_tensor(batch.queue, device=device),
        torch.as_tensor(batch.budget, device=device),
        torch.as_tensor(batch.base_value, dtype=torch.float64, device=device),
        **kwargs,
    )
    group = torch.as_tensor(batch.candidate_group, dtype=torch.long, device=device)
    action = torch.as_tensor(batch.candidate_action, dtype=torch.long, device=device)
    probability = torch.as_tensor(
        batch.candidate_probability, dtype=torch.float64, device=device
    )
    flat_index = group * batch.n_actions + action
    continuation = torch.zeros(
        batch.groups * batch.n_actions, dtype=torch.float64, device=device
    )
    continuation.index_add_(0, flat_index, probability * values)
    continuation = continuation.reshape(batch.groups, batch.n_actions)
    reward = torch.as_tensor(batch.reward, dtype=torch.float64, device=device)
    q_values = reward + float(gamma) * continuation
    feasible = torch.as_tensor(batch.feasible, dtype=torch.bool, device=device)
    return torch.where(feasible, q_values, torch.full_like(q_values, -torch.inf))


def _rank_loss(
    predicted_q: torch.Tensor,
    target_q: np.ndarray,
    margin: float,
) -> tuple[torch.Tensor, int]:
    target = torch.as_tensor(target_q, dtype=torch.float64, device=predicted_q.device)
    losses: list[torch.Tensor] = []
    for better in range(predicted_q.shape[1]):
        for worse in range(predicted_q.shape[1]):
            valid = torch.isfinite(target[:, better]) & torch.isfinite(target[:, worse])
            valid &= target[:, better] > target[:, worse] + 1.0e-10
            if torch.any(valid):
                losses.append(
                    torch.relu(
                        float(margin)
                        - (predicted_q[valid, better] - predicted_q[valid, worse])
                    )
                )
    if not losses:
        return predicted_q[torch.isfinite(predicted_q)].sum() * 0.0, 0
    combined = torch.cat(losses)
    return combined.mean(), int(combined.numel())


@torch.no_grad()
def _selection_metrics(
    model: ContextResidualValueModel,
    data: PreparedData,
    planning: PlanningBatch,
    gamma: float,
    device: torch.device,
) -> dict[str, float]:
    indices = np.arange(len(data.frame))
    predicted = _forward_prepared(model, data, indices, device).cpu().numpy()
    target = data.frame.target_value.to_numpy(float)
    q_values = _planning_q(model, planning, gamma, device).cpu().numpy()
    actions = np.argmax(q_values, axis=1)
    selected_q = planning.target_q[np.arange(planning.groups), actions]
    regrets = planning.target_value - selected_q
    pair_correct: list[bool] = []
    for left in range(planning.n_actions):
        for right in range(left + 1, planning.n_actions):
            valid = (
                np.isfinite(planning.target_q[:, left])
                & np.isfinite(planning.target_q[:, right])
                & np.isfinite(q_values[:, left])
                & np.isfinite(q_values[:, right])
            )
            true_delta = np.zeros(planning.groups, dtype=float)
            pred_delta = np.zeros(planning.groups, dtype=float)
            true_delta[valid] = (
                planning.target_q[valid, left] - planning.target_q[valid, right]
            )
            pred_delta[valid] = q_values[valid, left] - q_values[valid, right]
            valid &= np.abs(true_delta) > 1e-10
            pair_correct.extend((true_delta[valid] * pred_delta[valid] > 0).tolist())
    return {
        "value_mae": float(np.mean(np.abs(predicted - target))),
        "q_star_regret": float(np.mean(regrets)),
        "action_consistency": float(np.mean(actions == planning.target_action)),
        "pair_ranking_accuracy": float(np.mean(pair_correct)) if pair_correct else float("nan"),
    }


def train_context_value(
    mode: str,
    window: int,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    mdps: dict[str, ActionConditionedBudgetMDP],
    optima: dict[str, ActionDPResult],
    base_tables: dict[str, BudgetValueTable],
    transitions: dict[str, StructuredActionEffectModel],
    loss_weights: ContextLossWeights,
    config: ContextTrainingConfig,
    source_weights: dict[str, float],
    device: torch.device,
) -> TrainedContextValue:
    if mode not in {"current", "feature", "gru"}:
        raise ValueError("unsupported context mode")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    train_data = _prepare(training, mode, window)
    validation_data = _prepare(validation, mode, window)
    feature_mean = None
    feature_std = None
    if mode == "feature":
        feature_mean = train_data.features.mean(axis=0)  # type: ignore[union-attr]
        feature_std = train_data.features.std(axis=0)  # type: ignore[union-attr]
    first_mdp = next(iter(mdps.values()))
    model = ContextResidualValueModel(
        mode,
        first_mdp.config.horizon,
        first_mdp.n_loads,
        first_mdp.config.max_queue,
        first_mdp.config.max_budget,
        feature_dim=24,
        hidden_dim=config.hidden_dim,
        gru_hidden_dim=config.gru_hidden_dim,
        feature_mean=feature_mean,
        feature_std=feature_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    counts = training.source.value_counts().to_dict()
    probabilities = np.asarray(
        [float(source_weights.get(str(source), 0.0)) / max(counts[str(source)], 1) for source in training.source],
        dtype=float,
    )
    if probabilities.sum() <= 0:
        raise ValueError("source weights assign no training probability")
    probabilities /= probabilities.sum()
    train_rank_indices = rng.choice(
        len(training), size=min(config.rank_states, len(training)), replace=False
    )
    validation_rank_indices = rng.choice(
        len(validation), size=min(config.validation_rank_states, len(validation)), replace=False
    )
    train_planning = _build_planning_batch(
        train_data,
        train_rank_indices,
        mode,
        window,
        mdps,
        optima,
        base_tables,
        transitions,
    )
    validation_planning = _build_planning_batch(
        validation_data,
        validation_rank_indices,
        mode,
        window,
        mdps,
        optima,
        base_tables,
        transitions,
    )
    gamma = float(first_mdp.config.gamma)
    initial = _selection_metrics(
        model, validation_data, validation_planning, gamma, device
    )
    best_metrics = dict(initial)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    stale = 0
    history_rows: list[dict[str, float | int]] = []
    for epoch in range(config.max_epochs):
        batch_indices = rng.choice(
            len(training),
            size=min(config.batch_size, len(training)),
            replace=True,
            p=probabilities,
        )
        predicted = _forward_prepared(model, train_data, batch_indices, device)
        target = torch.as_tensor(
            training.iloc[batch_indices].target_value.to_numpy(float, copy=True),
            dtype=torch.float64,
            device=device,
        )
        value_loss = torch.mean((predicted - target).square())
        source = training.iloc[batch_indices].source.to_numpy(str)
        anchor_mask_np = np.isin(source, ["D0", "D1"])
        zero = predicted.sum() * 0.0
        anchor_loss = zero
        if loss_weights.anchor > 0 and anchor_mask_np.any():
            anchor_target = torch.as_tensor(
                training.iloc[batch_indices]
                .base_value.to_numpy(float, copy=True)[anchor_mask_np]
                .copy(),
                dtype=torch.float64,
                device=device,
            )
            anchor_loss = torch.mean(
                (predicted[torch.as_tensor(anchor_mask_np, device=device)] - anchor_target).square()
            )
        mono_loss = zero
        if loss_weights.mono > 0:
            batch_frame = training.iloc[batch_indices]
            mono_mask = batch_frame.remaining_budget.to_numpy(int) < first_mdp.config.max_budget
            if mono_mask.any():
                mono_indices = batch_indices[mono_mask]
                budgets_plus = training.iloc[mono_indices].remaining_budget.to_numpy(int) + 1
                bases_plus = np.asarray(
                    [
                        base_tables[
                            (
                                f"{row.scenario}::{int(row.model_seed)}"
                                if hasattr(row, "model_seed")
                                and f"{row.scenario}::{int(row.model_seed)}" in base_tables
                                else str(row.scenario)
                            )
                        ].predict(
                            int(row.load),
                            int(row.queue),
                            int(row.remaining_budget) + 1,
                            int(row.remaining_horizon),
                        )
                        for row in training.iloc[mono_indices].itertuples(index=False)
                    ]
                )
                value_plus = _forward_prepared(
                    model,
                    train_data,
                    mono_indices,
                    device,
                    budgets=budgets_plus,
                    base_values=bases_plus,
                )
                value_current = _forward_prepared(model, train_data, mono_indices, device)
                mono_loss = torch.mean(torch.relu(value_current - value_plus).square())
        predicted_q = _planning_q(model, train_planning, gamma, device)
        rank_loss, rank_pairs = _rank_loss(
            predicted_q, train_planning.target_q, config.rank_margin
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
        metrics = None
        if epoch % config.validation_interval == 0 or epoch == config.max_epochs - 1:
            metrics = _selection_metrics(
                model, validation_data, validation_planning, gamma, device
            )
            improved = metrics["q_star_regret"] < best_metrics["q_star_regret"] - 1.0e-10
            tied_better = (
                abs(metrics["q_star_regret"] - best_metrics["q_star_regret"]) <= 1.0e-10
                and metrics["value_mae"] < best_metrics["value_mae"] - 1.0e-10
            )
            if improved or tied_better:
                best_metrics = dict(metrics)
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
        history_rows.append(
            {
                "epoch": epoch,
                "total_loss": float(total.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "anchor_loss": float(anchor_loss.detach().cpu()),
                "mono_loss": float(mono_loss.detach().cpu()),
                "rank_loss": float(rank_loss.detach().cpu()),
                "rank_pairs": rank_pairs,
                "validation_q_star_regret": (
                    float(metrics["q_star_regret"]) if metrics is not None else np.nan
                ),
                "validation_value_mae": (
                    float(metrics["value_mae"]) if metrics is not None else np.nan
                ),
            }
        )
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    final_metrics = _selection_metrics(
        model, validation_data, validation_planning, gamma, device
    )
    return TrainedContextValue(
        model,
        pd.DataFrame(history_rows),
        best_epoch,
        final_metrics,
        initial,
        loss_weights,
        config,
        mode,
        window,
    )


def context_checkpoint_payload(trained: TrainedContextValue) -> dict[str, object]:
    return {
        "state_dict": trained.model.state_dict(),
        "mode": trained.mode,
        "window": trained.window,
        "best_epoch": trained.best_epoch,
        "selection_metrics": trained.selection_metrics,
        "initial_metrics": trained.initial_metrics,
        "loss_weights": asdict(trained.loss_weights),
        "training_config": asdict(trained.training_config),
        "model_config": {
            "horizon": trained.model.horizon,
            "n_loads": trained.model.n_loads,
            "max_queue": trained.model.max_queue,
            "max_budget": trained.model.max_budget,
            "feature_dim": trained.model.feature_dim,
        },
    }
