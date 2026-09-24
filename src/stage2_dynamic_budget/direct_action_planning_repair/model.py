from __future__ import annotations

from dataclasses import asdict, dataclass
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    state: float = 1.0
    effect: float = 0.0
    q: float = 0.0
    rank: float = 0.0
    rank_margin: float = 0.02


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int = 32
    learning_rate: float = 0.01
    max_epochs: int = 300
    patience: int = 40
    seed: int = 0
    use_priority_weights: bool = False


@dataclass(frozen=True)
class TrainedStructuredModel:
    model: "StructuredActionEffectModel"
    history: pd.DataFrame
    best_epoch: int
    selection_metrics: dict[str, float]
    loss_weights: LossWeights
    training_config: TrainingConfig


class StructuredActionEffectModel(nn.Module):
    """Predict only an exogenous-load base distribution plus an a0-anchored effect."""

    def __init__(self, horizon: int, n_loads: int, n_actions: int, hidden_dim: int = 32):
        super().__init__()
        self.horizon = int(horizon)
        self.n_loads = int(n_loads)
        self.n_actions = int(n_actions)
        input_dim = self.horizon + self.n_loads
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.n_loads),
        )
        self.effect = nn.Sequential(
            nn.Linear(input_dim + self.n_actions, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.n_loads),
        )

    def _features(self, t: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                F.one_hot(t.long(), self.horizon).to(torch.float32),
                F.one_hot(load.long(), self.n_loads).to(torch.float32),
            ],
            dim=-1,
        )

    def logits(self, t: torch.Tensor, load: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = self._features(t, load)
        base_logits = self.base(features)
        action_one_hot = F.one_hot(action.long(), self.n_actions).to(torch.float32)
        delta = self.effect(torch.cat([features, action_one_hot], dim=-1))
        anchored = delta * (action != 0).to(torch.float32).unsqueeze(-1)
        return base_logits + anchored

    def forward(self, t: torch.Tensor, load: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(t, load, action), dim=-1)

    @torch.no_grad()
    def predict_probabilities(self, t: int, load: int, action: int) -> np.ndarray:
        device = next(self.parameters()).device
        probabilities = self(
            torch.tensor([t], device=device),
            torch.tensor([load], device=device),
            torch.tensor([action], device=device),
        )[0]
        return probabilities.cpu().numpy().astype(np.float64)


def _tensor(
    frame: pd.DataFrame, column: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.as_tensor(frame[column].to_numpy(copy=True), dtype=dtype, device=device)


def _probability_target(frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    values = frame[
        ["next_load_prob_0", "next_load_prob_1", "next_load_prob_2"]
    ].to_numpy(copy=True)
    return torch.as_tensor(
        values,
        dtype=torch.float32,
        device=device,
    )


def _continuation(frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    values = frame[
        ["continuation_value_0", "continuation_value_1", "continuation_value_2"]
    ].to_numpy(copy=True)
    return torch.as_tensor(
        values,
        dtype=torch.float32,
        device=device,
    )


def planning_predictions(
    model: StructuredActionEffectModel,
    frame: pd.DataFrame,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    probabilities = model(
        _tensor(frame, "t", torch.long, device),
        _tensor(frame, "load", torch.long, device),
        _tensor(frame, "action", torch.long, device),
    )
    reward = _tensor(frame, "reward", torch.float32, device)
    return reward + float(gamma) * torch.sum(
        probabilities * _continuation(frame, device), dim=-1
    )


def _ranking_pair_indices(
    frame: pd.DataFrame,
    use_priority: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index_by_group: dict[tuple[int, int, int, int], list[int]] = {}
    for index, row in enumerate(frame.itertuples(index=False)):
        key = (int(row.t), int(row.load), int(row.queue), int(row.remaining_budget))
        index_by_group.setdefault(key, []).append(index)
    left, right, weights = [], [], []
    q_target = frame.q_lv.to_numpy()
    priority = frame.priority_weight.to_numpy() if use_priority else np.ones(len(frame))
    for indices in index_by_group.values():
        for offset, first in enumerate(indices[:-1]):
            for second in indices[offset + 1 :]:
                delta = q_target[first] - q_target[second]
                if abs(delta) <= 1.0e-9:
                    continue
                if delta > 0:
                    left.append(first)
                    right.append(second)
                else:
                    left.append(second)
                    right.append(first)
                weights.append(max(priority[first], priority[second]))
    return (
        torch.as_tensor(left, dtype=torch.long, device=device),
        torch.as_tensor(right, dtype=torch.long, device=device),
        torch.as_tensor(weights, dtype=torch.float32, device=device),
    )


def _ranking_loss(
    predicted_q: torch.Tensor,
    pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    margin: float,
) -> torch.Tensor:
    left_tensor, right_tensor, weight_tensor = pairs
    if left_tensor.numel() == 0:
        return predicted_q.sum() * 0.0
    losses = torch.relu(
        float(margin) - (predicted_q[left_tensor] - predicted_q[right_tensor])
    )
    return torch.sum(losses * weight_tensor) / torch.sum(weight_tensor)


@torch.no_grad()
def selection_metrics(
    model: StructuredActionEffectModel,
    frame: pd.DataFrame,
    gamma: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predicted = planning_predictions(model, frame, gamma, device).cpu().numpy()
    target = frame.q_lv.to_numpy()
    groups = frame.groupby(
        ["t", "load", "queue", "remaining_budget"], sort=False
    ).ngroup().to_numpy()
    actions = frame.action.to_numpy(dtype=np.int64)
    n_groups = int(groups.max()) + 1
    n_actions = int(actions.max()) + 1
    predicted_matrix = np.full((n_groups, n_actions), np.nan, dtype=np.float64)
    target_matrix = np.full_like(predicted_matrix, np.nan)
    qstar_matrix = np.full_like(predicted_matrix, np.nan)
    predicted_matrix[groups, actions] = predicted
    target_matrix[groups, actions] = target
    qstar_matrix[groups, actions] = frame.q_star.to_numpy()
    predicted_actions = np.nanargmax(predicted_matrix, axis=1)
    target_actions = np.nanargmax(target_matrix, axis=1)
    row_indices = np.arange(n_groups)
    damage = np.maximum(
        qstar_matrix[row_indices, target_actions]
        - qstar_matrix[row_indices, predicted_actions],
        0.0,
    )
    pair_correct: list[np.ndarray] = []
    for left in range(n_actions - 1):
        for right in range(left + 1, n_actions):
            target_delta = target_matrix[:, left] - target_matrix[:, right]
            predicted_delta = predicted_matrix[:, left] - predicted_matrix[:, right]
            comparable = np.isfinite(target_delta) & (np.abs(target_delta) > 1.0e-9)
            pair_correct.append((target_delta[comparable] * predicted_delta[comparable]) > 0.0)
    flattened_pairs = np.concatenate(pair_correct) if pair_correct else np.ones(1, dtype=bool)
    mean_damage = float(np.mean(damage))
    pair_error = 1.0 - float(np.mean(flattened_pairs))
    action_error = float(np.mean(predicted_actions != target_actions))
    q_mae = float(np.mean(np.abs(predicted - target)))
    selection_score = mean_damage + 0.25 * pair_error + 0.05 * q_mae
    return {
        "selection_score": selection_score,
        "lv_action_error_rate": action_error,
        "mean_exact_q_damage_vs_lv": mean_damage,
        "pair_ranking_error_rate": pair_error,
        "planning_q_mae": q_mae,
    }


def train_structured_model(
    labels: pd.DataFrame,
    horizon: int,
    n_loads: int,
    n_actions: int,
    gamma: float,
    loss_weights: LossWeights,
    config: TrainingConfig,
    device: torch.device | str = "cpu",
) -> TrainedStructuredModel:
    device = torch.device(device)
    torch.manual_seed(config.seed)
    model = StructuredActionEffectModel(
        horizon, n_loads, n_actions, hidden_dim=config.hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train = labels[labels.split == "train"].reset_index(drop=True)
    validation = labels[labels.split == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("both training and validation label rows are required")
    state = train.drop_duplicates(["t", "load", "queue", "action"]).reset_index(drop=True)
    target_state = _probability_target(state, device)
    state_t = _tensor(state, "t", torch.long, device)
    state_load = _tensor(state, "load", torch.long, device)
    state_action = _tensor(state, "action", torch.long, device)
    target_a0 = target_state.clone()
    a0_actions = torch.zeros_like(state_action)
    q_target = _tensor(train, "q_lv", torch.float32, device)
    priority = (
        _tensor(train, "priority_weight", torch.float32, device)
        if config.use_priority_weights
        else torch.ones(len(train), dtype=torch.float32, device=device)
    )
    ranking_pairs = _ranking_pair_indices(train, config.use_priority_weights, device)
    history: list[dict[str, float | int]] = []
    best_score = np.inf
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(config.max_epochs):
        model.train()
        predicted_state = model(state_t, state_load, state_action)
        state_loss = torch.mean((predicted_state - target_state) ** 2)
        predicted_a0 = model(state_t, state_load, a0_actions)
        effect_loss = torch.mean(
            ((predicted_state - predicted_a0) - (target_state - target_a0)) ** 2
        )
        predicted_q = planning_predictions(model, train, gamma, device)
        q_loss = torch.sum(priority * (predicted_q - q_target) ** 2) / torch.sum(priority)
        rank_loss = _ranking_loss(predicted_q, ranking_pairs, loss_weights.rank_margin)
        total = (
            loss_weights.state * state_loss
            + loss_weights.effect * effect_loss
            + loss_weights.q * q_loss
            + loss_weights.rank * rank_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        metrics = selection_metrics(model, validation, gamma, device)
        history.append(
            {
                "epoch": epoch,
                "total_loss": float(total.detach().cpu()),
                "state_loss": float(state_loss.detach().cpu()),
                "effect_loss": float(effect_loss.detach().cpu()),
                "q_loss": float(q_loss.detach().cpu()),
                "rank_loss": float(rank_loss.detach().cpu()),
                **metrics,
            }
        )
        if metrics["selection_score"] < best_score - 1.0e-8:
            best_score = metrics["selection_score"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    final_metrics = selection_metrics(model, validation, gamma, device)
    return TrainedStructuredModel(
        model=model,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        selection_metrics=final_metrics,
        loss_weights=loss_weights,
        training_config=config,
    )


def model_checkpoint_payload(trained: TrainedStructuredModel) -> dict[str, object]:
    model = trained.model
    return {
        "state_dict": model.state_dict(),
        "horizon": model.horizon,
        "n_loads": model.n_loads,
        "n_actions": model.n_actions,
        "hidden_dim": model.base[0].out_features,
        "best_epoch": trained.best_epoch,
        "selection_metrics": trained.selection_metrics,
        "loss_weights": asdict(trained.loss_weights),
        "training_config": asdict(trained.training_config),
    }
