from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from stage2_dynamic_budget.direct_action_planning_repair.model import (
    LossWeights,
    StructuredActionEffectModel,
    TrainedStructuredModel,
    TrainingConfig,
    _probability_target,
    _ranking_loss,
    _ranking_pair_indices,
    _tensor,
    planning_predictions,
    selection_metrics,
)


def train_controlled_model(
    labels: pd.DataFrame,
    horizon: int,
    n_loads: int,
    n_actions: int,
    gamma: float,
    loss_weights: LossWeights,
    config: TrainingConfig,
    retention_reference: StructuredActionEffectModel | None = None,
    retention_anchor: pd.DataFrame | None = None,
    retention_weight: float = 0.0,
    state_targets: pd.DataFrame | None = None,
    device: torch.device | str = "cpu",
) -> TrainedStructuredModel:
    """Frozen DAP-Repair training with an optional Q-score retention constraint."""

    device = torch.device(device)
    torch.manual_seed(config.seed)
    model = StructuredActionEffectModel(
        horizon, n_loads, n_actions, hidden_dim=config.hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train = labels[labels.split == "train"].reset_index(drop=True)
    validation = labels[labels.split == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("both training and validation rows are required")
    state = (
        state_targets.reset_index(drop=True)
        if state_targets is not None
        else train.drop_duplicates(["t", "load", "queue", "action"]).reset_index(drop=True)
    )
    target_state = _probability_target(state, device)
    state_t = _tensor(state, "t", torch.long, device)
    state_load = _tensor(state, "load", torch.long, device)
    state_action = _tensor(state, "action", torch.long, device)
    target_a0 = target_state.clone()
    a0_actions = torch.zeros_like(state_action)
    state_weight = (
        _tensor(state, "state_loss_weight", torch.float32, device)
        if "state_loss_weight" in state
        else torch.ones(len(state), dtype=torch.float32, device=device)
    )
    q_target = _tensor(train, "q_lv", torch.float32, device)
    priority = (
        _tensor(train, "priority_weight", torch.float32, device)
        if config.use_priority_weights
        else torch.ones(len(train), dtype=torch.float32, device=device)
    )
    ranking_pairs = _ranking_pair_indices(train, config.use_priority_weights, device)
    use_retention = (
        retention_reference is not None
        and retention_anchor is not None
        and not retention_anchor.empty
        and retention_weight > 0.0
    )
    if use_retention:
        retention_reference = copy.deepcopy(retention_reference).to(device).eval()
        retention_anchor = retention_anchor.reset_index(drop=True)
        with torch.no_grad():
            retention_target = planning_predictions(
                retention_reference, retention_anchor, gamma, device
            ).detach()
    history: list[dict[str, float | int]] = []
    best_score = np.inf
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(config.max_epochs):
        model.train()
        predicted_state = model(state_t, state_load, state_action)
        state_error = torch.mean((predicted_state - target_state) ** 2, dim=-1)
        state_loss = torch.sum(state_weight * state_error) / torch.sum(state_weight)
        predicted_a0 = model(state_t, state_load, a0_actions)
        effect_error = torch.mean(
            ((predicted_state - predicted_a0) - (target_state - target_a0)) ** 2,
            dim=-1,
        )
        effect_loss = torch.sum(state_weight * effect_error) / torch.sum(state_weight)
        predicted_q = planning_predictions(model, train, gamma, device)
        q_loss = torch.sum(priority * (predicted_q - q_target) ** 2) / torch.sum(priority)
        rank_loss = _ranking_loss(predicted_q, ranking_pairs, loss_weights.rank_margin)
        retention_loss = predicted_q.sum() * 0.0
        if use_retention:
            retention_prediction = planning_predictions(model, retention_anchor, gamma, device)
            retention_loss = torch.mean((retention_prediction - retention_target) ** 2)
        total = (
            loss_weights.state * state_loss
            + loss_weights.effect * effect_loss
            + loss_weights.q * q_loss
            + loss_weights.rank * rank_loss
            + float(retention_weight) * retention_loss
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
                "retention_loss": float(retention_loss.detach().cpu()),
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
    final_metrics["retention_enabled"] = float(use_retention)
    return TrainedStructuredModel(
        model=model,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        selection_metrics=final_metrics,
        loss_weights=loss_weights,
        training_config=config,
    )
