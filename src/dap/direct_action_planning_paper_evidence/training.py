from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time

import numpy as np
import torch
from torch import nn

from dap.direct_action_planning_dataset_validation.data import TraceDataset
from dap.direct_action_planning_dataset_validation.models import (
    FeatureNormalizer,
    FullTransitionNetwork,
)
from dap.direct_action_planning_dataset_validation.training import (
    BranchDataset,
    collect_branch_dataset,
    train_full_transition,
)

from .models import DistilledActionPolicy, EvidenceLoadForecaster, EvidenceValueNetwork
from .planning import make_evidence_planner, replace_next_load_numpy, replace_next_load_tensor


@dataclass(frozen=True)
class ForecasterWeights:
    load: float = 1.0
    planning_q: float = 4.0
    rank: float = 2.0
    rank_margin: float = 0.02


@dataclass(frozen=True)
class PaperTrainingArtifacts:
    base_value: EvidenceValueNetwork
    refreshed_value: EvidenceValueNetwork
    no_anchor_value: EvidenceValueNetwork | None
    no_budget_horizon_base_value: EvidenceValueNetwork | None
    no_budget_horizon_value: EvidenceValueNetwork | None
    decision_forecaster: EvidenceLoadForecaster
    state_forecaster: EvidenceLoadForecaster | None
    full_transition: FullTransitionNetwork | None
    training_data: BranchDataset
    validation_data: BranchDataset
    combined_data: BranchDataset
    histories: dict[str, list[dict[str, float]]]
    collection: dict[str, int]
    component_seconds: dict[str, float]
    training_seconds: float


def _bellman_targets(model, data: BranchDataset, gamma: float) -> np.ndarray:
    next_values = model.predict(data.next_observations.reshape(-1, 14)).reshape(
        data.n_states, 4
    )
    q_values = data.rewards.astype(np.float64) + float(gamma) * (
        ~data.done[:, None]
    ) * next_values
    q_values[~data.feasible] = -np.inf
    return np.max(q_values, axis=1).astype(np.float32)


def train_value_model(
    training: BranchDataset,
    validation: BranchDataset,
    *,
    seed: int,
    gamma: float,
    iterations: int,
    epochs_per_iteration: int,
    learning_rate: float,
    hidden_dim: int,
    feature_mask: np.ndarray | None = None,
    initial: EvidenceValueNetwork | None = None,
    anchors: np.ndarray | None = None,
    anchor_weight: float = 0.0,
    checkpoint_selection: str = "validation_bellman_residual",
) -> tuple[EvidenceValueNetwork, list[dict[str, float]]]:
    if checkpoint_selection not in {
        "validation_bellman_residual",
        "final_iteration",
    }:
        raise ValueError(
            "checkpoint_selection must be validation_bellman_residual or final_iteration"
        )
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    torch.manual_seed(seed)
    if initial is None:
        model = EvidenceValueNetwork(
            FeatureNormalizer.fit(training.observations), hidden_dim, feature_mask
        )
    else:
        model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    observations = torch.as_tensor(training.observations, dtype=torch.float32)
    anchor_tensor = None
    anchor_target = None
    if anchors is not None:
        if initial is None:
            raise ValueError("anchors require an initial value model")
        anchor_tensor = torch.as_tensor(anchors, dtype=torch.float32)
        with torch.no_grad():
            anchor_target = initial(anchor_tensor).detach()
    rng = np.random.default_rng(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = np.inf
    selected_iteration = 0
    history: list[dict[str, float]] = []
    for iteration in range(iterations):
        target_model = copy.deepcopy(model).train(False)
        targets = torch.as_tensor(_bellman_targets(target_model, training, gamma))
        losses: list[float] = []
        for _ in range(epochs_per_iteration):
            permutation = rng.permutation(len(observations))
            for start in range(0, len(observations), 256):
                index = torch.as_tensor(permutation[start : start + 256], dtype=torch.long)
                loss = nn.functional.smooth_l1_loss(model(observations[index]), targets[index])
                if anchor_tensor is not None and anchor_weight > 0:
                    loss = loss + float(anchor_weight) * nn.functional.mse_loss(
                        model(anchor_tensor), anchor_target
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        validation_target = _bellman_targets(model, validation, gamma)
        validation_prediction = model.predict(validation.observations)
        residual = float(np.mean(np.abs(validation_prediction - validation_target)))
        history.append(
            {
                "iteration": float(iteration),
                "training_loss": float(np.mean(losses)),
                "validation_bellman_mae": residual,
            }
        )
        if residual < best_validation:
            best_validation = residual
            best_state = copy.deepcopy(model.state_dict())
            selected_iteration = iteration
    if checkpoint_selection == "final_iteration":
        best_state = copy.deepcopy(model.state_dict())
        selected_iteration = iterations - 1
    for row in history:
        row["checkpoint_selected"] = float(
            int(row["iteration"]) == selected_iteration
        )
        row["selected_iteration"] = float(selected_iteration)
    model.load_state_dict(best_state)
    return model.train(False), history


def training_anchor_states(training: BranchDataset, limit: int = 512) -> np.ndarray:
    """Return fixed D0 anchors without exposing validation or evaluation states."""

    if limit <= 0:
        raise ValueError("anchor limit must be positive")
    return training.observations[: min(limit, training.n_states)].copy()


def planning_q_tensor(
    forecaster: EvidenceLoadForecaster,
    value: EvidenceValueNetwork,
    observations: torch.Tensor,
    next_observations: torch.Tensor,
    rewards: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    predicted_load = forecaster(observations)
    predicted_next = replace_next_load_tensor(next_observations, predicted_load)
    continuation = value(predicted_next.reshape(-1, 14)).reshape(-1, 4)
    return rewards + float(gamma) * (~done[:, None]) * continuation


def _ranking_loss(
    predicted_q: torch.Tensor,
    target_q: torch.Tensor,
    feasible: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for left in range(3):
        for right in range(left + 1, 4):
            comparable = feasible[:, left] & feasible[:, right]
            target_delta = target_q[:, left] - target_q[:, right]
            comparable &= torch.abs(target_delta) > 1.0e-6
            if not bool(comparable.any()):
                continue
            sign = torch.sign(target_delta[comparable])
            predicted_delta = predicted_q[comparable, left] - predicted_q[comparable, right]
            losses.append(torch.relu(float(margin) - sign * predicted_delta))
    if not losses:
        return predicted_q.sum() * 0.0
    return torch.mean(torch.cat(losses))


@torch.no_grad()
def forecaster_metrics(
    forecaster: EvidenceLoadForecaster,
    value: EvidenceValueNetwork,
    data: BranchDataset,
    gamma: float,
) -> dict[str, float]:
    observations = torch.as_tensor(data.observations, dtype=torch.float32)
    next_observations = torch.as_tensor(data.next_observations, dtype=torch.float32)
    rewards = torch.as_tensor(data.rewards, dtype=torch.float32)
    feasible = torch.as_tensor(data.feasible, dtype=torch.bool)
    done = torch.as_tensor(data.done, dtype=torch.bool)
    predicted_q = planning_q_tensor(
        forecaster, value, observations, next_observations, rewards, done, gamma
    )
    target_values = value(next_observations.reshape(-1, 14)).reshape(-1, 4)
    target_q = rewards + float(gamma) * (~done[:, None]) * target_values
    masked_pred = predicted_q.masked_fill(~feasible, -torch.inf)
    masked_target = target_q.masked_fill(~feasible, -torch.inf)
    action_error = torch.mean(
        (torch.argmax(masked_pred, dim=1) != torch.argmax(masked_target, dim=1)).float()
    )
    finite_q = feasible
    q_mae = torch.mean(torch.abs(predicted_q[finite_q] - target_q[finite_q]))
    forecast = forecaster(observations)
    load_mae = torch.mean(
        torch.abs(forecast - torch.as_tensor(data.next_load, dtype=torch.float32))
    )
    pair_correct: list[torch.Tensor] = []
    for left in range(3):
        for right in range(left + 1, 4):
            comparable = feasible[:, left] & feasible[:, right]
            target_delta = target_q[:, left] - target_q[:, right]
            comparable &= torch.abs(target_delta) > 1.0e-6
            if bool(comparable.any()):
                predicted_delta = predicted_q[:, left] - predicted_q[:, right]
                pair_correct.append(
                    (target_delta[comparable] * predicted_delta[comparable]) > 0
                )
    pair_error = (
        1.0 - float(torch.mean(torch.cat(pair_correct).float()))
        if pair_correct
        else 0.0
    )
    return {
        "validation_load_mae": float(load_mae),
        "validation_planning_q_mae": float(q_mae),
        "validation_action_error": float(action_error),
        "validation_pair_error": pair_error,
    }


def train_forecaster(
    training: BranchDataset,
    validation: BranchDataset,
    value: EvidenceValueNetwork,
    *,
    seed: int,
    gamma: float,
    epochs: int,
    weights: ForecasterWeights,
) -> tuple[EvidenceLoadForecaster, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = EvidenceLoadForecaster(value.normalizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    for parameter in value.parameters():
        parameter.requires_grad_(False)
    value.train(False)

    x = torch.as_tensor(training.observations, dtype=torch.float32)
    next_x = torch.as_tensor(training.next_observations, dtype=torch.float32)
    rewards = torch.as_tensor(training.rewards, dtype=torch.float32)
    feasible = torch.as_tensor(training.feasible, dtype=torch.bool)
    done = torch.as_tensor(training.done, dtype=torch.bool)
    y_load = torch.as_tensor(training.next_load, dtype=torch.float32)
    with torch.no_grad():
        target_value = value(next_x.reshape(-1, 14)).reshape(-1, 4)
        target_q = rewards + float(gamma) * (~done[:, None]) * target_value
        q_scale = torch.std(target_q[feasible]).clamp_min(1.0)

    rng = np.random.default_rng(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_score = np.inf
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        losses: list[float] = []
        load_losses: list[float] = []
        q_losses: list[float] = []
        rank_losses: list[float] = []
        permutation = rng.permutation(len(x))
        for start in range(0, len(x), 256):
            index = torch.as_tensor(permutation[start : start + 256], dtype=torch.long)
            prediction = model(x[index])
            load_loss = nn.functional.smooth_l1_loss(
                torch.log1p(prediction), torch.log1p(y_load[index])
            )
            predicted_q = planning_q_tensor(
                model,
                value,
                x[index],
                next_x[index],
                rewards[index],
                done[index],
                gamma,
            )
            q_loss = nn.functional.smooth_l1_loss(
                predicted_q[feasible[index]] / q_scale,
                target_q[index][feasible[index]] / q_scale,
            )
            rank_loss = _ranking_loss(
                predicted_q / q_scale,
                target_q[index] / q_scale,
                feasible[index],
                weights.rank_margin,
            )
            total = (
                weights.load * load_loss
                + weights.planning_q * q_loss
                + weights.rank * rank_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(total.detach()))
            load_losses.append(float(load_loss.detach()))
            q_losses.append(float(q_loss.detach()))
            rank_losses.append(float(rank_loss.detach()))
        metrics = forecaster_metrics(model, value, validation, gamma)
        if math.isclose(weights.planning_q, 0.0) and math.isclose(weights.rank, 0.0):
            score = metrics["validation_load_mae"]
        else:
            score = (
                metrics["validation_action_error"]
                + 0.25 * metrics["validation_planning_q_mae"] / float(q_scale)
                + 0.05 * metrics["validation_load_mae"]
            )
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "load_loss": float(np.mean(load_losses)),
                "planning_q_loss": float(np.mean(q_losses)),
                "rank_loss": float(np.mean(rank_losses)),
                "selection_score": float(score),
                **metrics,
            }
        )
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    for parameter in value.parameters():
        parameter.requires_grad_(True)
    return model.train(False), history


def planner_action_targets(
    data: BranchDataset,
    value: EvidenceValueNetwork,
    forecaster: EvidenceLoadForecaster,
    gamma: float,
) -> np.ndarray:
    predicted_load = np.asarray(
        [forecaster.predict(row) for row in data.observations], dtype=np.float32
    )
    predicted_next = replace_next_load_numpy(data.next_observations, predicted_load)
    continuation = value.predict(predicted_next.reshape(-1, 14)).reshape(-1, 4)
    q_values = data.rewards.astype(np.float64) + float(gamma) * (
        ~data.done[:, None]
    ) * continuation
    q_values[~data.feasible] = -np.inf
    return np.argmax(q_values, axis=1).astype(np.int64)


def train_distilled_policy(
    training: BranchDataset,
    validation: BranchDataset,
    value: EvidenceValueNetwork,
    forecaster: EvidenceLoadForecaster,
    *,
    action_costs: np.ndarray,
    episode_budget: float,
    gamma: float,
    seed: int,
    hidden_dim: int,
    epochs: int,
) -> tuple[DistilledActionPolicy, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = DistilledActionPolicy(
        value.normalizer, action_costs, episode_budget, hidden_dim=hidden_dim
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    x = torch.as_tensor(training.observations, dtype=torch.float32)
    y = torch.as_tensor(planner_action_targets(training, value, forecaster, gamma))
    vx = torch.as_tensor(validation.observations, dtype=torch.float32)
    vy = torch.as_tensor(planner_action_targets(validation, value, forecaster, gamma))
    rng = np.random.default_rng(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -np.inf
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        losses: list[float] = []
        permutation = rng.permutation(len(x))
        for start in range(0, len(x), 256):
            index = torch.as_tensor(permutation[start : start + 256], dtype=torch.long)
            loss = nn.functional.cross_entropy(model(x[index]), y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            accuracy = float(torch.mean((torch.argmax(model(vx), dim=1) == vy).float()))
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "validation_accuracy": accuracy,
            }
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.train(False), history


def train_paper_components(
    dataset: TraceDataset,
    *,
    horizon: int,
    budget: float,
    seed: int,
    gamma: float,
    collection_episodes: int,
    validation_episodes: int,
    fvi_iterations: int,
    refresh_iterations: int,
    model_epochs: int,
    hidden_dim: int,
    anchor_weight: float,
    checkpoint_selection: str = "validation_bellman_residual",
    validation_split: str = "validation_fit",
    include_ablations: bool = True,
) -> PaperTrainingArtifacts:
    started = time.perf_counter()
    component_seconds: dict[str, float] = {}
    component_started = time.perf_counter()
    training = collect_branch_dataset(
        dataset,
        split="train",
        horizon=horizon,
        budget=budget,
        episodes_per_domain=collection_episodes,
        seed=seed,
    )
    component_seconds["d0_collection"] = time.perf_counter() - component_started
    component_started = time.perf_counter()
    validation = collect_branch_dataset(
        dataset,
        split=validation_split,
        horizon=horizon,
        budget=budget,
        episodes_per_domain=validation_episodes,
        seed=seed + 10_000_019,
    )
    component_seconds["model_validation_collection"] = (
        time.perf_counter() - component_started
    )
    component_started = time.perf_counter()
    base_value, base_history = train_value_model(
        training,
        validation,
        seed=seed,
        gamma=gamma,
        iterations=fvi_iterations,
        epochs_per_iteration=2,
        learning_rate=1.0e-3,
        hidden_dim=hidden_dim,
        checkpoint_selection=checkpoint_selection,
    )
    component_seconds["base_value"] = time.perf_counter() - component_started
    state_forecaster = None
    state_forecast_history: list[dict[str, float]] = []
    if include_ablations:
        component_started = time.perf_counter()
        state_forecaster, state_forecast_history = train_forecaster(
            training,
            validation,
            base_value,
            seed=seed + 1,
            gamma=gamma,
            epochs=model_epochs,
            weights=ForecasterWeights(load=1.0, planning_q=0.0, rank=0.0),
        )
        component_seconds["state_forecaster"] = time.perf_counter() - component_started
    component_started = time.perf_counter()
    decision_forecaster, decision_forecast_history = train_forecaster(
        training,
        validation,
        base_value,
        seed=seed + 1,
        gamma=gamma,
        epochs=model_epochs,
        weights=ForecasterWeights(),
    )
    component_seconds["decision_forecaster"] = time.perf_counter() - component_started
    transition = None
    transition_history: list[dict[str, float]] = []
    if include_ablations:
        component_started = time.perf_counter()
        transition, transition_history = train_full_transition(
            training,
            validation,
            base_value.normalizer,
            seed=seed + 2,
            epochs=model_epochs,
        )
        component_seconds["full_transition"] = time.perf_counter() - component_started
    base_planner = make_evidence_planner(
        value=base_value,
        forecaster=decision_forecaster,
        gamma=gamma,
    )
    component_started = time.perf_counter()
    on_policy = collect_branch_dataset(
        dataset,
        split="train",
        horizon=horizon,
        budget=budget,
        episodes_per_domain=max(collection_episodes // 2, 1),
        seed=seed + 20_000_033,
        planner=base_planner,
    )
    component_seconds["on_policy_collection"] = time.perf_counter() - component_started
    combined = training.concatenate(on_policy)
    anchors = training_anchor_states(training)
    component_started = time.perf_counter()
    refreshed, refresh_history = train_value_model(
        combined,
        validation,
        seed=seed + 3,
        gamma=gamma,
        iterations=refresh_iterations,
        epochs_per_iteration=2,
        learning_rate=5.0e-4,
        hidden_dim=hidden_dim,
        checkpoint_selection=checkpoint_selection,
        initial=base_value,
        anchors=anchors,
        anchor_weight=anchor_weight,
    )
    component_seconds["value_refresh_anchor"] = time.perf_counter() - component_started
    no_anchor = None
    no_anchor_history: list[dict[str, float]] = []
    no_bh_base = None
    no_bh_base_history: list[dict[str, float]] = []
    no_bh = None
    no_bh_history: list[dict[str, float]] = []
    if include_ablations:
        component_started = time.perf_counter()
        no_anchor, no_anchor_history = train_value_model(
            combined,
            validation,
            seed=seed + 3,
            gamma=gamma,
            iterations=refresh_iterations,
            epochs_per_iteration=2,
            learning_rate=5.0e-4,
            hidden_dim=hidden_dim,
            checkpoint_selection=checkpoint_selection,
            initial=base_value,
            anchors=None,
            anchor_weight=0.0,
        )
        component_seconds["value_refresh_no_anchor"] = (
            time.perf_counter() - component_started
        )
        feature_mask = np.ones(14, dtype=np.float32)
        feature_mask[-2:] = 0.0
        component_started = time.perf_counter()
        no_bh_base, no_bh_base_history = train_value_model(
            training,
            validation,
            seed=seed,
            gamma=gamma,
            iterations=fvi_iterations,
            epochs_per_iteration=2,
            learning_rate=1.0e-3,
            hidden_dim=hidden_dim,
            checkpoint_selection=checkpoint_selection,
            feature_mask=feature_mask,
        )
        component_seconds["no_budget_horizon_base"] = (
            time.perf_counter() - component_started
        )
        component_started = time.perf_counter()
        no_bh, no_bh_history = train_value_model(
            combined,
            validation,
            seed=seed + 3,
            gamma=gamma,
            iterations=refresh_iterations,
            epochs_per_iteration=2,
            learning_rate=5.0e-4,
            hidden_dim=hidden_dim,
            checkpoint_selection=checkpoint_selection,
            initial=no_bh_base,
            anchors=anchors,
            anchor_weight=anchor_weight,
        )
        component_seconds["no_budget_horizon_refresh"] = (
            time.perf_counter() - component_started
        )
    branch_actions = int(np.sum(training.feasible) + np.sum(on_policy.feasible))
    return PaperTrainingArtifacts(
        base_value=base_value,
        refreshed_value=refreshed,
        no_anchor_value=no_anchor,
        no_budget_horizon_base_value=no_bh_base,
        no_budget_horizon_value=no_bh,
        decision_forecaster=decision_forecaster,
        state_forecaster=state_forecaster,
        full_transition=transition,
        training_data=training,
        validation_data=validation,
        combined_data=combined,
        histories={
            "base_value": base_history,
            "state_forecaster": state_forecast_history,
            "decision_forecaster": decision_forecast_history,
            "full_transition": transition_history,
            "value_refresh_anchor": refresh_history,
            "value_refresh_no_anchor": no_anchor_history,
            "no_budget_horizon_base": no_bh_base_history,
            "no_budget_horizon_refresh": no_bh_history,
        },
        collection={
            "training_states": training.n_states,
            "validation_states": validation.n_states,
            "refresh_states": on_policy.n_states,
            "branch_action_transitions": branch_actions,
        },
        component_seconds=component_seconds,
        training_seconds=time.perf_counter() - started,
    )
