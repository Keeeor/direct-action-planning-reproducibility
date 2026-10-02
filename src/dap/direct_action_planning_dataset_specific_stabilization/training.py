from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from dap.direct_action_planning_dataset_validation.models import (
    FeatureNormalizer,
)
from dap.direct_action_planning_dataset_validation.training import (
    BranchDataset,
)
from .models import ScaledEvidenceValueNetwork


def compute_value_target_scale(data: BranchDataset, *, horizon: int) -> float:
    """Derive a fixed value scale from training rewards only."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    rewards = np.asarray(data.rewards[data.feasible], dtype=np.float64)
    if rewards.size == 0 or not np.isfinite(rewards).all():
        raise ValueError("feasible training rewards must be finite and non-empty")
    reward_rms = float(np.sqrt(np.mean(np.square(rewards))))
    return max(1.0, reward_rms * float(np.sqrt(horizon)))


def _bellman_targets(
    model: ScaledEvidenceValueNetwork,
    data: BranchDataset,
    gamma: float,
) -> np.ndarray:
    next_values = model.predict(data.next_observations.reshape(-1, 14)).reshape(
        data.n_states, 4
    )
    q_values = data.rewards.astype(np.float64) + float(gamma) * (
        ~data.done[:, None]
    ) * next_values
    q_values[~data.feasible] = -np.inf
    return np.max(q_values, axis=1).astype(np.float32)


def _validate_candidate_grid(
    iterations: int,
    candidate_iterations: tuple[int, ...],
) -> tuple[int, ...]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    candidates = tuple(int(value) for value in candidate_iterations)
    if not candidates:
        raise ValueError("candidate_iterations must not be empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_iterations must be unique")
    if any(value < 0 or value >= iterations for value in candidates):
        raise ValueError("candidate iteration is outside the trained FVI range")
    return tuple(sorted(candidates))


def train_value_candidates(
    training: BranchDataset,
    validation: BranchDataset,
    *,
    seed: int,
    gamma: float,
    iterations: int,
    candidate_iterations: tuple[int, ...],
    epochs_per_iteration: int,
    learning_rate: float,
    hidden_dim: int,
    target_scale: float = 1.0,
    zero_initialize_output: bool = False,
) -> tuple[dict[int, ScaledEvidenceValueNetwork], list[dict[str, float]]]:
    """Train one FVI sequence and retain only the preregistered rounds."""

    candidates = _validate_candidate_grid(iterations, candidate_iterations)
    if epochs_per_iteration <= 0:
        raise ValueError("epochs_per_iteration must be positive")
    scale = float(target_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("target_scale must be finite and positive")
    torch.manual_seed(seed)
    model = ScaledEvidenceValueNetwork(
        FeatureNormalizer.fit(training.observations),
        hidden_dim=hidden_dim,
        output_scale=scale,
        zero_initialize_output=zero_initialize_output,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    observations = torch.as_tensor(training.observations, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    retained: dict[int, ScaledEvidenceValueNetwork] = {}
    history: list[dict[str, float]] = []
    for iteration in range(iterations):
        target_model = copy.deepcopy(model).train(False)
        targets = torch.as_tensor(
            _bellman_targets(target_model, training, gamma), dtype=torch.float32
        )
        losses: list[float] = []
        for _ in range(epochs_per_iteration):
            permutation = rng.permutation(len(observations))
            for start in range(0, len(observations), 256):
                index = torch.as_tensor(
                    permutation[start : start + 256], dtype=torch.long
                )
                loss = nn.functional.smooth_l1_loss(
                    model(observations[index]) / scale,
                    targets[index] / scale,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        validation_target = _bellman_targets(model, validation, gamma)
        validation_prediction = model.predict(validation.observations)
        residual = float(
            np.mean(np.abs(validation_prediction - validation_target))
        )
        history.append(
            {
                "iteration": float(iteration),
                "training_loss": float(np.mean(losses)),
                "validation_bellman_mae": residual,
                "validation_bellman_nmae": residual / scale,
                "value_target_scale": scale,
                "zero_initialized_output": float(bool(zero_initialize_output)),
                "retained_candidate": float(iteration in candidates),
            }
        )
        if iteration in candidates:
            retained[iteration] = copy.deepcopy(model).train(False)
    if tuple(retained) != candidates:
        raise AssertionError("not all registered value candidates were retained")
    return retained, history
