from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from stage2_dynamic_budget.direct_action_planning_dataset_validation.models import (
    FeatureNormalizer,
)
from stage2_dynamic_budget.direct_action_planning_dataset_validation.training import (
    BranchDataset,
)
from stage2_dynamic_budget.direct_action_planning_paper_closure.models import (
    ScaledEvidenceValueNetwork,
)

from .planning import DEFAULT_EXOGENOUS_INDICES, postdecision_state


@dataclass(frozen=True)
class PostDecisionRegression:
    states: np.ndarray
    targets: dict[int, np.ndarray]


def build_postdecision_regression(
    data: BranchDataset,
    source_values: dict[int, object],
    *,
    exogenous_indices: tuple[int, ...] = DEFAULT_EXOGENOUS_INDICES,
) -> PostDecisionRegression:
    """Build sampled E[V(next pre-state) | post-state] regression pairs."""

    if not source_values:
        raise ValueError("source_values must not be empty")
    feasible = np.asarray(data.feasible, dtype=bool)
    if feasible.shape != data.next_observations.shape[:2]:
        raise ValueError("feasibility and next-observation shapes disagree")
    states = postdecision_state(
        data.next_observations, exogenous_indices=exogenous_indices
    )[feasible].astype(np.float32, copy=False)
    if len(states) == 0:
        raise ValueError("post-decision regression has no feasible samples")
    flat_next = data.next_observations.reshape(-1, 14)
    flat_feasible = feasible.reshape(-1)
    terminal = np.repeat(np.asarray(data.done, dtype=bool), feasible.shape[1])[flat_feasible]
    targets: dict[int, np.ndarray] = {}
    for iteration, value in sorted(source_values.items()):
        prediction = np.asarray(value.predict(flat_next), dtype=np.float64)[flat_feasible]
        prediction = np.where(terminal, 0.0, prediction)
        if not np.isfinite(prediction).all():
            raise ValueError(f"non-finite source target for candidate {iteration}")
        targets[int(iteration)] = prediction.astype(np.float32)
    return PostDecisionRegression(states=states, targets=targets)


def train_postdecision_candidates(
    training: BranchDataset,
    validation: BranchDataset,
    *,
    source_values: dict[int, object],
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    batch_size: int,
    target_scale: float,
    exogenous_indices: tuple[int, ...] = DEFAULT_EXOGENOUS_INDICES,
) -> tuple[dict[int, ScaledEvidenceValueNetwork], list[dict[str, float]]]:
    """Fit one standard afterstate VFA for each frozen source-value candidate."""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    scale = float(target_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("target_scale must be finite and positive")
    training_pairs = build_postdecision_regression(
        training, source_values, exogenous_indices=exogenous_indices
    )
    validation_pairs = build_postdecision_regression(
        validation, source_values, exogenous_indices=exogenous_indices
    )
    normalizer = FeatureNormalizer.fit(training_pairs.states)
    observations = torch.as_tensor(training_pairs.states, dtype=torch.float32)
    validation_observations = torch.as_tensor(
        validation_pairs.states, dtype=torch.float32
    )
    models: dict[int, ScaledEvidenceValueNetwork] = {}
    history: list[dict[str, float]] = []
    for offset, iteration in enumerate(sorted(source_values)):
        model_seed = int(seed) + 104_729 * offset
        torch.manual_seed(model_seed)
        model = ScaledEvidenceValueNetwork(
            normalizer,
            hidden_dim=int(hidden_dim),
            output_scale=scale,
            zero_initialize_output=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        targets = torch.as_tensor(training_pairs.targets[iteration], dtype=torch.float32)
        validation_targets = torch.as_tensor(
            validation_pairs.targets[iteration], dtype=torch.float32
        )
        rng = np.random.default_rng(model_seed)
        for epoch in range(int(epochs)):
            losses: list[float] = []
            permutation = rng.permutation(len(observations))
            for start in range(0, len(observations), int(batch_size)):
                index = torch.as_tensor(
                    permutation[start : start + int(batch_size)], dtype=torch.long
                )
                prediction = model(observations[index])
                loss = nn.functional.smooth_l1_loss(
                    prediction / scale, targets[index] / scale
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            with torch.no_grad():
                validation_prediction = model(validation_observations)
                validation_mae = float(
                    torch.mean(torch.abs(validation_prediction - validation_targets))
                )
            history.append(
                {
                    "candidate_iteration": float(iteration),
                    "epoch": float(epoch),
                    "training_loss": float(np.mean(losses)),
                    "validation_mae": validation_mae,
                    "target_scale": scale,
                }
            )
        models[int(iteration)] = model.train(False)
    return models, history
