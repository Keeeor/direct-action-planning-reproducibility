from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
)


def _basis(remaining_horizon: np.ndarray | float, horizon: int, degree: int) -> np.ndarray:
    values = np.asarray(remaining_horizon, dtype=np.float64)
    scaled = 2.0 * (values - 1.0) / max(horizon - 1, 1) - 1.0
    return np.stack([scaled**power for power in range(degree + 1)], axis=-1)


@dataclass(frozen=True)
class DAVSModel:
    coefficients: np.ndarray
    action_costs: np.ndarray
    horizon: int
    degree: int
    ridge: float
    rank_beta: float
    training_rows: int
    training_target: str = "Q_branch"

    def predict_scores(
        self,
        load: int,
        queue: int,
        budget: int,
        remaining_horizon: int,
    ) -> np.ndarray:
        if not 0 <= remaining_horizon <= self.horizon:
            raise ValueError("remaining horizon is outside the fitted grid")
        coefficients = self.coefficients[load, queue, budget]
        phi = _basis(float(remaining_horizon), self.horizon, self.degree)
        scores = coefficients @ phi
        scores = np.asarray(scores, dtype=np.float64)
        scores[self.action_costs > budget] = np.nan
        return scores


def _fit_group(
    group: pd.DataFrame,
    feasible_actions: np.ndarray,
    horizon: int,
    degree: int,
    ridge: float,
    rank_beta: float,
    rank_margin: float = 0.10,
) -> np.ndarray:
    width = degree + 1
    action_to_block = {int(action): index for index, action in enumerate(feasible_actions)}
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for row in group.itertuples(index=False):
        design = np.zeros(len(feasible_actions) * width, dtype=np.float64)
        block = action_to_block[int(row.action)]
        design[block * width : (block + 1) * width] = _basis(
            float(row.remaining_horizon), horizon, degree
        )
        rows.append(design)
        targets.append(float(row.Q_branch))
    if rank_beta > 0:
        pivot = group.pivot_table(
            index="remaining_horizon", columns="action", values="Q_branch", aggfunc="mean"
        )
        weight = np.sqrt(float(rank_beta))
        for remaining_horizon, action_values in pivot.iterrows():
            available = [int(action) for action in feasible_actions if action in action_values.index]
            phi = _basis(float(remaining_horizon), horizon, degree)
            for left_index, left in enumerate(available):
                for right in available[left_index + 1 :]:
                    left_value = float(action_values[left])
                    right_value = float(action_values[right])
                    if not np.isfinite(left_value) or not np.isfinite(right_value):
                        continue
                    design = np.zeros(len(feasible_actions) * width, dtype=np.float64)
                    left_block = action_to_block[left]
                    right_block = action_to_block[right]
                    design[left_block * width : (left_block + 1) * width] = weight * phi
                    design[right_block * width : (right_block + 1) * width] = -weight * phi
                    rows.append(design)
                    gap = left_value - right_value
                    signed_margin = np.sign(gap) * max(abs(gap), rank_margin)
                    targets.append(weight * signed_margin)
    design_matrix = np.stack(rows)
    target = np.asarray(targets, dtype=np.float64)
    gram = design_matrix.T @ design_matrix
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
    solution = np.linalg.solve(gram + regularizer, design_matrix.T @ target)
    return solution.reshape(len(feasible_actions), width)


def fit_davs_model(
    mdp: ActionConditionedBudgetMDP,
    frame: pd.DataFrame,
    degree: int,
    ridge: float,
    rank_beta: float,
    split: str = "train",
) -> DAVSModel:
    if degree < 1 or ridge < 0 or rank_beta < 0:
        raise ValueError("invalid DAVS fit hyperparameter")
    required = {
        "split",
        "load",
        "queue",
        "remaining_budget",
        "remaining_horizon",
        "action",
        "Q_branch",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"DAVS training table is missing: {sorted(missing)}")
    training = frame[frame.split == split].copy()
    if training.empty:
        raise ValueError(f"no rows are assigned to the {split} split")
    aggregated = training.groupby(
        ["load", "queue", "remaining_budget", "remaining_horizon", "action"],
        as_index=False,
    ).Q_branch.mean()
    shape = (
        mdp.n_loads,
        mdp.config.max_queue + 1,
        mdp.config.max_budget + 1,
        mdp.n_actions,
        degree + 1,
    )
    coefficients = np.full(shape, np.nan, dtype=np.float64)
    for load, queue, budget in np.ndindex(shape[:3]):
        group = aggregated[
            (aggregated.load == load)
            & (aggregated.queue == queue)
            & (aggregated.remaining_budget == budget)
        ]
        feasible = np.flatnonzero(mdp.action_costs <= budget)
        if group.empty or set(group.action.astype(int)) != set(feasible.tolist()):
            raise ValueError("training coverage is incomplete for a state-action cell")
        fitted = _fit_group(
            group, feasible, mdp.config.horizon, degree, ridge, rank_beta
        )
        coefficients[load, queue, budget, feasible] = fitted
    return DAVSModel(
        coefficients=coefficients,
        action_costs=mdp.action_costs.copy(),
        horizon=mdp.config.horizon,
        degree=int(degree),
        ridge=float(ridge),
        rank_beta=float(rank_beta),
        training_rows=int(len(training)),
    )


@dataclass(frozen=True)
class EnsemblePrediction:
    mean: np.ndarray
    variance: np.ndarray
    top_action_votes: np.ndarray
    ranking_uncertainty: float


@dataclass(frozen=True)
class DAVSEnsemble:
    models: tuple[DAVSModel, ...]

    @classmethod
    def fit(
        cls,
        mdp: ActionConditionedBudgetMDP,
        frame: pd.DataFrame,
        members: int,
        seed: int,
        degree: int,
        ridge: float,
        rank_beta: float,
    ) -> "DAVSEnsemble":
        if members < 2:
            raise ValueError("an ensemble requires at least two members")
        replications = np.sort(frame.branch_replication.unique())
        if len(replications) < 2:
            raise ValueError("ensemble bootstrap requires multiple branch replications")
        rng = np.random.default_rng(int(seed))
        fitted: list[DAVSModel] = []
        for _ in range(members):
            selected = rng.choice(replications, size=len(replications), replace=True)
            boot = pd.concat(
                [frame[frame.branch_replication == replication] for replication in selected],
                ignore_index=True,
            )
            fitted.append(
                fit_davs_model(
                    mdp,
                    boot,
                    degree=degree,
                    ridge=ridge,
                    rank_beta=rank_beta,
                )
            )
        return cls(tuple(fitted))

    def predict(
        self, load: int, queue: int, budget: int, remaining_horizon: int
    ) -> EnsemblePrediction:
        scores = np.stack(
            [
                model.predict_scores(load, queue, budget, remaining_horizon)
                for model in self.models
            ]
        )
        feasible = np.flatnonzero(self.models[0].action_costs <= budget)
        mean = np.full(scores.shape[1], np.nan, dtype=np.float64)
        variance = np.full(scores.shape[1], np.nan, dtype=np.float64)
        mean[feasible] = np.mean(scores[:, feasible], axis=0)
        variance[feasible] = np.var(scores[:, feasible], axis=0)
        votes = np.zeros(len(mean), dtype=np.float64)
        for row in scores:
            votes[int(np.nanargmax(row))] += 1.0
        votes /= len(scores)
        vote_uncertainty = 1.0 - float(np.max(votes))
        pair_uncertainties: list[float] = []
        for left_index, left in enumerate(feasible):
            for right in feasible[left_index + 1 :]:
                probability = float(np.mean(scores[:, left] > scores[:, right]))
                pair_uncertainties.append(4.0 * probability * (1.0 - probability))
        pair_uncertainty = float(np.mean(pair_uncertainties)) if pair_uncertainties else 0.0
        ordered = np.sort(mean[feasible])
        gap = float(ordered[-1] - ordered[-2]) if len(ordered) >= 2 else np.inf
        scale = float(np.sqrt(np.nanmax(variance[feasible]))) if len(feasible) else 0.0
        variance_ratio = scale / (scale + max(gap, 0.0) + 1e-12)
        return EnsemblePrediction(
            mean=mean,
            variance=variance,
            top_action_votes=votes,
            ranking_uncertainty=float(
                np.clip(max(vote_uncertainty, pair_uncertainty, variance_ratio), 0.0, 1.0)
            ),
        )


@dataclass(frozen=True)
class DAVSOutput:
    action: torch.Tensor
    q_values: torch.Tensor
    ranking_uncertainty: torch.Tensor


class DAVSAgent(nn.Module):
    """Canonical-observation adapter that directly selects learned action values."""

    def __init__(self, mdp: ActionConditionedBudgetMDP, scorer: DAVSModel | DAVSEnsemble):
        super().__init__()
        self.mdp = mdp
        self.scorer = scorer

    def reset_budget_controller(self) -> None:
        return None

    def act(self, observation: torch.Tensor, deterministic: bool = True, **kwargs) -> DAVSOutput:
        del deterministic, kwargs
        if observation.ndim != 2 or observation.shape[-1] != 14:
            raise ValueError("canonical observation must have shape [batch, 14]")
        cpu = observation.detach().to(device="cpu", dtype=torch.float64)
        load_values = torch.as_tensor(self.mdp.load_arrivals, dtype=torch.float64)
        actions: list[int] = []
        scores: list[np.ndarray] = []
        uncertainty: list[float] = []
        for row in cpu:
            budget = int(
                torch.round(row[-2] * self.mdp.config.max_budget)
                .clamp(0, self.mdp.config.max_budget)
                .item()
            )
            remaining_horizon = int(
                torch.round(row[-1] * self.mdp.config.horizon)
                .clamp(1, self.mdp.config.horizon)
                .item()
            )
            queue = int(torch.round(row[2]).clamp(0, self.mdp.config.max_queue).item())
            load = int(torch.argmin((row[0] - load_values).abs()).item())
            if isinstance(self.scorer, DAVSEnsemble):
                prediction = self.scorer.predict(load, queue, budget, remaining_horizon)
                values = prediction.mean
                uncertainty.append(prediction.ranking_uncertainty)
            else:
                values = self.scorer.predict_scores(load, queue, budget, remaining_horizon)
                uncertainty.append(0.0)
            actions.append(int(np.nanargmax(values)))
            scores.append(values)
        return DAVSOutput(
            action=torch.as_tensor(actions, dtype=torch.long, device=observation.device),
            q_values=torch.as_tensor(
                np.stack(scores), dtype=observation.dtype, device=observation.device
            ),
            ranking_uncertainty=torch.as_tensor(
                uncertainty, dtype=observation.dtype, device=observation.device
            ),
        )
