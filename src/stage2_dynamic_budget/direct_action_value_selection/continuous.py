from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


OBS_COLUMNS = tuple(f"obs_{index}" for index in range(14))


def _raw_features(
    observations: np.ndarray, actions: np.ndarray, action_dim: int
) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    if observations.ndim != 2 or observations.shape[1] != len(OBS_COLUMNS):
        raise ValueError("continuous DAVS observations must have shape [n, 14]")
    if actions.shape != (len(observations),):
        raise ValueError("one action is required per observation")
    one_hot = np.eye(action_dim, dtype=np.float64)[actions]
    interactions = (one_hot[:, :, None] * observations[:, None, :]).reshape(
        len(observations), -1
    )
    return np.concatenate(
        [
            np.ones((len(observations), 1), dtype=np.float64),
            observations,
            np.square(observations),
            one_hot,
            interactions,
        ],
        axis=1,
    )


@dataclass(frozen=True)
class ContinuousDAVSModel:
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float
    action_costs: np.ndarray
    budget_scale: float
    ridge: float
    rank_beta: float
    training_states: int

    def predict_all(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (len(OBS_COLUMNS),):
            raise ValueError("continuous DAVS observation must have 14 fields")
        actions = np.arange(len(self.action_costs), dtype=np.int64)
        repeated = np.repeat(observation[None, :], len(actions), axis=0)
        raw = _raw_features(repeated, actions, len(actions))
        design = (raw - self.feature_mean) / self.feature_scale
        scores = (design @ self.coefficients) * self.target_scale + self.target_mean
        remaining = max(float(observation[-2]) * self.budget_scale, 0.0)
        scores[self.action_costs > remaining + 1e-8] = np.nan
        return scores


def _state_subset(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("data fraction must be in (0, 1]")
    states = frame.state_id.drop_duplicates().to_numpy()
    if fraction == 1:
        return frame.copy()
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        states, size=max(1, int(np.ceil(len(states) * fraction))), replace=False
    )
    return frame[frame.state_id.isin(selected)].copy()


def fit_continuous_davs(
    frame: pd.DataFrame,
    action_costs: np.ndarray,
    budget_scale: float,
    ridge: float,
    rank_beta: float,
    rank_margin: float,
    *,
    split: str = "train",
    data_fraction: float = 1.0,
    label_noise_std: float = 0.0,
    seed: int = 0,
) -> ContinuousDAVSModel:
    required = {"state_id", "split", "action", "Q_branch", *OBS_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"continuous branch table is missing: {sorted(missing)}")
    if ridge < 0 or rank_beta < 0 or rank_margin < 0 or label_noise_std < 0:
        raise ValueError("invalid continuous DAVS hyperparameter")
    training = frame[frame.split == split].copy()
    if training.empty:
        raise ValueError(f"no rows are assigned to split {split}")
    training = _state_subset(training, data_fraction, seed)
    observations = training.loc[:, OBS_COLUMNS].to_numpy(dtype=np.float64)
    actions = training.action.to_numpy(dtype=np.int64)
    target = training.Q_branch.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    if label_noise_std > 0:
        target = target + rng.normal(
            0.0, label_noise_std * max(float(np.std(target)), 1e-8), size=len(target)
        )

    raw = _raw_features(observations, actions, len(action_costs))
    feature_mean = raw.mean(axis=0)
    feature_scale = raw.std(axis=0)
    feature_mean[0] = 0.0
    feature_scale[feature_scale < 1e-8] = 1.0
    design = (raw - feature_mean) / feature_scale
    target_mean = float(np.mean(target))
    target_scale = max(float(np.std(target)), 1e-8)
    standardized_target = (target - target_mean) / target_scale

    if rank_beta > 0:
        extra_x: list[np.ndarray] = []
        extra_y: list[float] = []
        target_frame = training[["state_id", "action"]].copy()
        target_frame["target"] = target
        lookup = dict(zip(training.index.to_numpy(), range(len(training))))
        weight = np.sqrt(rank_beta)
        for _, group in target_frame.groupby("state_id", sort=False):
            indices = [lookup[index] for index in group.index]
            for left_offset, left in enumerate(indices):
                for right in indices[left_offset + 1 :]:
                    gap = float(target[left] - target[right])
                    if abs(gap) <= 1e-12:
                        continue
                    extra_x.append(weight * (design[left] - design[right]))
                    signed = np.sign(gap) * max(abs(gap), rank_margin)
                    extra_y.append(weight * signed / target_scale)
        if extra_x:
            design = np.concatenate([design, np.stack(extra_x)], axis=0)
            standardized_target = np.concatenate(
                [standardized_target, np.asarray(extra_y, dtype=np.float64)]
            )

    gram = design.T @ design
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * ridge
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + regularizer, design.T @ standardized_target
    )
    return ContinuousDAVSModel(
        coefficients=coefficients,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        action_costs=np.asarray(action_costs, dtype=np.float64),
        budget_scale=float(budget_scale),
        ridge=float(ridge),
        rank_beta=float(rank_beta),
        training_states=int(training.state_id.nunique()),
    )


@dataclass(frozen=True)
class ContinuousEnsemblePrediction:
    mean: np.ndarray
    variance: np.ndarray
    votes: np.ndarray
    ranking_uncertainty: float


@dataclass(frozen=True)
class ContinuousDAVSEnsemble:
    models: tuple[ContinuousDAVSModel, ...]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        action_costs: np.ndarray,
        budget_scale: float,
        ridge: float,
        rank_beta: float,
        rank_margin: float,
        members: int,
        seed: int,
        *,
        data_fraction: float = 1.0,
        label_noise_std: float = 0.0,
    ) -> "ContinuousDAVSEnsemble":
        if members < 2:
            raise ValueError("an ensemble requires at least two members")
        training = frame[frame.split == "train"]
        states = training.state_id.drop_duplicates().to_numpy()
        groups = {state: group for state, group in training.groupby("state_id", sort=False)}
        rng = np.random.default_rng(seed)
        models: list[ContinuousDAVSModel] = []
        for member in range(members):
            selected = rng.choice(states, size=len(states), replace=True)
            pieces = []
            for draw_index, state in enumerate(selected):
                piece = groups[state].copy()
                piece["state_id"] = piece.state_id.astype(str) + f"|boot={draw_index}"
                pieces.append(piece)
            boot = pd.concat(pieces, ignore_index=True)
            models.append(
                fit_continuous_davs(
                    boot,
                    action_costs,
                    budget_scale,
                    ridge,
                    rank_beta,
                    rank_margin,
                    data_fraction=data_fraction,
                    label_noise_std=label_noise_std,
                    seed=seed + member * 1009,
                )
            )
        return cls(tuple(models))

    def predict(self, observation: np.ndarray) -> ContinuousEnsemblePrediction:
        scores = np.stack([model.predict_all(observation) for model in self.models])
        feasible = np.flatnonzero(np.isfinite(scores[0]))
        mean = np.full(scores.shape[1], np.nan, dtype=np.float64)
        variance = np.full(scores.shape[1], np.nan, dtype=np.float64)
        mean[feasible] = scores[:, feasible].mean(axis=0)
        variance[feasible] = scores[:, feasible].var(axis=0)
        votes = np.zeros(scores.shape[1], dtype=np.float64)
        for row in scores:
            votes[int(np.nanargmax(row))] += 1.0
        votes /= len(scores)
        pair_uncertainty: list[float] = []
        for left_offset, left in enumerate(feasible):
            for right in feasible[left_offset + 1 :]:
                probability = float(np.mean(scores[:, left] > scores[:, right]))
                pair_uncertainty.append(4.0 * probability * (1.0 - probability))
        vote_uncertainty = 1.0 - float(votes.max())
        pair_value = float(np.mean(pair_uncertainty)) if pair_uncertainty else 0.0
        ordered = np.sort(mean[feasible])
        gap = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else np.inf
        scale = float(np.sqrt(np.nanmax(variance[feasible]))) if len(feasible) else 0.0
        variance_ratio = scale / (scale + max(gap, 0.0) + 1e-12)
        return ContinuousEnsemblePrediction(
            mean=mean,
            variance=variance,
            votes=votes,
            ranking_uncertainty=float(
                np.clip(max(vote_uncertainty, pair_value, variance_ratio), 0.0, 1.0)
            ),
        )


@dataclass(frozen=True)
class ContinuousDAVSOutput:
    action: torch.Tensor
    q_values: torch.Tensor
    ranking_uncertainty: torch.Tensor


class ContinuousDAVSAgent:
    def __init__(
        self, scorer: ContinuousDAVSModel | ContinuousDAVSEnsemble
    ) -> None:
        self.scorer = scorer

    def reset_budget_controller(self) -> None:
        return None

    def act(
        self, observation: torch.Tensor, deterministic: bool = True, **kwargs
    ) -> ContinuousDAVSOutput:
        del deterministic, kwargs
        if observation.ndim != 2 or observation.shape[1] != len(OBS_COLUMNS):
            raise ValueError("continuous observation must have shape [batch, 14]")
        actions: list[int] = []
        values: list[np.ndarray] = []
        uncertainties: list[float] = []
        for row in observation.detach().cpu().numpy():
            if isinstance(self.scorer, ContinuousDAVSEnsemble):
                prediction = self.scorer.predict(row)
                scores = prediction.mean
                uncertainty = prediction.ranking_uncertainty
            else:
                scores = self.scorer.predict_all(row)
                uncertainty = 0.0
            actions.append(int(np.nanargmax(scores)))
            values.append(scores)
            uncertainties.append(uncertainty)
        return ContinuousDAVSOutput(
            action=torch.as_tensor(actions, dtype=torch.long, device=observation.device),
            q_values=torch.as_tensor(
                np.stack(values), dtype=observation.dtype, device=observation.device
            ),
            ranking_uncertainty=torch.as_tensor(
                uncertainties, dtype=observation.dtype, device=observation.device
            ),
        )
