from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import torch

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)

from .models import PooledValueNetwork


SPACE_PREFIXES = {
    "raw": "raw_",
    "structured": "structured_",
    "hidden": "hidden_",
    "q_vector": "q_vector_",
}


def _columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith(prefix))


def _nearest_distance(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    if len(train) == 0 or len(test) == 0:
        raise ValueError("support distance requires non-empty train and test arrays")
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale > 1.0e-9, scale, 1.0)
    train_z = (train - mean) / scale
    test_z = (test - mean) / scale
    distances, _ = cKDTree(train_z).query(test_z, k=1, workers=1)
    return np.asarray(distances, dtype=np.float64)


def compute_support_distances(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    result = test.copy()
    for space, prefix in SPACE_PREFIXES.items():
        train_columns = _columns(train, prefix)
        test_columns = _columns(test, prefix)
        if not train_columns or train_columns != test_columns:
            raise ValueError(f"support feature mismatch for {space}")
        result[space] = _nearest_distance(
            train[train_columns].to_numpy(dtype=np.float64),
            test[test_columns].to_numpy(dtype=np.float64),
        )
    return result


@torch.no_grad()
def build_support_features(
    frame: pd.DataFrame,
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    model: PooledValueNetwork,
) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    raw = np.stack(
        [
            output.t.to_numpy(float) / max(mdp.config.horizon - 1, 1),
            output.load.to_numpy(float) / max(mdp.n_loads - 1, 1),
            output.queue.to_numpy(float) / max(mdp.config.max_queue, 1),
            output.remaining_budget.to_numpy(float) / max(mdp.config.max_budget, 1),
        ],
        axis=1,
    )
    for index in range(raw.shape[1]):
        output[f"raw_{index}"] = raw[:, index]

    structured_rows: list[np.ndarray] = []
    q_rows: list[np.ndarray] = []
    for row in output.itertuples(index=False):
        time_encoding = np.zeros(mdp.config.horizon, dtype=np.float64)
        time_encoding[int(row.t)] = 1.0
        load_encoding = np.zeros(mdp.n_loads, dtype=np.float64)
        load_encoding[int(row.load)] = 1.0
        probabilities = mdp.load_probabilities(int(row.t), int(row.load))
        effects = []
        for action in range(mdp.n_actions):
            next_queue, reward, _ = mdp.outcome(int(row.queue), int(row.load), action)
            effects.extend(
                [
                    next_queue / max(mdp.config.max_queue, 1),
                    reward / 6.0,
                    float(mdp.action_costs[action]) / max(float(mdp.action_costs.max()), 1.0),
                ]
            )
        structured_rows.append(
            np.concatenate([time_encoding, load_encoding, probabilities, np.asarray(effects)])
        )
        state = (int(row.t), int(row.load), int(row.queue), int(row.remaining_budget))
        q = optimum.q_values[state].astype(np.float64, copy=True)
        finite = np.isfinite(q)
        q[~finite] = float(np.min(q[finite]) - 1.0)
        q_rows.append(q)
    structured = np.stack(structured_rows)
    for index in range(structured.shape[1]):
        output[f"structured_{index}"] = structured[:, index]

    normalized = torch.as_tensor(
        np.stack(
            [
                output.remaining_horizon.to_numpy(float) / max(mdp.config.horizon, 1),
                raw[:, 1],
                raw[:, 2],
                raw[:, 3],
            ],
            axis=1,
        ),
        dtype=torch.float64,
    )
    hidden = model.hidden(normalized).cpu().numpy()
    for index in range(hidden.shape[1]):
        output[f"hidden_{index}"] = hidden[:, index]
    q_values = np.stack(q_rows)
    for index in range(q_values.shape[1]):
        output[f"q_vector_{index}"] = q_values[:, index]
    return output


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return 0.0
    value = spearmanr(left, right, nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else 0.0


def support_relationship(
    frame: pd.DataFrame,
    *,
    spearman_floor: float,
    regret_share_floor: float,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for space in SPACE_PREFIXES:
        threshold = float(frame[space].quantile(0.75))
        high = frame[space] >= threshold
        total_regret = float(frame.Q_star_regret.clip(lower=0).sum())
        share = (
            float(frame.loc[high, "Q_star_regret"].clip(lower=0).sum()) / total_regret
            if total_regret > 0
            else 0.0
        )
        regret_rho = _safe_spearman(frame[space], frame.Q_star_regret)
        error_rho = _safe_spearman(frame[space], frame.action_error)
        rows.append(
            {
                "space": space,
                "regret_spearman": regret_rho,
                "action_error_spearman": error_rho,
                "top_quartile_regret_share": share,
                "clear": bool(
                    max(abs(regret_rho), abs(error_rho)) >= spearman_floor
                    and share >= regret_share_floor
                ),
            }
        )
    deployable = [row for row in rows if row["space"] != "q_vector"]
    return {
        "schema": "direct_action_planning_support_adaptation.support_relationship.v1",
        "spaces": rows,
        "deployable_support_clear": any(bool(row["clear"]) for row in deployable),
        "oracle_q_support_clear": bool(next(row["clear"] for row in rows if row["space"] == "q_vector")),
    }
