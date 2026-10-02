from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dap.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
)

from .planning import BudgetValueTable


TRANSITION_COLUMNS = (
    "t",
    "load",
    "queue",
    "action",
    "next_load",
    "next_queue",
    "reward",
    "cost",
)


@dataclass(frozen=True)
class ActionModelPrediction:
    next_state_probabilities: np.ndarray
    reward: float
    cost: float


@dataclass(frozen=True)
class EmpiricalActionModel:
    next_load_probabilities: np.ndarray
    next_queue_probabilities: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray
    samples_per_state_action: int
    smoothing: float

    def predict(self, t: int, load: int, queue: int, action: int) -> ActionModelPrediction:
        load_probabilities = self.next_load_probabilities[t, load, queue, action]
        queue_probabilities = self.next_queue_probabilities[t, load, queue, action]
        joint = np.outer(load_probabilities, queue_probabilities)
        return ActionModelPrediction(
            next_state_probabilities=joint,
            reward=float(self.rewards[t, load, queue, action]),
            cost=float(self.costs[t, load, queue, action]),
        )


def collect_transition_samples(
    mdp: ActionConditionedBudgetMDP,
    samples_per_state_action: int,
    seed: int,
) -> pd.DataFrame:
    if samples_per_state_action <= 0:
        raise ValueError("samples_per_state_action must be positive")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for t, load, queue, action in np.ndindex(
        mdp.config.horizon,
        mdp.n_loads,
        mdp.config.max_queue + 1,
        mdp.n_actions,
    ):
        next_queue, reward, metrics = mdp.outcome(queue, load, action)
        probabilities = mdp.load_probabilities(t, load)
        next_loads = rng.choice(
            mdp.n_loads,
            size=samples_per_state_action,
            replace=True,
            p=probabilities,
        )
        for next_load in next_loads:
            rows.append(
                {
                    "t": t,
                    "load": load,
                    "queue": queue,
                    "action": action,
                    "next_load": int(next_load),
                    "next_queue": next_queue,
                    "reward": float(reward),
                    "cost": float(metrics["cost"]),
                }
            )
    return pd.DataFrame(rows, columns=TRANSITION_COLUMNS)


def fit_empirical_action_model(
    mdp: ActionConditionedBudgetMDP,
    samples: pd.DataFrame,
    smoothing: float,
) -> EmpiricalActionModel:
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    missing = set(TRANSITION_COLUMNS) - set(samples.columns)
    if missing:
        raise ValueError(f"transition samples are missing columns: {sorted(missing)}")
    shape = (
        mdp.config.horizon,
        mdp.n_loads,
        mdp.config.max_queue + 1,
        mdp.n_actions,
    )
    load_counts = np.full(shape + (mdp.n_loads,), float(smoothing), dtype=np.float64)
    queue_counts = np.zeros(
        shape + (mdp.config.max_queue + 1,), dtype=np.float64
    )
    reward_sums = np.zeros(shape, dtype=np.float64)
    cost_sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int64)
    for row in samples.itertuples(index=False):
        index = (int(row.t), int(row.load), int(row.queue), int(row.action))
        load_counts[index + (int(row.next_load),)] += 1.0
        queue_counts[index + (int(row.next_queue),)] += 1.0
        reward_sums[index] += float(row.reward)
        cost_sums[index] += float(row.cost)
        counts[index] += 1
    if np.any(counts == 0):
        raise ValueError("every time/load/queue/action cell requires transition samples")
    per_cell = np.unique(counts)
    if len(per_cell) != 1:
        raise ValueError("each state-action cell must have equal sample coverage")
    load_probabilities = load_counts / load_counts.sum(axis=-1, keepdims=True)
    queue_probabilities = queue_counts / queue_counts.sum(axis=-1, keepdims=True)
    rewards = reward_sums / counts
    costs = cost_sums / counts
    return EmpiricalActionModel(
        next_load_probabilities=load_probabilities,
        next_queue_probabilities=queue_probabilities,
        rewards=rewards,
        costs=costs,
        samples_per_state_action=int(per_cell[0]),
        smoothing=float(smoothing),
    )


def solve_empirical_value(
    mdp: ActionConditionedBudgetMDP,
    model: EmpiricalActionModel,
) -> BudgetValueTable:
    cfg = mdp.config
    values = np.zeros(
        (cfg.horizon + 1, mdp.n_loads, cfg.max_queue + 1, cfg.max_budget + 1),
        dtype=np.float64,
    )
    for remaining_horizon in range(1, cfg.horizon + 1):
        t = cfg.horizon - remaining_horizon
        for load, queue, budget in np.ndindex(
            mdp.n_loads, cfg.max_queue + 1, cfg.max_budget + 1
        ):
            action_values = np.full(mdp.n_actions, np.nan, dtype=np.float64)
            for action in range(mdp.n_actions):
                true_cost = int(mdp.action_costs[action])
                if true_cost > budget:
                    continue
                prediction = model.predict(t, load, queue, action)
                predicted_cost = int(np.rint(max(prediction.cost, 0.0)))
                next_budget = max(budget - predicted_cost, 0)
                continuation = float(
                    np.sum(
                        prediction.next_state_probabilities
                        * values[
                            remaining_horizon - 1,
                            :,
                            :,
                            next_budget,
                        ]
                    )
                )
                action_values[action] = prediction.reward + cfg.gamma * continuation
            values[remaining_horizon, load, queue, budget] = np.nanmax(action_values)
    return BudgetValueTable(values=values, source="empirical_model_bellman")
