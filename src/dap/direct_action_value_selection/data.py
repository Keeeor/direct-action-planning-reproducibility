from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from dap.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)


REQUIRED_COLUMNS = (
    "scenario",
    "state",
    "state_id",
    "split",
    "t",
    "load",
    "queue",
    "remaining_budget",
    "remaining_horizon",
    "branch_replication",
    "action",
    "action_cost",
    "Q_branch",
    "Q_star",
    "optimal_action",
    "first_uniform",
    "random_tape_sha256",
)


def _split_lookup(split_by_t: dict[str, list[int]], horizon: int) -> dict[int, str]:
    required = {"train", "validation", "test", "embargo"}
    if set(split_by_t) != required:
        raise ValueError(f"time split must have exactly {sorted(required)}")
    lookup: dict[int, str] = {}
    for split, times in split_by_t.items():
        for raw_t in times:
            t = int(raw_t)
            if not 0 <= t < horizon:
                raise ValueError("time split contains an index outside the horizon")
            if t in lookup:
                raise ValueError(f"time index {t} occurs in multiple splits")
            lookup[t] = split
    if set(lookup) != set(range(horizon)):
        missing = sorted(set(range(horizon)) - set(lookup))
        raise ValueError(f"time split does not cover the horizon: {missing}")
    return lookup


def _uniform_hash(value: float) -> str:
    payload = np.asarray([value], dtype="<f8").tobytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def generate_k1_branch_data(
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    scenario: str,
    seed: int,
    replications: int,
    split_by_t: dict[str, list[int]],
) -> pd.DataFrame:
    """Generate one-step common-random-number targets for every feasible action."""

    if replications <= 0:
        raise ValueError("branch replications must be positive")
    expected_shape = (
        mdp.config.horizon,
        mdp.n_loads,
        mdp.config.max_queue + 1,
        mdp.config.max_budget + 1,
    )
    if optimum.actions.shape != expected_shape:
        raise ValueError("DP optimum does not match the supplied MDP")
    split_lookup = _split_lookup(split_by_t, mdp.config.horizon)
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    for t, load, queue, budget in np.ndindex(expected_shape):
        state_id = f"{scenario}|t={t}|load={load}|queue={queue}|budget={budget}"
        state_text = f"load={load};queue={queue}"
        for replication in range(replications):
            uniform = float(rng.random())
            probabilities = mdp.load_probabilities(t, load)
            next_load = int(
                min(
                    np.searchsorted(np.cumsum(probabilities), uniform, side="right"),
                    mdp.n_loads - 1,
                )
            )
            tape_hash = _uniform_hash(uniform)
            for action in np.flatnonzero(mdp.action_costs <= budget):
                action = int(action)
                next_queue, reward, metrics = mdp.outcome(queue, load, action)
                terminal_value = 0.0
                if t + 1 < mdp.config.horizon:
                    terminal_value = float(
                        optimum.values[
                            t + 1,
                            next_load,
                            next_queue,
                            budget - int(mdp.action_costs[action]),
                        ]
                    )
                q_branch = float(reward) + mdp.config.gamma * terminal_value
                rows.append(
                    {
                        "scenario": scenario,
                        "state": state_text,
                        "state_id": state_id,
                        "split": split_lookup[t],
                        "t": t,
                        "load": load,
                        "queue": queue,
                        "remaining_budget": budget,
                        "remaining_horizon": mdp.config.horizon - t,
                        "branch_replication": replication,
                        "action": action,
                        "action_cost": int(mdp.action_costs[action]),
                        "Q_branch": q_branch,
                        "Q_star": float(optimum.q_values[t, load, queue, budget, action]),
                        "optimal_action": int(optimum.actions[t, load, queue, budget]),
                        "immediate_reward": float(reward),
                        "next_load": next_load,
                        "next_queue": next_queue,
                        "terminal_value": terminal_value,
                        "first_uniform": uniform,
                        "random_tape_sha256": tape_hash,
                        "k": 1,
                        "served": float(metrics["served"]),
                        "slo_violation": float(metrics["slo_violation"]),
                    }
                )
    return pd.DataFrame(rows)


def validate_split_integrity(
    frame: pd.DataFrame,
    split_by_t: dict[str, list[int]],
    horizon: int,
    action_costs: np.ndarray | None = None,
) -> dict[str, object]:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"branch table is missing columns: {sorted(missing)}")
    lookup = _split_lookup(split_by_t, horizon)
    cross_split = int((frame.groupby("state_id").split.nunique() > 1).sum())
    observed_mismatch = int(
        sum(
            int(row.split != lookup[int(row.t)])
            for row in frame[["t", "split"]].drop_duplicates().itertuples(index=False)
        )
    )
    active = {"train", "validation", "test"}
    adjacent = 0
    for t in range(horizon - 1):
        left, right = lookup[t], lookup[t + 1]
        if left in active and right in active and left != right:
            adjacent += 1
    common_random_failures = int(
        (
            frame.groupby(["state_id", "branch_replication"])[
                ["first_uniform", "random_tape_sha256"]
            ].nunique()
            > 1
        ).any(axis=1).sum()
    )
    incomplete_action_groups = 0
    if action_costs is not None:
        costs = np.asarray(action_costs, dtype=int)
        grouped = frame.groupby(["state_id", "branch_replication"])
        for (_, _), group in grouped:
            budget = int(group.remaining_budget.iloc[0])
            expected = set(np.flatnonzero(costs <= budget).tolist())
            if set(group.action.astype(int)) != expected:
                incomplete_action_groups += 1
    checks = {
        "cross_split_state_count": cross_split,
        "split_assignment_mismatch_count": observed_mismatch,
        "adjacent_active_split_count": adjacent,
        "common_random_number_failure_count": common_random_failures,
        "incomplete_action_group_count": incomplete_action_groups,
    }
    return {
        "schema": "direct_action_value_selection.split_integrity.v1",
        "status": "PASS" if not any(checks.values()) else "FAIL",
        **checks,
        "rows": int(len(frame)),
        "states": int(frame.state_id.nunique()),
        "split_state_counts": {
            key: int(value)
            for key, value in frame.groupby("split").state_id.nunique().items()
        },
    }
