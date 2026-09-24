from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)


STATE_COLUMNS = ["t", "load", "queue", "remaining_budget"]


def _states(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(STATE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"state frame is missing columns: {sorted(missing)}")
    return frame[STATE_COLUMNS].drop_duplicates().reset_index(drop=True)


def value_state_targets(
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for source, frame in sources.items():
        if frame.empty:
            continue
        states = _states(frame)
        index = tuple(states[column].to_numpy(dtype=np.int64) for column in STATE_COLUMNS)
        states["remaining_horizon"] = mdp.config.horizon - states.t
        states["target_value"] = optimum.values[index]
        states["source"] = source
        rows.append(states)
    if not rows:
        raise ValueError("at least one non-empty value source is required")
    return pd.concat(rows, ignore_index=True)


def build_anchor_states(
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    d0_states: pd.DataFrame,
    d1_states: pd.DataFrame,
    scenario: str,
    high_value_quantile: float,
) -> pd.DataFrame:
    if not 0.0 <= high_value_quantile <= 1.0:
        raise ValueError("high_value_quantile must be in [0, 1]")
    d0 = _states(d0_states)
    d1 = _states(d1_states)
    reasons: defaultdict[tuple[int, int, int, int], set[str]] = defaultdict(set)

    for state in d0.itertuples(index=False, name=None):
        reasons[tuple(map(int, state))].add("D0_fixed")

    if not d1.empty:
        index = tuple(d1[column].to_numpy(dtype=np.int64) for column in STATE_COLUMNS)
        values = optimum.values[index]
        cutoff = float(np.quantile(values, high_value_quantile))
        for state, value in zip(d1.itertuples(index=False, name=None), values):
            if value >= cutoff:
                reasons[tuple(map(int, state))].add("D1_high_value")

    risk_cutoff = float(
        np.quantile(
            [
                load + queue
                for load in range(mdp.n_loads)
                for queue in range(mdp.config.max_queue + 1)
            ],
            0.75,
        )
    )
    all_states = pd.concat([d0, d1], ignore_index=True).drop_duplicates()
    for row in all_states.itertuples(index=False):
        state = (int(row.t), int(row.load), int(row.queue), int(row.remaining_budget))
        if scenario == "late_burst" and row.t >= mdp.config.horizon // 2:
            reasons[state].add("late_burst")
        if row.remaining_budget <= mdp.config.max_budget // 3:
            reasons[state].add("tight_budget")
        if (
            row.load + row.queue >= risk_cutoff
            and int(optimum.actions[state]) == int(np.argmin(mdp.action_costs))
        ):
            reasons[state].add("high_risk_optimal_low_cost")

    rows = []
    for state, labels in sorted(reasons.items()):
        t, load, queue, budget = state
        rows.append(
            {
                "t": t,
                "load": load,
                "queue": queue,
                "remaining_budget": budget,
                "remaining_horizon": mdp.config.horizon - t,
                "target_value": float(optimum.values[state]),
                "anchor_reason": "|".join(sorted(labels)),
            }
        )
    return pd.DataFrame(rows)

