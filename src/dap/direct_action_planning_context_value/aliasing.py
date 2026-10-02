from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from dap.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)


def future_load_window(
    mdp: ActionConditionedBudgetMDP,
    t: int,
    current_load: int,
    steps: int,
) -> np.ndarray:
    """Expected arrival window used only for post-hoc alias diagnosis."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    distribution = np.zeros(mdp.n_loads, dtype=np.float64)
    distribution[current_load] = 1.0
    expected: list[float] = []
    for offset in range(steps):
        expected.append(float(distribution @ mdp.load_arrivals))
        if t + offset >= mdp.config.horizon - 1:
            continue
        next_distribution = np.zeros_like(distribution)
        for load, mass in enumerate(distribution):
            if mass:
                next_distribution += mass * mdp.load_probabilities(t + offset, load)
        distribution = next_distribution
    return np.asarray(expected, dtype=np.float64)


def oracle_phase(mdp: ActionConditionedBudgetMDP, t: int) -> np.ndarray:
    """Privileged phase descriptor. It is prohibited from formal online models."""

    if not 0 <= t < mdp.config.horizon:
        raise ValueError("t is outside the active horizon")
    phase = t / max(mdp.config.horizon - 1, 1)
    scenario = mdp.config.scenario
    if scenario == "periodic":
        angle = 6.0 * np.pi * phase - np.pi / 2.0
        load_intensity = 0.10 + 0.55 * 0.5 * (1.0 + np.sin(angle))
        trend = 0.55 * 0.5 * 6.0 * np.pi * np.cos(angle)
        family = (0.0, 0.0, 1.0)
    else:
        center = 0.25 if scenario == "early_burst" else 0.75
        z = (phase - center) / 0.11
        burst = np.exp(-0.5 * z * z)
        load_intensity = 0.65 * burst
        trend = -0.65 * z * burst / 0.11
        family = (1.0, 0.0, 0.0) if scenario == "early_burst" else (0.0, 1.0, 0.0)
    return np.asarray((*family, phase, load_intensity, trend), dtype=np.float64)


def build_alias_table(
    mdps: dict[str, ActionConditionedBudgetMDP],
    optima: dict[str, ActionDPResult],
    future_steps: int = 4,
    material_q_gap: float = 1.0e-8,
) -> pd.DataFrame:
    if set(mdps) != set(optima) or len(mdps) < 2:
        raise ValueError("matching MDP and optimum maps for at least two scenarios are required")
    scenarios = sorted(mdps)
    first = mdps[scenarios[0]]
    expected_shape = (
        first.config.horizon,
        first.n_loads,
        first.config.max_queue + 1,
        first.config.max_budget + 1,
    )
    if any(optima[name].actions.shape != expected_shape for name in scenarios):
        raise ValueError("all scenario grids must match")
    rows: list[dict[str, object]] = []
    for t, load, queue, budget in np.ndindex(expected_shape):
        values = np.asarray(
            [optima[name].values[t, load, queue, budget] for name in scenarios],
            dtype=np.float64,
        )
        actions = np.asarray(
            [optima[name].actions[t, load, queue, budget] for name in scenarios],
            dtype=np.int64,
        )
        future = {
            name: future_load_window(mdps[name], t, load, future_steps) for name in scenarios
        }
        future_difference = max(
            float(np.max(np.abs(future[left] - future[right])))
            for left, right in combinations(scenarios, 2)
        )
        row: dict[str, object] = {
            "t": t,
            "load": load,
            "queue": queue,
            "remaining_budget": budget,
            "remaining_horizon": first.config.horizon - t,
            "cross_scenario_value_variance": float(np.var(values)),
            "cross_scenario_value_range": float(np.ptp(values)),
            "optimal_action_conflict": bool(np.unique(actions).size > 1),
            "max_future_window_difference": future_difference,
        }
        material_conflict = False
        for index, name in enumerate(scenarios):
            optimum = optima[name]
            optimal_action = int(actions[index])
            feasible = optimum.q_values[t, load, queue, budget]
            ordered = np.sort(feasible[np.isfinite(feasible)])[::-1]
            gap = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0
            row[f"V_star__{name}"] = values[index]
            row[f"optimal_action__{name}"] = optimal_action
            row[f"optimal_gap__{name}"] = gap
            row[f"future_window__{name}"] = "|".join(f"{value:.8f}" for value in future[name])
            for action in range(first.n_actions):
                q = feasible[action]
                row[f"Q_star_a{action}__{name}"] = float(q) if np.isfinite(q) else np.nan
            for other_action in np.unique(actions):
                if int(other_action) == optimal_action:
                    continue
                alternative = feasible[int(other_action)]
                if np.isfinite(alternative) and values[index] - alternative > material_q_gap:
                    material_conflict = True
        row["material_action_conflict"] = material_conflict
        for action in range(first.n_actions):
            q_values = np.asarray(
                [row[f"Q_star_a{action}__{name}"] for name in scenarios], dtype=float
            )
            finite = q_values[np.isfinite(q_values)]
            row[f"cross_scenario_Q_range_a{action}"] = (
                float(np.ptp(finite)) if len(finite) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def alias_summary(table: pd.DataFrame) -> dict[str, float | int]:
    required = {
        "cross_scenario_value_variance",
        "optimal_action_conflict",
        "material_action_conflict",
        "max_future_window_difference",
    }
    if missing := required - set(table.columns):
        raise ValueError(f"alias table missing columns: {sorted(missing)}")
    return {
        "exact_alias_states": int(len(table)),
        "mean_cross_scenario_value_variance": float(
            table.cross_scenario_value_variance.mean()
        ),
        "median_cross_scenario_value_variance": float(
            table.cross_scenario_value_variance.median()
        ),
        "optimal_action_conflict_rate": float(table.optimal_action_conflict.mean()),
        "material_action_conflict_rate": float(table.material_action_conflict.mean()),
        "mean_future_window_difference": float(
            table.max_future_window_difference.mean()
        ),
    }


def build_near_alias_pairs(
    mdps: dict[str, ActionConditionedBudgetMDP],
    optima: dict[str, ActionDPResult],
    radius: float,
) -> pd.DataFrame:
    """Enumerate local cross-scenario pairs without treating them as exact aliases."""

    if not 0.0 < radius <= 1.0:
        raise ValueError("radius must be in (0, 1]")
    scenarios = sorted(mdps)
    first = mdps[scenarios[0]]
    cfg = first.config
    spans = (cfg.horizon, max(first.n_loads - 1, 1), cfg.max_queue, cfg.max_budget)
    limits = tuple(int(np.floor(radius * span + 1.0e-12)) for span in spans)
    offsets = [
        (dt, dl, dq, db)
        for dt in range(-limits[0], limits[0] + 1)
        for dl in range(-limits[1], limits[1] + 1)
        for dq in range(-limits[2], limits[2] + 1)
        for db in range(-limits[3], limits[3] + 1)
        if (dt, dl, dq, db) != (0, 0, 0, 0)
        and max(
            abs(dt) / spans[0],
            abs(dl) / spans[1],
            abs(dq) / spans[2],
            abs(db) / spans[3],
        )
        <= radius + 1.0e-12
    ]
    rows: list[dict[str, object]] = []
    shape = (cfg.horizon, first.n_loads, cfg.max_queue + 1, cfg.max_budget + 1)
    for left_scenario, right_scenario in combinations(scenarios, 2):
        left_optimum, right_optimum = optima[left_scenario], optima[right_scenario]
        for t, load, queue, budget in np.ndindex(shape):
            for dt, dl, dq, db in offsets:
                other = (t + dt, load + dl, queue + dq, budget + db)
                if not (
                    0 <= other[0] < cfg.horizon
                    and 0 <= other[1] < first.n_loads
                    and 0 <= other[2] <= cfg.max_queue
                    and 0 <= other[3] <= cfg.max_budget
                ):
                    continue
                distance = max(
                    abs(dt) / spans[0],
                    abs(dl) / spans[1],
                    abs(dq) / spans[2],
                    abs(db) / spans[3],
                )
                left_action = int(left_optimum.actions[t, load, queue, budget])
                right_action = int(right_optimum.actions[other])
                rows.append(
                    {
                        "left_scenario": left_scenario,
                        "right_scenario": right_scenario,
                        "t": t,
                        "load": load,
                        "queue": queue,
                        "remaining_budget": budget,
                        "other_t": other[0],
                        "other_load": other[1],
                        "other_queue": other[2],
                        "other_remaining_budget": other[3],
                        "normalized_linf_distance": distance,
                        "V_star_absolute_difference": abs(
                            float(left_optimum.values[t, load, queue, budget])
                            - float(right_optimum.values[other])
                        ),
                        "optimal_action_conflict": left_action != right_action,
                    }
                )
    return pd.DataFrame(rows)
