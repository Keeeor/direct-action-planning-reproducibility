from __future__ import annotations

import numpy as np
import pandas as pd

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)

from .environment import make_branchable_env


STATE_COLUMNS = ["t", "load", "queue", "remaining_budget"]


def full_state_value_frame(
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    scenario_id: str,
    *,
    region: str = "unspecified",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for t, load, queue, budget in np.ndindex(optimum.actions.shape):
        state = (t, load, queue, budget)
        row: dict[str, object] = {
            "scenario_id": scenario_id,
            "region": region,
            "t": t,
            "load": load,
            "queue": queue,
            "remaining_budget": budget,
            "remaining_horizon": mdp.config.horizon - t,
            "target_value": float(optimum.values[state]),
            "optimal_action": int(optimum.actions[state]),
        }
        for action in range(mdp.n_actions):
            value = float(optimum.q_values[state + (action,)])
            row[f"q_star_a{action}"] = value if np.isfinite(value) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def sample_value_states(
    frame: pd.DataFrame,
    *,
    count: int,
    seed: int,
) -> pd.DataFrame:
    if count <= 0:
        raise ValueError("sample count must be positive")
    if len(frame) <= count:
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=count, random_state=seed, replace=False).reset_index(drop=True)


def collect_value_trajectories(
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    *,
    scenario_id: str,
    region: str,
    budgets: list[int],
    seed: int,
    episodes: int,
) -> pd.DataFrame:
    if not budgets or episodes <= 0:
        raise ValueError("budgets and positive episode count are required")
    rows: list[dict[str, object]] = []
    for episode in range(episodes):
        budget = int(budgets[episode % len(budgets)])
        eval_seed = seed * 1_000_003 + episode * 100_003 + 29
        rng = np.random.default_rng(eval_seed)
        tape = rng.random(mdp.config.horizon)
        env = make_branchable_env(mdp, budget)
        env.reset(seed=eval_seed)
        trajectory_id = f"{scenario_id}:s{seed}:e{episode}:b{budget}"
        behavior = "exact_dp" if episode % 2 == 0 else "random_feasible"
        for uniform in tape:
            state = (env.t, env.load, env.queue, env.remaining_budget)
            t, load, queue, remaining_budget = map(int, state)
            row: dict[str, object] = {
                "scenario_id": scenario_id,
                "region": region,
                "collection_seed": seed,
                "eval_seed": eval_seed,
                "episode": episode,
                "trajectory_id": trajectory_id,
                "row_id": f"{trajectory_id}:t{t}",
                "behavior": behavior,
                "budget": budget,
                "t": t,
                "load": load,
                "queue": queue,
                "remaining_budget": remaining_budget,
                "remaining_horizon": mdp.config.horizon - t,
                "target_value": float(optimum.values[state]),
                "optimal_action": int(optimum.actions[state]),
            }
            for action in range(mdp.n_actions):
                value = float(optimum.q_values[state + (action,)])
                row[f"q_star_a{action}"] = value if np.isfinite(value) else np.nan
            rows.append(row)
            feasible = np.flatnonzero(mdp.action_costs <= remaining_budget)
            if behavior == "exact_dp":
                action = int(optimum.actions[state])
            else:
                action = int(rng.choice(feasible))
            _, _, terminated, _, _ = env.step_with_uniform(action, float(uniform))
            if terminated:
                break
    frame = pd.DataFrame(rows)
    if not frame.row_id.is_unique:
        raise RuntimeError("trajectory row identifiers are not unique")
    return frame
