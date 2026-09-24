from __future__ import annotations

import numpy as np
import pandas as pd

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)
from stage2_dynamic_budget.direct_action_planning.planning import (
    BudgetValueTable,
    one_step_plan,
)
from stage2_dynamic_budget.direct_action_planning_repair.model import (
    StructuredActionEffectModel,
)
from stage2_dynamic_budget.direct_action_planning_repair.planning import (
    structured_one_step_plan,
)

from .data import STATE_COLUMNS


def causal_decomposition(
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    fixed_value: BudgetValueTable,
    repaired_transition: StructuredActionEffectModel,
    source_states: dict[str, pd.DataFrame],
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    oracle = BudgetValueTable.from_exact_dp(optimum.values)
    rows: list[dict[str, object]] = []
    for source, frame in source_states.items():
        states = frame[STATE_COLUMNS].drop_duplicates()
        for item in states.itertuples(index=False):
            state = (
                int(item.t),
                int(item.load),
                int(item.queue),
                int(item.remaining_budget),
            )
            plans = {
                "true_transition_fixed_value": one_step_plan(mdp, fixed_value, *state),
                "repaired_transition_fixed_value": structured_one_step_plan(
                    mdp, fixed_value, repaired_transition, *state
                ),
                "repaired_transition_oracle_value": structured_one_step_plan(
                    mdp, oracle, repaired_transition, *state
                ),
                "true_transition_oracle_value": one_step_plan(mdp, oracle, *state),
            }
            optimal_action = int(optimum.actions[state])
            for planner, plan in plans.items():
                action = int(plan.action)
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": int(seed),
                        "source": source,
                        "t": state[0],
                        "load": state[1],
                        "queue": state[2],
                        "remaining_budget": state[3],
                        "remaining_horizon": mdp.config.horizon - state[0],
                        "planner": planner,
                        "action": action,
                        "optimal_action": optimal_action,
                        "action_consistency": action == optimal_action,
                        "q_star_regret": float(
                            optimum.values[state] - optimum.q_values[state + (action,)]
                        ),
                        **{
                            f"q_{candidate}": float(plan.q_values[candidate])
                            for candidate in range(mdp.n_actions)
                        },
                    }
                )
    return pd.DataFrame(rows)


def summarize_causal_decomposition(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"scenario", "seed", "source", "planner", "action_consistency", "q_star_regret"}
    if missing := required - set(rows.columns):
        raise ValueError(f"causal rows are missing columns: {sorted(missing)}")
    return (
        rows.groupby(["scenario", "seed", "source", "planner"], as_index=False)
        .agg(
            states=("action", "size"),
            action_consistency=("action_consistency", "mean"),
            q_star_regret=("q_star_regret", "mean"),
        )
    )


def causal_pair_flip_rates(
    rows: pd.DataFrame,
    optimum: ActionDPResult,
    n_actions: int,
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    keys = ["scenario", "seed", "source", "planner"]
    for group_key, frame in rows.groupby(keys, sort=False):
        indices = tuple(
            frame[column].to_numpy(dtype=np.int64)
            for column in ("t", "load", "queue", "remaining_budget")
        )
        truth = optimum.q_values[indices]
        for first in range(n_actions):
            for second in range(first + 1, n_actions):
                predicted_first = frame[f"q_{first}"].to_numpy(float)
                predicted_second = frame[f"q_{second}"].to_numpy(float)
                valid = (
                    np.isfinite(predicted_first)
                    & np.isfinite(predicted_second)
                    & np.isfinite(truth[:, first])
                    & np.isfinite(truth[:, second])
                    & (np.abs(truth[:, first] - truth[:, second]) > 1.0e-10)
                )
                flips = (
                    np.sign(predicted_first[valid] - predicted_second[valid])
                    != np.sign(truth[valid, first] - truth[valid, second])
                )
                output.append(
                    {
                        **dict(zip(keys, group_key)),
                        "action_pair": f"{first}>{second}",
                        "comparable_states": int(np.sum(valid)),
                        "flip_rate": float(np.mean(flips)) if np.any(valid) else np.nan,
                    }
                )
    return pd.DataFrame(output)
