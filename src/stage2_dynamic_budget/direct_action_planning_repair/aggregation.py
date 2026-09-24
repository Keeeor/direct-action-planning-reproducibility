from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch

from stage2_dynamic_budget.action_conditioned_budget_advantage.branching import (
    BranchableDiscreteEnv,
)
from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)
from stage2_dynamic_budget.direct_action_planning.planning import BudgetValueTable, one_step_plan


def collect_closed_loop_branch_labels(
    agent,
    mdp: ActionConditionedBudgetMDP,
    value: BudgetValueTable,
    optimum: ActionDPResult,
    budgets: list[int],
    scenario: str,
    seed: int,
    episodes: int,
    round_index: int,
    device: torch.device | str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the current planner and label every feasible action at each visited state."""

    device = torch.device(device)
    sample_rows: list[dict[str, object]] = []
    visit_rows: list[dict[str, object]] = []
    a0 = int(np.argmin(mdp.action_costs))
    for budget in budgets:
        for episode in range(episodes):
            rollout_seed = (
                seed
                + 1_700_003
                + round_index * 10_000_019
                + episode * 100_003
                + budget * 1_009
            )
            rng = np.random.default_rng(rollout_seed)
            env = BranchableDiscreteEnv(
                mdp.config, initial_budget=budget, budget_scale=mdp.config.max_budget
            )
            observation, _ = env.reset(seed=rollout_seed)
            for step in range(mdp.config.horizon):
                state = (env.t, env.load, env.queue, env.remaining_budget)
                uniform = float(rng.random())
                tape_hash = hashlib.sha256(np.asarray([uniform]).tobytes()).hexdigest()
                t, load, queue, remaining_budget = state
                lv = one_step_plan(mdp, value, t, load, queue, remaining_budget)
                base_queue, _, _ = mdp.outcome(queue, load, a0)
                probabilities = mdp.load_probabilities(t, load)
                next_load = int(
                    min(
                        np.searchsorted(np.cumsum(probabilities), uniform, side="right"),
                        mdp.n_loads - 1,
                    )
                )
                group = (
                    f"D{round_index}|{scenario}|s{seed}|b{budget}|e{episode}|t{step}"
                )
                for action in np.flatnonzero(mdp.action_costs <= remaining_budget):
                    action = int(action)
                    next_queue, reward, metrics = mdp.outcome(queue, load, action)
                    sample_rows.append(
                        {
                            "branch_group_id": group,
                            "scenario": scenario,
                            "seed": seed,
                            "split": "train",
                            "sample_index": episode,
                            "t": t,
                            "load": load,
                            "queue": queue,
                            "remaining_budget": remaining_budget,
                            "remaining_horizon": mdp.config.horizon - t,
                            "action": action,
                            "transition_uniform": uniform,
                            "true_next_load": next_load,
                            "true_next_queue": next_queue,
                            "base_next_load": next_load,
                            "base_next_queue": base_queue,
                            "action_effect_load": 0,
                            "action_effect_queue": next_queue - base_queue,
                            "reward": float(reward),
                            "cost": float(metrics["cost"]),
                            "q_lv": float(lv.q_values[action]),
                            "q_star": float(
                                optimum.q_values[t, load, queue, remaining_budget, action]
                            ),
                            "random_tape_sha256": tape_hash,
                            "source": f"D{round_index}_closed_loop",
                        }
                    )
                with torch.no_grad():
                    selected = int(
                        agent.act(
                            torch.as_tensor(
                                observation, dtype=torch.float32, device=device
                            ).unsqueeze(0),
                            deterministic=True,
                        ).action.item()
                    )
                visit_rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "round": round_index,
                        "budget": budget,
                        "episode": episode,
                        "t": t,
                        "load": load,
                        "queue": queue,
                        "remaining_budget": remaining_budget,
                        "action": selected,
                        "optimal_action": int(optimum.actions[state]),
                        "q_star_regret": float(
                            optimum.values[state] - optimum.q_values[state + (selected,)]
                        ),
                    }
                )
                observation, _, terminated, _, _ = env.step_with_uniform(selected, uniform)
                if terminated:
                    break
    return pd.DataFrame(sample_rows), pd.DataFrame(visit_rows)


def aggregate_branch_labels(
    base_labels: pd.DataFrame,
    base_samples: pd.DataFrame,
    aggregated_samples: list[pd.DataFrame],
) -> pd.DataFrame:
    """Update only train transition targets; keep validation and Q labels frozen."""

    if not aggregated_samples:
        result = base_labels.copy()
        result["aggregation_round"] = 0
        return result
    combined = pd.concat(
        [base_samples[base_samples.split == "train"], *aggregated_samples],
        ignore_index=True,
        sort=False,
    )
    keys = ["t", "load", "queue", "action"]
    counts = (
        combined.groupby(keys + ["true_next_load"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = counts.groupby(keys, as_index=False)["count"].sum().rename(
        columns={"count": "sample_count_aggregated"}
    )
    probability = counts.pivot_table(
        index=keys, columns="true_next_load", values="count", fill_value=0
    ).reset_index()
    for load in (0, 1, 2):
        if load not in probability.columns:
            probability[load] = 0
    probability = probability.merge(totals, on=keys, validate="one_to_one")
    for load in (0, 1, 2):
        probability[f"aggregated_prob_{load}"] = (
            probability[load] / probability.sample_count_aggregated
        )
    result = base_labels.copy()
    train_mask = result.split == "train"
    train = result.loc[train_mask].drop(
        columns=["next_load_prob_0", "next_load_prob_1", "next_load_prob_2"]
    )
    train = train.merge(
        probability[
            keys
            + [
                "aggregated_prob_0",
                "aggregated_prob_1",
                "aggregated_prob_2",
                "sample_count_aggregated",
            ]
        ],
        on=keys,
        how="left",
        validate="many_to_one",
    ).rename(
        columns={
            "aggregated_prob_0": "next_load_prob_0",
            "aggregated_prob_1": "next_load_prob_1",
            "aggregated_prob_2": "next_load_prob_2",
        }
    )
    validation = result.loc[~train_mask].copy()
    validation["sample_count_aggregated"] = validation.sample_count
    combined_labels = pd.concat([train, validation], ignore_index=True, sort=False)
    combined_labels["aggregation_round"] = len(aggregated_samples)
    return combined_labels


def visitation_distribution_distance(
    base_samples: pd.DataFrame, visits: pd.DataFrame
) -> float:
    """Return total variation between uniform D0 state support and rollout visitation."""

    keys = ["t", "load", "queue"]
    base_states = base_samples[keys].drop_duplicates()
    base = base_states.assign(base_mass=1.0 / len(base_states))
    visit = visits.groupby(keys, as_index=False).size()
    visit["visit_mass"] = visit["size"] / visit["size"].sum()
    joined = base.merge(visit[keys + ["visit_mass"]], on=keys, how="outer").fillna(0.0)
    return 0.5 * float(np.abs(joined.base_mass - joined.visit_mass).sum())
