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


def collect_mixed_policy_branches(
    repair_agent,
    reference_agent,
    mdp: ActionConditionedBudgetMDP,
    value: BudgetValueTable,
    optimum: ActionDPResult,
    budgets: list[int],
    scenario: str,
    seed: int,
    episodes: int,
    round_index: int,
    rho: float,
    previous_samples: list[pd.DataFrame] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    device = torch.device(device)
    prior = pd.concat(previous_samples, ignore_index=True) if previous_samples else pd.DataFrame()
    prior_counts = (
        prior.groupby(["t", "load", "queue"]).size() if not prior.empty else pd.Series(dtype=float)
    )
    low_coverage_cutoff = float(prior_counts.quantile(0.25)) if len(prior_counts) else -1.0
    sample_rows: list[dict[str, object]] = []
    visit_rows: list[dict[str, object]] = []
    a0 = int(np.argmin(mdp.action_costs))
    for budget in budgets:
        for episode in range(episodes):
            rollout_seed = (
                seed + 2_300_003 + round_index * 10_000_019 + episode * 100_003 + budget * 1_009
            )
            rng = np.random.default_rng(rollout_seed)
            env = BranchableDiscreteEnv(
                mdp.config, initial_budget=budget, budget_scale=mdp.config.max_budget
            )
            observation, _ = env.reset(seed=rollout_seed)
            for step in range(mdp.config.horizon):
                state = (env.t, env.load, env.queue, env.remaining_budget)
                t, load, queue, remaining_budget = state
                transition_uniform = float(rng.random())
                mixture_uniform = float(rng.random())
                use_repair = mixture_uniform < rho
                selected_agent = repair_agent if use_repair else reference_agent
                with torch.no_grad():
                    selected = int(
                        selected_agent.act(
                            torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0),
                            deterministic=True,
                        ).action.item()
                    )
                lv = one_step_plan(mdp, value, t, load, queue, remaining_budget)
                qstar_regret = float(
                    optimum.values[state]
                    - optimum.q_values[t, load, queue, remaining_budget, selected]
                )
                ranking_error = selected != lv.action
                low_coverage = bool(
                    len(prior_counts)
                    and prior_counts.get((t, load, queue), 0.0) <= low_coverage_cutoff
                )
                late_burst_priority = scenario == "late_burst" and t >= mdp.config.horizon // 2
                tight_budget = remaining_budget <= mdp.config.max_budget // 3
                high_regret = False
                selective_keep = bool(
                    ranking_error or low_coverage or late_burst_priority or tight_budget
                )
                group = f"D{round_index}|{scenario}|s{seed}|b{budget}|e{episode}|t{step}"
                probabilities = mdp.load_probabilities(t, load)
                next_load = int(
                    min(
                        np.searchsorted(
                            np.cumsum(probabilities), transition_uniform, side="right"
                        ),
                        mdp.n_loads - 1,
                    )
                )
                base_queue, _, _ = mdp.outcome(queue, load, a0)
                tape_hash = hashlib.sha256(
                    np.asarray([transition_uniform], dtype=np.float64).tobytes()
                ).hexdigest()
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
                            "transition_uniform": transition_uniform,
                            "mixture_uniform": mixture_uniform,
                            "collector_component": "repair" if use_repair else "reference",
                            "collector_action": selected,
                            "lv_action": lv.action,
                            "collector_q_star_regret": qstar_regret,
                            "ranking_error": ranking_error,
                            "high_regret": high_regret,
                            "low_coverage": low_coverage,
                            "late_burst_priority": late_burst_priority,
                            "tight_budget_priority": tight_budget,
                            "selective_keep": selective_keep,
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
                            "source": f"D{round_index}_controlled_closed_loop",
                        }
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
                        "lv_action": lv.action,
                        "optimal_action": int(optimum.actions[state]),
                        "q_star_regret": qstar_regret,
                        "collector_component": "repair" if use_repair else "reference",
                        "selective_keep": selective_keep,
                    }
                )
                observation, _, terminated, _, _ = env.step_with_uniform(
                    selected, transition_uniform
                )
                if terminated:
                    break
    samples = pd.DataFrame(sample_rows)
    visits = pd.DataFrame(visit_rows)
    positive = visits.loc[visits.q_star_regret > 0.0, "q_star_regret"]
    regret_threshold = float(positive.quantile(0.75)) if len(positive) else np.inf
    high_groups = set(
        visits.loc[visits.q_star_regret > regret_threshold]
        .assign(
            branch_group_id=lambda frame: [
                f"D{round_index}|{scenario}|s{seed}|b{row.budget}|e{row.episode}|t{row.t}"
                for row in frame.itertuples(index=False)
            ]
        )
        .branch_group_id
    )
    samples["high_regret"] = samples.branch_group_id.isin(high_groups)
    samples["selective_keep"] = samples.selective_keep | samples.high_regret
    visits["high_regret"] = visits.q_star_regret > regret_threshold
    visits["selective_keep"] = visits.selective_keep | visits.high_regret
    visits["high_regret_threshold"] = regret_threshold
    return samples, visits


def selective_samples(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    keep_groups = samples.loc[samples.selective_keep, "branch_group_id"].unique()
    return samples[samples.branch_group_id.isin(keep_groups)].reset_index(drop=True)
