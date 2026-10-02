from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dap.action_conditioned_budget_advantage.branching import (
    BranchableDiscreteEnv,
)
from dap.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)
from dap.direct_action_planning.planning import BudgetValueTable

from .history import CausalHistory, leakage_audit, serialize_history


def collect_context_trajectories(
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    base_value: BudgetValueTable,
    policies: dict[str, object | None],
    budgets: list[int],
    seed: int,
    episodes_per_source: int,
    device: torch.device,
) -> pd.DataFrame:
    if not policies or not budgets or episodes_per_source <= 0:
        raise ValueError("policies, budgets, and positive episodes are required")
    rows: list[dict[str, object]] = []
    for source_index, (source, policy) in enumerate(policies.items()):
        for episode in range(episodes_per_source):
            budget = int(budgets[episode % len(budgets)])
            eval_seed = seed * 1_000_003 + source_index * 100_003 + episode * 997 + 17
            rng = np.random.default_rng(eval_seed)
            tape = rng.random(mdp.config.horizon)
            env = BranchableDiscreteEnv(
                mdp.config, initial_budget=budget, budget_scale=mdp.config.max_budget
            )
            observation, _ = env.reset(seed=eval_seed)
            if policy is not None and hasattr(policy, "reset_budget_controller"):
                policy.reset_budget_controller()
            history = CausalHistory(
                arrivals=(float(mdp.load_arrivals[env.load]),),
                queues=(float(env.queue),),
            )
            for uniform in tape:
                state = (env.t, env.load, env.queue, env.remaining_budget)
                t, load, queue, remaining_budget = map(int, state)
                remaining_horizon = mdp.config.horizon - t
                target = float(optimum.values[state])
                base = base_value.predict(load, queue, remaining_budget, remaining_horizon)
                rows.append(
                    {
                        "scenario": mdp.config.scenario,
                        "source": source,
                        "collection_seed": seed,
                        "eval_seed": eval_seed,
                        "episode": episode,
                        "budget": budget,
                        "t": t,
                        "load": load,
                        "queue": queue,
                        "remaining_budget": remaining_budget,
                        "remaining_horizon": remaining_horizon,
                        "target_value": target,
                        "base_value": base,
                        "target_residual": target - base,
                        "optimal_action": int(optimum.actions[state]),
                        **serialize_history(history),
                    }
                )
                feasible = np.flatnonzero(mdp.action_costs <= remaining_budget)
                if policy is None:
                    action = int(rng.choice(feasible))
                else:
                    with torch.no_grad():
                        output = policy.act(
                            torch.as_tensor(
                                observation, dtype=torch.float32, device=device
                            ).unsqueeze(0),
                            deterministic=True,
                        )
                    action = int(output.action.item())
                next_observation, _, terminated, _, _ = env.step_with_uniform(
                    action, float(uniform)
                )
                if terminated:
                    break
                history = history.advance(
                    action,
                    int(mdp.action_capacity[action]),
                    int(mdp.load_arrivals[env.load]),
                    int(env.queue),
                )
                observation = next_observation
    frame = pd.DataFrame(rows)
    audit = leakage_audit(frame)
    if not audit["passed"]:
        raise RuntimeError(f"context collection violated causal history: {audit['violations'][:3]}")
    return frame
