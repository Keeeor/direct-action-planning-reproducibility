from __future__ import annotations

import time
from typing import Any

import numpy as np

from stage2_dynamic_budget.direct_action_planning_dataset_validation.data import (
    TraceDataset,
    make_trace_env,
)


def evaluate_agent(
    dataset: TraceDataset,
    agent: Any,
    *,
    split: str,
    horizon: int,
    budget: float,
    seed: int,
    episodes_per_domain: int,
    gamma: float = 0.99,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one method on paired causal trace windows."""
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(dataset.domain_names):
        for episode in range(episodes_per_domain):
            window_seed = seed + domain_index * 1_000_003 + episode * 100_003
            env, window_start = make_trace_env(
                dataset,
                domain,
                split,
                horizon=horizon,
                budget=budget,
                window_seed=window_seed,
            )
            observation, _ = env.reset(seed=window_seed)
            if hasattr(agent, "reset"):
                agent.reset()
            rewards: list[float] = []
            served: list[float] = []
            arrivals: list[float] = []
            costs: list[float] = []
            queues: list[float] = []
            slos: list[float] = []
            latencies: list[float] = []
            step_index = 0
            while True:
                started = time.perf_counter_ns()
                action, scores = agent.select(env, observation)
                decision_ms = (time.perf_counter_ns() - started) / 1.0e6
                remaining = max(budget - env.cumulative_cost, 0.0)
                if env.action_costs[action] > remaining + 1.0e-8:
                    raise AssertionError(f"{agent.name} selected an unaffordable action")
                next_observation, reward, terminated, truncated, info = env.step(int(action))
                rewards.append(float(reward))
                served.append(float(info["served"]))
                arrivals.append(float(info["arrivals"]))
                costs.append(float(info["resource_cost"]))
                queues.append(float(info["queue_length"]))
                slos.append(float(info["slo_violation"]))
                latencies.append(float(decision_ms))
                row = {
                    "dataset": dataset.name,
                    "domain": domain,
                    "split": split,
                    "method": agent.name,
                    "budget": float(budget),
                    "seed": int(seed),
                    "episode": int(episode),
                    "window_seed": int(window_seed),
                    "window_start": int(window_start),
                    "step": int(step_index),
                    "action": int(action),
                    "reward": float(reward),
                    "served": float(info["served"]),
                    "arrivals": float(info["arrivals"]),
                    "cost": float(info["resource_cost"]),
                    "remaining_budget": float(info["remaining_budget"]),
                    "queue": float(info["queue_length"]),
                    "slo_violation": float(info["slo_violation"]),
                    "decision_ms": float(decision_ms),
                }
                for action_index, value in enumerate(np.asarray(scores, dtype=np.float64)):
                    row[f"score_{action_index}"] = float(value)
                steps.append(row)
                observation = next_observation
                step_index += 1
                if terminated or truncated:
                    break
            rewards_array = np.asarray(rewards, dtype=np.float64)
            total_arrivals = float(np.sum(arrivals))
            total_cost = float(np.sum(costs))
            episodes.append({
                "dataset": dataset.name,
                "domain": domain,
                "split": split,
                "method": agent.name,
                "budget": float(budget),
                "seed": int(seed),
                "episode": int(episode),
                "window_seed": int(window_seed),
                "window_start": int(window_start),
                "return": float(rewards_array.sum()),
                "discounted_return": float(np.sum((gamma ** np.arange(len(rewards_array))) * rewards_array)),
                "completion_ratio": float(np.sum(served)) / max(total_arrivals, 1.0),
                "slo_violation_rate": float(np.mean(slos)),
                "total_cost": total_cost,
                "budget_overspend": max(total_cost - float(budget), 0.0),
                "queue_area": float(np.sum(queues)),
                "final_queue": float(queues[-1]),
                "decision_ms_mean": float(np.mean(latencies)),
                "decision_ms_p95": float(np.quantile(latencies, 0.95)),
            })
    return episodes, steps

