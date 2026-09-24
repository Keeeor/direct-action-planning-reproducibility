from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .data import TraceDataset, make_trace_env


def evaluate_methods(
    dataset: TraceDataset,
    methods: dict[str, Any],
    *,
    split: str,
    horizon: int,
    budget: float,
    seed: int,
    episodes_per_domain: int,
    gamma: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(dataset.domain_names):
        for episode in range(episodes_per_domain):
            window_seed = seed + domain_index * 1_000_003 + episode * 100_003
            for method, agent in methods.items():
                env, start = make_trace_env(
                    dataset,
                    domain,
                    split,
                    horizon=horizon,
                    budget=budget,
                    window_seed=window_seed,
                )
                observation, _ = env.reset(seed=window_seed)
                rewards: list[float] = []
                served: list[float] = []
                arrivals: list[float] = []
                costs: list[float] = []
                queues: list[float] = []
                slo: list[float] = []
                latencies: list[float] = []
                step = 0
                while True:
                    started = time.perf_counter_ns()
                    if callable(agent) and not hasattr(agent, "act"):
                        action, q_values = agent(env, observation)
                    else:
                        with torch.no_grad():
                            output = agent.act(
                                torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1),
                                deterministic=True,
                            )
                        action = int(output.action.item())
                        q_values = np.full(4, np.nan)
                    decision_ms = (time.perf_counter_ns() - started) / 1.0e6
                    remaining = max(budget - env.cumulative_cost, 0.0)
                    if env.action_costs[action] > remaining + 1.0e-8:
                        raise AssertionError(f"{method} selected an unaffordable action")
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    rewards.append(float(reward))
                    served.append(float(info["served"]))
                    arrivals.append(float(info["arrivals"]))
                    costs.append(float(info["resource_cost"]))
                    queues.append(float(info["queue_length"]))
                    slo.append(float(info["slo_violation"]))
                    latencies.append(decision_ms)
                    step_rows.append(
                        {
                            "dataset": dataset.name,
                            "domain": domain,
                            "split": split,
                            "method": method,
                            "budget": budget,
                            "seed": seed,
                            "episode": episode,
                            "window_seed": window_seed,
                            "window_start": start,
                            "step": step,
                            "action": action,
                            "reward": reward,
                            "served": info["served"],
                            "arrivals": info["arrivals"],
                            "cost": info["resource_cost"],
                            "remaining_budget": info["remaining_budget"],
                            "queue": info["queue_length"],
                            "slo_violation": info["slo_violation"],
                            "decision_ms": decision_ms,
                            **{f"q_{index}": float(value) for index, value in enumerate(q_values)},
                        }
                    )
                    observation = next_observation
                    step += 1
                    if terminated or truncated:
                        break
                discounted = float(
                    np.sum(np.power(gamma, np.arange(len(rewards))) * np.asarray(rewards))
                )
                total_arrivals = float(np.sum(arrivals))
                total_served = float(np.sum(served))
                total_cost = float(np.sum(costs))
                episode_rows.append(
                    {
                        "dataset": dataset.name,
                        "domain": domain,
                        "split": split,
                        "method": method,
                        "budget": budget,
                        "seed": seed,
                        "episode": episode,
                        "window_seed": window_seed,
                        "window_start": start,
                        "return": float(np.sum(rewards)),
                        "discounted_return": discounted,
                        "completion_ratio": total_served / max(total_arrivals, 1.0),
                        "slo_violation_rate": float(np.mean(slo)),
                        "total_cost": total_cost,
                        "budget_overspend": max(total_cost - budget, 0.0),
                        "queue_area": float(np.sum(queues)),
                        "final_queue": queues[-1],
                        "decision_ms_mean": float(np.mean(latencies)),
                        "decision_ms_p95": float(np.quantile(latencies, 0.95)),
                    }
                )
    return episode_rows, step_rows
