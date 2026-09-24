from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch

from stage2_dynamic_budget.agents.rules import AggressiveRule, ConservativeRule
from stage2_dynamic_budget.direct_action_planning_dataset_validation.data import (
    TraceDataset,
)
from stage2_dynamic_budget.direct_action_planning_dataset_validation.training import (
    BranchDataset,
)

from .environment import make_calibrated_trace_env


def _feasible_actions(env) -> np.ndarray:
    remaining = max(env.config.budget - env.cumulative_cost, 0.0)
    return np.flatnonzero(env.action_costs <= remaining + 1.0e-8)


def _collection_action(
    env,
    observation: np.ndarray,
    episode: int,
    rng: np.random.Generator,
) -> int:
    feasible = _feasible_actions(env)
    mode = episode % 4
    if mode == 0:
        return int(rng.choice(feasible))
    if mode == 1:
        proposed = AggressiveRule().act(observation)
    elif mode == 2:
        proposed = ConservativeRule().act(observation)
    else:
        proposed = 0
    return int(proposed if proposed in feasible else feasible[-1])


def collect_calibrated_branch_dataset(
    dataset: TraceDataset,
    *,
    split: str,
    horizon: int,
    budget: float,
    episodes_per_domain: int,
    seed: int,
    quantile: float,
) -> BranchDataset:
    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    feasibilities: list[np.ndarray] = []
    dones: list[bool] = []
    next_loads: list[float] = []
    domains: list[str] = []
    starts: list[int] = []
    rng = np.random.default_rng(seed)
    for domain_index, domain in enumerate(dataset.domain_names):
        for episode in range(episodes_per_domain):
            window_seed = seed + domain_index * 1_000_003 + episode * 9_973
            env, start, _ = make_calibrated_trace_env(
                dataset,
                domain,
                split,
                horizon=horizon,
                budget=budget,
                window_seed=window_seed,
                quantile=quantile,
            )
            observation, _ = env.reset(seed=window_seed)
            while True:
                feasible = np.zeros(4, dtype=bool)
                branch_next = np.zeros((4, 14), dtype=np.float32)
                branch_reward = np.full(4, -np.inf, dtype=np.float32)
                branch_done = False
                for action in _feasible_actions(env):
                    branch = copy.deepcopy(env)
                    next_observation, reward, terminated, truncated, _ = branch.step(
                        int(action)
                    )
                    feasible[action] = True
                    branch_next[action] = next_observation
                    branch_reward[action] = reward
                    branch_done = bool(terminated or truncated)
                observations.append(observation.copy())
                next_observations.append(branch_next)
                rewards.append(branch_reward)
                feasibilities.append(feasible)
                dones.append(branch_done)
                next_loads.append(float(branch_next[0, 0]) if not branch_done else 0.0)
                domains.append(domain)
                starts.append(start)
                action = _collection_action(env, observation, episode, rng)
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    return BranchDataset(
        observations=np.asarray(observations, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        feasible=np.asarray(feasibilities, dtype=bool),
        done=np.asarray(dones, dtype=bool),
        next_load=np.asarray(next_loads, dtype=np.float32),
        domain=np.asarray(domains, dtype=str),
        window_start=np.asarray(starts, dtype=np.int64),
    )


def evaluate_calibrated_methods(
    dataset: TraceDataset,
    methods: dict[str, Any],
    *,
    split: str,
    horizon: int,
    budget: float,
    seed: int,
    episodes_per_domain: int,
    gamma: float,
    quantile: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(dataset.domain_names):
        for episode in range(episodes_per_domain):
            window_seed = seed + domain_index * 1_000_003 + episode * 100_003
            for method, agent in methods.items():
                env, start, calibration = make_calibrated_trace_env(
                    dataset,
                    domain,
                    split,
                    horizon=horizon,
                    budget=budget,
                    window_seed=window_seed,
                    quantile=quantile,
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
                                torch.as_tensor(
                                    observation, dtype=torch.float32
                                ).reshape(1, -1),
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
                            "capacity_multiplier": calibration.capacity_multiplier,
                            **{
                                f"q_{index}": float(value)
                                for index, value in enumerate(q_values)
                            },
                        }
                    )
                    observation = next_observation
                    step += 1
                    if terminated or truncated:
                        break
                discounted = float(
                    np.sum(np.power(gamma, np.arange(len(rewards))) * rewards)
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
                        "capacity_multiplier": calibration.capacity_multiplier,
                    }
                )
    return episode_rows, step_rows
