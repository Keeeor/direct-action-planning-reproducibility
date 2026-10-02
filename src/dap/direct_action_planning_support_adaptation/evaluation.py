from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from dap.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)

from .environment import make_branchable_env

__all__ = ["make_branchable_env", "evaluate_full_state", "evaluate_rollouts"]


def _balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    recalls = []
    for label in (False, True):
        mask = np.asarray(target, dtype=bool) == label
        if mask.any():
            recalls.append(float(np.mean(np.asarray(prediction, dtype=bool)[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def evaluate_full_state(
    agent,
    method: str,
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    *,
    scenario_id: str,
    region: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    env = make_branchable_env(mdp, mdp.config.max_budget)
    env.reset(seed=seed)
    risks = np.asarray(
        [load + queue for load in range(mdp.n_loads) for queue in range(mdp.config.max_queue + 1)]
    )
    threshold = float(np.quantile(risks, 0.75))
    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    for t, load, queue, budget in np.ndindex(optimum.actions.shape):
        observation = env.set_markov_state(t, load, queue, budget)
        started = time.perf_counter_ns()
        with torch.no_grad():
            output = agent.act(torch.as_tensor(observation).unsqueeze(0), deterministic=True)
        latencies.append((time.perf_counter_ns() - started) / 1.0e6)
        action = int(output.action.item())
        state = (t, load, queue, budget)
        optimal_action = int(optimum.actions[state])
        risk = float(load + queue)
        row: dict[str, object] = {
            "method": method,
            "scenario_id": scenario_id,
            "region": region,
            "seed": seed,
            "t": t,
            "load": load,
            "queue": queue,
            "remaining_budget": budget,
            "remaining_horizon": mdp.config.horizon - t,
            "action": action,
            "optimal_action": optimal_action,
            "action_consistent": float(action == optimal_action),
            "action_error": float(action != optimal_action),
            "Q_star_regret": float(optimum.values[state] - optimum.q_values[state + (action,)]),
            "high_risk": bool(risk >= threshold),
            "target_low_cost": bool(optimal_action == 0),
            "predicted_low_cost": bool(action == 0),
        }
        for candidate in range(mdp.n_actions):
            value = float(optimum.q_values[state + (candidate,)])
            row[f"q_star_a{candidate}"] = value if np.isfinite(value) else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows)
    high = frame[frame.high_risk]
    return frame, {
        "method": method,
        "scenario_id": scenario_id,
        "region": region,
        "seed": seed,
        "states": len(frame),
        "action_consistency": float(frame.action_consistent.mean()),
        "mean_Q_star_regret": float(frame.Q_star_regret.mean()),
        "high_risk_low_cost_balanced_accuracy": _balanced_accuracy(
            high.target_low_cost.to_numpy(), high.predicted_low_cost.to_numpy()
        ),
        "planning_ms_per_decision": float(np.mean(latencies)),
    }


def evaluate_rollouts(
    agent,
    method: str,
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    *,
    budgets: list[int],
    scenario_id: str,
    seed: int,
    episodes: int,
    metric_start_t: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if not 0 <= metric_start_t < mdp.config.horizon:
        raise ValueError("metric_start_t must fall inside the horizon")
    episode_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []
    latencies: list[float] = []
    for budget in budgets:
        for episode in range(episodes):
            eval_seed = seed * 1_000_003 + budget * 10_007 + episode * 997 + 71
            tape = np.random.default_rng(eval_seed).random(mdp.config.horizon)
            env = make_branchable_env(mdp, int(budget))
            optimal_env = make_branchable_env(mdp, int(budget))
            observation, _ = env.reset(seed=eval_seed)
            optimal_env.reset(seed=eval_seed)
            if hasattr(agent, "reset_budget_controller"):
                agent.reset_budget_controller()
            suffix_rewards: list[float] = []
            suffix_optimal_rewards: list[float] = []
            regrets: list[float] = []
            budget_gaps: list[float] = []
            served: list[float] = []
            arrivals: list[float] = []
            violations: list[float] = []
            suffix_consistency: list[float] = []
            suffix_low_target: list[bool] = []
            suffix_low_predicted: list[bool] = []
            for uniform in tape:
                state = (env.t, env.load, env.queue, env.remaining_budget)
                paired_state = (
                    optimal_env.t,
                    optimal_env.load,
                    optimal_env.queue,
                    optimal_env.remaining_budget,
                )
                optimal_same = int(optimum.actions[state])
                paired_action = int(optimum.actions[paired_state])
                started = time.perf_counter_ns()
                with torch.no_grad():
                    output = agent.act(
                        torch.as_tensor(observation).unsqueeze(0), deterministic=True
                    )
                latencies.append((time.perf_counter_ns() - started) / 1.0e6)
                action = int(output.action.item())
                next_observation, reward, done, _, info = env.step_with_uniform(
                    action, float(uniform)
                )
                _, optimal_reward, optimal_done, _, optimal_info = optimal_env.step_with_uniform(
                    paired_action, float(uniform)
                )
                regret = float(optimum.values[state] - optimum.q_values[state + (action,)])
                in_suffix = int(state[0]) >= metric_start_t
                step_rows.append(
                    {
                        "method": method,
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "budget": budget,
                        "episode": episode,
                        "eval_seed": eval_seed,
                        "trajectory_id": f"{scenario_id}:test_s{seed}:e{episode}:b{budget}",
                        "row_id": f"{scenario_id}:test_s{seed}:e{episode}:b{budget}:t{state[0]}",
                        "t": state[0],
                        "load": state[1],
                        "queue": state[2],
                        "remaining_budget": state[3],
                        "remaining_horizon": mdp.config.horizon - state[0],
                        "action": action,
                        "optimal_action": optimal_same,
                        "action_consistent": float(action == optimal_same),
                        "action_error": float(action != optimal_same),
                        "Q_star_regret": regret,
                        "reward": float(reward),
                        "resource_cost": float(info["resource_cost"]),
                        "remaining_budget_after": float(info["remaining_budget"]),
                        "optimal_remaining_budget": float(optimal_info["remaining_budget"]),
                        "budget_trajectory_gap": abs(env.cumulative_cost - optimal_env.cumulative_cost),
                        "served": float(info["served"]),
                        "arrivals": float(info["arrivals"]),
                        "slo_violation": float(info["slo_violation"]),
                        "target_low_cost": bool(optimal_same == 0),
                        "predicted_low_cost": bool(action == 0),
                        "in_metric_suffix": in_suffix,
                    }
                )
                if in_suffix:
                    suffix_rewards.append(float(reward))
                    suffix_optimal_rewards.append(float(optimal_reward))
                    regrets.append(regret)
                    budget_gaps.append(abs(env.cumulative_cost - optimal_env.cumulative_cost))
                    served.append(float(info["served"]))
                    arrivals.append(float(info["arrivals"]))
                    violations.append(float(info["slo_violation"]))
                    suffix_consistency.append(float(action == optimal_same))
                    suffix_low_target.append(optimal_same == 0)
                    suffix_low_predicted.append(action == 0)
                observation = next_observation
                if done:
                    if not optimal_done:
                        raise RuntimeError("paired optimal trajectory length mismatch")
                    break
            discounts = np.power(mdp.config.gamma, np.arange(len(suffix_rewards)))
            discounted = float(np.dot(discounts, suffix_rewards))
            optimal_discounted = float(np.dot(discounts, suffix_optimal_rewards))
            episode_rows.append(
                {
                    "method": method,
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "budget": budget,
                    "episode": episode,
                    "eval_seed": eval_seed,
                    "metric_start_t": metric_start_t,
                    "discounted_return": discounted,
                    "paired_optimal_discounted_return": optimal_discounted,
                    "return_gap_to_paired_optimal": optimal_discounted - discounted,
                    "action_consistency_rate": float(np.mean(suffix_consistency)),
                    "mean_Q_star_regret": float(np.mean(regrets)),
                    "budget_trajectory_mae": float(np.mean(budget_gaps)),
                    "total_cost": float(env.cumulative_cost),
                    "optimal_total_cost": float(optimal_env.cumulative_cost),
                    "slo_violation_rate": float(np.mean(violations)),
                    "completion_rate": float(sum(served) / max(sum(arrivals), 1.0)),
                    "high_risk_low_cost_balanced_accuracy": _balanced_accuracy(
                        np.asarray(suffix_low_target), np.asarray(suffix_low_predicted)
                    ),
                }
            )
    return (
        pd.DataFrame(episode_rows),
        pd.DataFrame(step_rows),
        {
            "decision_latency_ms_mean": float(np.mean(latencies)),
            "decision_latency_ms_p95": float(np.quantile(latencies, 0.95)),
        },
    )
