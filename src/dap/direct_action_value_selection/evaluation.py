from __future__ import annotations

import time

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
from dap.action_conditioned_budget_advantage.evaluation import (
    high_risk_threshold,
)


def _balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    recalls: list[float] = []
    for label in (False, True):
        mask = target == label
        if mask.any():
            recalls.append(float(np.mean(prediction[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def evaluate_test_states(
    agent,
    method: str,
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    scenario: str,
    seed: int,
    test_times: list[int],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    env = BranchableDiscreteEnv(
        mdp.config,
        initial_budget=mdp.config.max_budget,
        budget_scale=mdp.config.max_budget,
    )
    env.reset(seed=seed)
    threshold = high_risk_threshold(mdp)
    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    for t in test_times:
        for load, queue, budget in np.ndindex(
            mdp.n_loads, mdp.config.max_queue + 1, mdp.config.max_budget + 1
        ):
            observation = env.set_markov_state(t, load, queue, budget)
            started = time.perf_counter_ns()
            with torch.no_grad():
                output = agent.act(
                    torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0),
                    deterministic=True,
                )
            latencies.append((time.perf_counter_ns() - started) / 1e6)
            action = int(output.action.item())
            optimal_action = int(optimum.actions[t, load, queue, budget])
            selected_q = float(optimum.q_values[t, load, queue, budget, action])
            optimal_q = float(optimum.values[t, load, queue, budget])
            risk = float(load + queue)
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "seed": seed,
                    "t": t,
                    "load": load,
                    "queue": queue,
                    "budget": budget,
                    "remaining_horizon": mdp.config.horizon - t,
                    "action": action,
                    "optimal_action": optimal_action,
                    "action_consistent": float(action == optimal_action),
                    "Q_star_selected": selected_q,
                    "Q_star_optimal": optimal_q,
                    "Q_star_regret": optimal_q - selected_q,
                    "risk_score": risk,
                    "high_risk": bool(risk >= threshold),
                    "target_low_cost": bool(optimal_action == 0),
                    "predicted_low_cost": bool(action == 0),
                }
            )
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for budget, group in frame.groupby("budget"):
        high = group[group.high_risk]
        summaries.append(
            {
                "method": method,
                "scenario": scenario,
                "seed": seed,
                "budget": int(budget),
                "states": int(len(group)),
                "action_consistency_rate": float(group.action_consistent.mean()),
                "mean_Q_star_regret": float(group.Q_star_regret.mean()),
                "high_risk_low_cost_balanced_accuracy": _balanced_accuracy(
                    high.target_low_cost.to_numpy(), high.predicted_low_cost.to_numpy()
                ),
            }
        )
    return (
        frame,
        pd.DataFrame(summaries),
        {
            "decision_latency_ms_mean": float(np.mean(latencies)),
            "decision_latency_ms_p95": float(np.quantile(latencies, 0.95)),
        },
    )


def _terminal_value(
    optimum: ActionDPResult, env: BranchableDiscreteEnv
) -> float:
    if env.t >= env.config.horizon:
        return 0.0
    return float(optimum.values[env.t, env.load, env.queue, env.remaining_budget])


def evaluate_test_window_rollouts(
    agent,
    method: str,
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    budgets: list[int],
    scenario: str,
    seed: int,
    episodes: int,
    test_times: list[int],
    device: torch.device,
) -> pd.DataFrame:
    ordered_times = sorted(int(value) for value in test_times)
    if not ordered_times or ordered_times != list(
        range(ordered_times[0], ordered_times[-1] + 1)
    ):
        raise ValueError("test-window rollout requires one contiguous time block")
    start_t, end_t = ordered_times[0], ordered_times[-1]
    threshold = high_risk_threshold(mdp)
    rows: list[dict[str, object]] = []
    for budget in budgets:
        for episode in range(episodes):
            eval_seed = seed + 900_001 + episode * 100_003
            tape = np.random.default_rng(eval_seed).random(mdp.config.horizon)
            candidate = BranchableDiscreteEnv(
                mdp.config, budget, budget_scale=mdp.config.max_budget
            )
            optimal = BranchableDiscreteEnv(
                mdp.config, budget, budget_scale=mdp.config.max_budget
            )
            observation, _ = candidate.reset(seed=eval_seed)
            optimal.reset(seed=eval_seed)
            for t in range(start_t):
                action = int(
                    optimum.actions[
                        candidate.t,
                        candidate.load,
                        candidate.queue,
                        candidate.remaining_budget,
                    ]
                )
                observation, _, done, _, _ = candidate.step_with_uniform(action, float(tape[t]))
                _, _, optimal_done, _, _ = optimal.step_with_uniform(action, float(tape[t]))
                if done or optimal_done:
                    raise RuntimeError("test prefix terminated before the frozen test block")
            if candidate.snapshot() != optimal.snapshot():
                raise RuntimeError("paired test-window environments diverged during prefix")
            prefix_cost = float(candidate.cumulative_cost)
            if hasattr(agent, "reset_budget_controller"):
                agent.reset_budget_controller()
            candidate_rewards: list[float] = []
            optimal_rewards: list[float] = []
            q_regrets: list[float] = []
            budget_gaps: list[float] = []
            served: list[float] = []
            arrivals: list[float] = []
            violations: list[float] = []
            high_target: list[bool] = []
            high_prediction: list[bool] = []
            consistent: list[float] = []
            for t in range(start_t, end_t + 1):
                state = (
                    candidate.t,
                    candidate.load,
                    candidate.queue,
                    candidate.remaining_budget,
                )
                optimal_same_state = int(optimum.actions[state])
                paired_optimal_action = int(
                    optimum.actions[
                        optimal.t,
                        optimal.load,
                        optimal.queue,
                        optimal.remaining_budget,
                    ]
                )
                with torch.no_grad():
                    output = agent.act(
                        torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0),
                        deterministic=True,
                    )
                action = int(output.action.item())
                observation, reward, done, _, info = candidate.step_with_uniform(
                    action, float(tape[t])
                )
                _, optimal_reward, optimal_done, _, optimal_info = optimal.step_with_uniform(
                    paired_optimal_action, float(tape[t])
                )
                candidate_rewards.append(float(reward))
                optimal_rewards.append(float(optimal_reward))
                q_regrets.append(
                    float(optimum.values[state] - optimum.q_values[state + (action,)])
                )
                budget_gaps.append(
                    abs(candidate.remaining_budget - optimal.remaining_budget)
                )
                served.append(float(info["served"]))
                arrivals.append(float(info["arrivals"]))
                violations.append(float(info["slo_violation"]))
                risk = float(state[1] + state[2])
                if risk >= threshold:
                    high_target.append(optimal_same_state == 0)
                    high_prediction.append(action == 0)
                consistent.append(float(action == optimal_same_state))
                if done != optimal_done:
                    raise RuntimeError("paired test-window environments ended inconsistently")
            discounts = np.power(mdp.config.gamma, np.arange(len(candidate_rewards)))
            candidate_return = float(np.dot(discounts, candidate_rewards))
            optimal_return = float(np.dot(discounts, optimal_rewards))
            bootstrap_discount = mdp.config.gamma ** len(candidate_rewards)
            candidate_bootstrapped = candidate_return + bootstrap_discount * _terminal_value(
                optimum, candidate
            )
            optimal_bootstrapped = optimal_return + bootstrap_discount * _terminal_value(
                optimum, optimal
            )
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "seed": seed,
                    "budget": budget,
                    "episode": episode,
                    "eval_seed": eval_seed,
                    "test_start_t": start_t,
                    "test_end_t": end_t,
                    "action_consistency_rate": float(np.mean(consistent)),
                    "mean_Q_star_regret": float(np.mean(q_regrets)),
                    "return_gap_to_paired_optimal": optimal_bootstrapped
                    - candidate_bootstrapped,
                    "candidate_bootstrapped_return": candidate_bootstrapped,
                    "paired_optimal_bootstrapped_return": optimal_bootstrapped,
                    "high_risk_low_cost_balanced_accuracy": _balanced_accuracy(
                        np.asarray(high_target), np.asarray(high_prediction)
                    ),
                    "budget_trajectory_mae": float(np.mean(budget_gaps)),
                    "completion_rate": float(sum(served) / max(sum(arrivals), 1.0)),
                    "slo_violation_rate": float(np.mean(violations)),
                    "total_cost": float(candidate.cumulative_cost - prefix_cost),
                    "optimal_total_cost": float(optimal.cumulative_cost - prefix_cost),
                }
            )
    return pd.DataFrame(rows)
