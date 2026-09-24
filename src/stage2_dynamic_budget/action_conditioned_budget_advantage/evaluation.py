from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .branching import BranchableDiscreteEnv
from .dp import ActionConditionedBudgetMDP, ActionDPResult


class ExactDPAgent(nn.Module):
    def __init__(self, mdp: ActionConditionedBudgetMDP, optimum: ActionDPResult):
        super().__init__()
        self.mdp = mdp
        self.optimum = optimum

    def act(self, observation: torch.Tensor, deterministic: bool = True):
        horizon_ratio = observation[..., -1]
        budget_ratio = observation[..., -2]
        t = torch.round((1.0 - horizon_ratio) * self.mdp.config.horizon).to(torch.long)
        budget = torch.round(budget_ratio * self.mdp.config.max_budget).to(torch.long)
        queue = torch.round(observation[..., 2]).to(torch.long)
        arrivals = observation[..., 0]
        load_values = torch.as_tensor(
            self.mdp.load_arrivals, dtype=arrivals.dtype, device=arrivals.device
        )
        load = torch.argmin((arrivals.unsqueeze(-1) - load_values).abs(), dim=-1)
        selected = [
            self.optimum.actions[int(ti), int(li), int(qi), int(bi)]
            for ti, li, qi, bi in zip(t, load, queue, budget)
        ]
        action = torch.as_tensor(selected, dtype=torch.long, device=observation.device)
        return type("ExactOutput", (), {"action": action})()

    def reset_budget_controller(self) -> None:
        return None


def _balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    recalls: list[float] = []
    for label in (False, True):
        mask = target == label
        if mask.any():
            recalls.append(float(np.mean(prediction[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def high_risk_threshold(mdp: ActionConditionedBudgetMDP) -> float:
    risks = [load + queue for load in range(mdp.n_loads) for queue in range(mdp.config.max_queue + 1)]
    return float(np.quantile(risks, 0.75))


def evaluate_full_state_policy(
    agent,
    method: str,
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    scenario: str,
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, object]]:
    env = BranchableDiscreteEnv(
        mdp.config,
        initial_budget=mdp.config.max_budget,
        budget_scale=mdp.config.max_budget,
    )
    env.reset(seed=seed)
    threshold = high_risk_threshold(mdp)
    rows: list[dict[str, object]] = []
    for t, load, queue, budget in np.ndindex(optimum.actions.shape):
        observation = env.set_markov_state(t, load, queue, budget)
        with torch.no_grad():
            output = agent.act(
                torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0),
                deterministic=True,
            )
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
                "remaining_budget": budget,
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
    high = frame[frame.high_risk]
    summary = {
        "method": method,
        "scenario": scenario,
        "seed": seed,
        "states": len(frame),
        "action_consistency_rate": float(frame.action_consistent.mean()),
        "mean_Q_star_regret": float(frame.Q_star_regret.mean()),
        "high_risk_states": len(high),
        "high_risk_optimal_low_cost_rate": float(high.target_low_cost.mean()),
        "high_risk_low_cost_balanced_accuracy": _balanced_accuracy(
            high.target_low_cost.to_numpy(), high.predicted_low_cost.to_numpy()
        ),
    }
    return frame, summary


def evaluate_policy_rollouts(
    agent,
    method: str,
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    budgets: list[int],
    scenario: str,
    seed: int,
    episodes: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    episode_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []
    latencies: list[float] = []
    threshold = high_risk_threshold(mdp)
    for budget in budgets:
        for episode in range(episodes):
            eval_seed = seed + 700_001 + episode * 100_003
            random_tape = np.random.default_rng(eval_seed).random(mdp.config.horizon)
            env = BranchableDiscreteEnv(mdp.config, budget, budget_scale=mdp.config.max_budget)
            optimal_env = BranchableDiscreteEnv(
                mdp.config, budget, budget_scale=mdp.config.max_budget
            )
            observation, _ = env.reset(seed=eval_seed)
            optimal_env.reset(seed=eval_seed)
            if hasattr(agent, "reset_budget_controller"):
                agent.reset_budget_controller()
            rewards: list[float] = []
            optimal_rewards: list[float] = []
            q_regrets: list[float] = []
            budget_gaps: list[float] = []
            served: list[float] = []
            arrivals: list[float] = []
            violations: list[float] = []
            for t, uniform in enumerate(random_tape):
                state = (env.t, env.load, env.queue, env.remaining_budget)
                optimal_action_same_state = int(optimum.actions[state])
                paired_optimal_action = int(
                    optimum.actions[
                        optimal_env.t,
                        optimal_env.load,
                        optimal_env.queue,
                        optimal_env.remaining_budget,
                    ]
                )
                started = time.perf_counter_ns()
                with torch.no_grad():
                    output = agent.act(
                        torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0),
                        deterministic=True,
                    )
                latencies.append((time.perf_counter_ns() - started) / 1e6)
                action = int(output.action.item())
                next_observation, reward, terminated, _, info = env.step_with_uniform(
                    action, float(uniform)
                )
                _, optimal_reward, optimal_done, _, optimal_info = optimal_env.step_with_uniform(
                    paired_optimal_action, float(uniform)
                )
                regret = float(optimum.values[state] - optimum.q_values[state + (action,)])
                risk = float(state[1] + state[2])
                rewards.append(float(reward))
                optimal_rewards.append(float(optimal_reward))
                q_regrets.append(regret)
                budget_gaps.append(abs(env.cumulative_cost - optimal_env.cumulative_cost))
                served.append(float(info["served"]))
                arrivals.append(float(info["arrivals"]))
                violations.append(float(info["slo_violation"]))
                step_rows.append(
                    {
                        "method": method,
                        "scenario": scenario,
                        "seed": seed,
                        "budget": budget,
                        "episode": episode,
                        "eval_seed": eval_seed,
                        "t": t,
                        "load": state[1],
                        "queue": state[2],
                        "budget_before": state[3],
                        "action": action,
                        "optimal_action_same_state": optimal_action_same_state,
                        "action_consistent": float(action == optimal_action_same_state),
                        "Q_star_regret": regret,
                        "reward": float(reward),
                        "resource_cost": float(info["resource_cost"]),
                        "remaining_budget": float(info["remaining_budget"]),
                        "optimal_remaining_budget": float(optimal_info["remaining_budget"]),
                        "budget_trajectory_gap": budget_gaps[-1],
                        "served": float(info["served"]),
                        "arrivals": float(info["arrivals"]),
                        "slo_violation": float(info["slo_violation"]),
                        "risk_score": risk,
                        "high_risk": bool(risk >= threshold),
                        "target_low_cost": bool(optimal_action_same_state == 0),
                        "predicted_low_cost": bool(action == 0),
                    }
                )
                observation = next_observation
                if terminated:
                    if not optimal_done:
                        raise RuntimeError("paired optimal rollout ended at a different time")
                    break
            discounts = np.power(mdp.config.gamma, np.arange(len(rewards)))
            discounted_return = float(np.dot(discounts, rewards))
            paired_optimal_return = float(np.dot(discounts, optimal_rewards))
            high_steps = [row for row in step_rows[-len(rewards) :] if row["high_risk"]]
            high_accuracy = _balanced_accuracy(
                np.asarray([row["target_low_cost"] for row in high_steps]),
                np.asarray([row["predicted_low_cost"] for row in high_steps]),
            )
            episode_rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "seed": seed,
                    "budget": budget,
                    "episode": episode,
                    "eval_seed": eval_seed,
                    "discounted_return": discounted_return,
                    "paired_optimal_discounted_return": paired_optimal_return,
                    "return_gap_to_paired_optimal": paired_optimal_return - discounted_return,
                    "value_gap_to_exact_expectation": float(
                        optimum.values[0, 1, 0, budget] - discounted_return
                    ),
                    "action_consistency_rate": float(
                        np.mean(
                            [row["action_consistent"] for row in step_rows[-len(rewards) :]]
                        )
                    ),
                    "mean_Q_star_regret": float(np.mean(q_regrets)),
                    "budget_trajectory_mae": float(np.mean(budget_gaps)),
                    "total_cost": float(env.cumulative_cost),
                    "optimal_total_cost": float(optimal_env.cumulative_cost),
                    "slo_violation_rate": float(np.mean(violations)),
                    "completion_rate": float(sum(served) / max(sum(arrivals), 1.0)),
                    "high_risk_low_cost_balanced_accuracy": high_accuracy,
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


def confusion_table(state_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        state_rows.groupby(["method", "optimal_action", "action"], as_index=False)
        .size()
        .rename(columns={"size": "count", "action": "predicted_action"})
    )
