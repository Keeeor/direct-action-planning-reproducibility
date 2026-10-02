from __future__ import annotations

import numpy as np

from dap.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)

from .learning import EmpiricalActionModel
from .planning import BudgetValueTable, one_step_plan


def exact_values_by_horizon(optimum: ActionDPResult) -> np.ndarray:
    return optimum.values[::-1].copy()


def value_diagnostics(
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    value: BudgetValueTable,
) -> dict[str, float | int]:
    exact = exact_values_by_horizon(optimum)
    error = value.values - exact
    budget_differences = np.diff(value.values, axis=-1)
    violations = budget_differences < -1e-10
    violation_magnitudes = -budget_differences[violations]
    correct = 0
    states = 0
    rank_correlations: list[float] = []
    for t, load, queue, budget in np.ndindex(optimum.actions.shape):
        plan = one_step_plan(mdp, value, t, load, queue, budget)
        correct += int(plan.action == int(optimum.actions[t, load, queue, budget]))
        states += 1
        feasible = np.isfinite(plan.q_values) & np.isfinite(
            optimum.q_values[t, load, queue, budget]
        )
        if int(feasible.sum()) >= 2:
            left = plan.q_values[feasible]
            right = optimum.q_values[t, load, queue, budget][feasible]
            left_rank = np.argsort(np.argsort(left))
            right_rank = np.argsort(np.argsort(right))
            rank_correlations.append(float(np.corrcoef(left_rank, right_rank)[0, 1]))
    return {
        "value_cells": int(error.size),
        "value_mae": float(np.mean(np.abs(error))),
        "value_rmse": float(np.sqrt(np.mean(error**2))),
        "value_max_absolute_error": float(np.max(np.abs(error))),
        "budget_monotonicity_comparisons": int(budget_differences.size),
        "budget_monotonicity_violation_rate": float(np.mean(violations)),
        "budget_monotonicity_mean_violation": (
            float(np.mean(violation_magnitudes)) if len(violation_magnitudes) else 0.0
        ),
        "true_transition_action_ranking_accuracy": correct / max(states, 1),
        "true_transition_mean_action_rank_correlation": float(np.mean(rank_correlations)),
    }


def transfer_value_error(
    optimum: ActionDPResult,
    value: BudgetValueTable,
) -> dict[str, float | int]:
    error = value.values - exact_values_by_horizon(optimum)
    return {
        "value_cells": int(error.size),
        "value_mae": float(np.mean(np.abs(error))),
        "value_rmse": float(np.sqrt(np.mean(error**2))),
        "value_max_absolute_error": float(np.max(np.abs(error))),
    }


def transition_model_diagnostics(
    mdp: ActionConditionedBudgetMDP,
    model: EmpiricalActionModel,
) -> dict[str, float | int]:
    probability_mae: list[float] = []
    total_variation: list[float] = []
    queue_mae: list[float] = []
    queue_top_correct: list[float] = []
    reward_error: list[float] = []
    cost_error: list[float] = []
    for t, load, queue, action in np.ndindex(
        mdp.config.horizon,
        mdp.n_loads,
        mdp.config.max_queue + 1,
        mdp.n_actions,
    ):
        prediction = model.predict(t, load, queue, action)
        next_queue, reward, metrics = mdp.outcome(queue, load, action)
        true_joint = np.zeros_like(prediction.next_state_probabilities)
        true_joint[:, next_queue] = mdp.load_probabilities(t, load)
        delta = prediction.next_state_probabilities - true_joint
        probability_mae.append(float(np.mean(np.abs(delta))))
        total_variation.append(float(0.5 * np.sum(np.abs(delta))))
        queue_distribution = prediction.next_state_probabilities.sum(axis=0)
        queue_expectation = float(
            np.dot(queue_distribution, np.arange(mdp.config.max_queue + 1))
        )
        queue_mae.append(abs(queue_expectation - next_queue))
        queue_top_correct.append(float(int(np.argmax(queue_distribution)) == next_queue))
        reward_error.append(abs(prediction.reward - reward))
        cost_error.append(abs(prediction.cost - metrics["cost"]))
    return {
        "state_action_cells": len(probability_mae),
        "next_state_probability_mae": float(np.mean(probability_mae)),
        "next_state_total_variation_mean": float(np.mean(total_variation)),
        "next_queue_expected_mae": float(np.mean(queue_mae)),
        "next_queue_top1_accuracy": float(np.mean(queue_top_correct)),
        "reward_mae": float(np.mean(reward_error)),
        "cost_mae": float(np.mean(cost_error)),
    }
