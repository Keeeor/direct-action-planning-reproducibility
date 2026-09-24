from __future__ import annotations

import time

import numpy as np
import pandas as pd

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)
from stage2_dynamic_budget.action_conditioned_budget_advantage.evaluation import (
    high_risk_threshold,
)
from stage2_dynamic_budget.direct_action_planning_repair.model import StructuredActionEffectModel

from .history import CausalHistory, deserialize_history
from .model import ContextValuePredictor
from .planning import context_one_step_plan


def _balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    recalls = []
    for label in (False, True):
        mask = target == label
        if mask.any():
            recalls.append(float(np.mean(prediction[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def _history_bank(frame: pd.DataFrame) -> dict[tuple[int, int], list[CausalHistory]]:
    bank: dict[tuple[int, int], list[CausalHistory]] = {}
    for row in frame.itertuples(index=False):
        key = (int(row.t), int(row.load))
        bank.setdefault(key, []).append(deserialize_history(row))
    return bank


def _state_history(
    bank: dict[tuple[int, int], list[CausalHistory]],
    mdp: ActionConditionedBudgetMDP,
    t: int,
    load: int,
    queue: int,
    budget: int,
) -> CausalHistory:
    candidates = bank.get((t, load), [])
    if candidates:
        selected = candidates[(queue * (mdp.config.max_budget + 1) + budget) % len(candidates)]
        return CausalHistory(
            arrivals=(*selected.arrivals[:-1], float(mdp.load_arrivals[load])),
            queues=(*selected.queues[:-1], float(queue)),
            actions=selected.actions,
            capacities=selected.capacities,
        )
    return CausalHistory(
        arrivals=tuple(float(mdp.load_arrivals[load]) for _ in range(t + 1)),
        queues=tuple(float(queue) for _ in range(t + 1)),
        actions=tuple(0.0 for _ in range(t)),
        capacities=tuple(float(mdp.action_capacity[0]) for _ in range(t)),
    )


def evaluate_context_state_grid(
    predictor: ContextValuePredictor,
    transition: StructuredActionEffectModel,
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    context_frame: pd.DataFrame,
    aliases: pd.DataFrame,
    method: str,
    model_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bank = _history_bank(context_frame)
    alias_map = aliases.set_index(["t", "load", "queue", "remaining_budget"])[
        "material_action_conflict"
    ]
    threshold = high_risk_threshold(mdp)
    rows: list[dict[str, object]] = []
    for t, load, queue, budget in np.ndindex(optimum.actions.shape):
        history = _state_history(bank, mdp, t, load, queue, budget)
        started = time.perf_counter_ns()
        plan = context_one_step_plan(
            mdp, predictor, transition, history, t, load, queue, budget
        )
        latency = (time.perf_counter_ns() - started) / 1.0e6
        state = (t, load, queue, budget)
        optimal_action = int(optimum.actions[state])
        selected_q = float(optimum.q_values[state + (plan.action,)])
        optimal_q = float(optimum.values[state])
        row: dict[str, object] = {
            "method": method,
            "scenario": mdp.config.scenario,
            "model_seed": model_seed,
            "t": t,
            "load": load,
            "queue": queue,
            "remaining_budget": budget,
            "remaining_horizon": mdp.config.horizon - t,
            "action": plan.action,
            "optimal_action": optimal_action,
            "action_consistent": float(plan.action == optimal_action),
            "Q_star_regret": optimal_q - selected_q,
            "risk_score": float(load + queue),
            "high_risk": bool(load + queue >= threshold),
            "target_low_cost": bool(optimal_action == 0),
            "predicted_low_cost": bool(plan.action == 0),
            "material_action_conflict": bool(alias_map.loc[state]),
            "planning_ms": latency,
        }
        for action in range(mdp.n_actions):
            row[f"q_plan_a{action}"] = plan.q_values[action]
            row[f"q_star_a{action}"] = optimum.q_values[state + (action,)]
        rows.append(row)
    states = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for alias_region in ("all", False, True):
        subset = (
            states
            if alias_region == "all"
            else states[states.material_action_conflict == alias_region]
        )
        high = subset[subset.high_risk]
        summaries.append(
            {
                "method": method,
                "scenario": mdp.config.scenario,
                "model_seed": model_seed,
                "alias_region": alias_region,
                "states": len(subset),
                "action_consistency": float(subset.action_consistent.mean()),
                "mean_Q_star_regret": float(subset.Q_star_regret.mean()),
                "high_risk_low_cost_balanced_accuracy": _balanced_accuracy(
                    high.target_low_cost.to_numpy(), high.predicted_low_cost.to_numpy()
                ),
                "planning_ms_per_decision": float(subset.planning_ms.mean()),
            }
        )
        for left in range(mdp.n_actions):
            for right in range(left + 1, mdp.n_actions):
                true_left = subset[f"q_star_a{left}"].to_numpy(float)
                true_right = subset[f"q_star_a{right}"].to_numpy(float)
                plan_left = subset[f"q_plan_a{left}"].to_numpy(float)
                plan_right = subset[f"q_plan_a{right}"].to_numpy(float)
                valid = (
                    np.isfinite(true_left)
                    & np.isfinite(true_right)
                    & np.isfinite(plan_left)
                    & np.isfinite(plan_right)
                )
                true_delta = true_left[valid] - true_right[valid]
                plan_delta = plan_left[valid] - plan_right[valid]
                informative = np.abs(true_delta) > 1.0e-10
                if informative.any():
                    accuracy = float(
                        np.mean(true_delta[informative] * plan_delta[informative] > 0)
                    )
                    count = int(informative.sum())
                else:
                    accuracy = float("nan")
                    count = 0
                pair_rows.append(
                    {
                        "method": method,
                        "scenario": mdp.config.scenario,
                        "model_seed": model_seed,
                        "alias_region": alias_region,
                        "action_i": left,
                        "action_j": right,
                        "pairs": count,
                        "pair_ranking_accuracy": accuracy,
                    }
                )
    return states, pd.DataFrame(summaries), pd.DataFrame(pair_rows)
