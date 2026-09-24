from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)
from stage2_dynamic_budget.direct_action_planning.learning import EmpiricalActionModel
from stage2_dynamic_budget.direct_action_planning.planning import BudgetValueTable, one_step_plan


@dataclass(frozen=True)
class BranchDataset:
    samples: pd.DataFrame
    labels: pd.DataFrame


def _sample_next_loads(probabilities: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    return np.minimum(
        np.searchsorted(np.cumsum(probabilities), uniforms, side="right"),
        len(probabilities) - 1,
    ).astype(np.int64)


def collect_common_random_branch_data(
    mdp: ActionConditionedBudgetMDP,
    value: BudgetValueTable,
    optimum: ActionDPResult,
    scenario: str,
    seed: int,
    train_samples: int,
    validation_samples: int,
) -> BranchDataset:
    """Label every action with shared exogenous draws and exact known state components."""

    if train_samples <= 0 or validation_samples <= 0:
        raise ValueError("both train and validation sample counts must be positive")
    rng = np.random.default_rng(seed)
    sample_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    cfg = mdp.config
    a0 = int(np.argmin(mdp.action_costs))
    for t, load, queue in np.ndindex(cfg.horizon, mdp.n_loads, cfg.max_queue + 1):
        probabilities = mdp.load_probabilities(t, load)
        count = train_samples + validation_samples
        uniforms = rng.random(count)
        next_loads = _sample_next_loads(probabilities, uniforms)
        base_queue, _, _ = mdp.outcome(queue, load, a0)
        tape_hash = hashlib.sha256(uniforms.tobytes()).hexdigest()
        for sample_index, (uniform, next_load) in enumerate(zip(uniforms, next_loads)):
            split = "train" if sample_index < train_samples else "validation"
            group = f"{scenario}|s{seed}|t{t}|l{load}|q{queue}|n{sample_index}"
            for action in range(mdp.n_actions):
                next_queue, reward, metrics = mdp.outcome(queue, load, action)
                sample_rows.append(
                    {
                        "branch_group_id": group,
                        "scenario": scenario,
                        "seed": seed,
                        "split": split,
                        "sample_index": sample_index,
                        "t": t,
                        "load": load,
                        "queue": queue,
                        "action": action,
                        "transition_uniform": float(uniform),
                        "true_next_load": int(next_load),
                        "true_next_queue": next_queue,
                        "base_next_load": int(next_load),
                        "base_next_queue": base_queue,
                        "action_effect_load": 0,
                        "action_effect_queue": next_queue - base_queue,
                        "reward": float(reward),
                        "cost": float(metrics["cost"]),
                        "random_tape_sha256": tape_hash,
                    }
                )
        for split, split_loads in (
            ("train", next_loads[:train_samples]),
            ("validation", next_loads[train_samples:]),
        ):
            empirical = np.bincount(split_loads, minlength=mdp.n_loads).astype(float)
            empirical /= empirical.sum()
            for budget in range(cfg.max_budget + 1):
                lv_plan = one_step_plan(mdp, value, t, load, queue, budget)
                for action in range(mdp.n_actions):
                    cost = int(mdp.action_costs[action])
                    if cost > budget:
                        continue
                    next_queue, reward, metrics = mdp.outcome(queue, load, action)
                    continuation = [
                        value.predict(
                            next_load,
                            next_queue,
                            budget - cost,
                            cfg.horizon - t - 1,
                        )
                        for next_load in range(mdp.n_loads)
                    ]
                    label_rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "split": split,
                            "t": t,
                            "load": load,
                            "queue": queue,
                            "remaining_budget": budget,
                            "remaining_horizon": cfg.horizon - t,
                            "action": action,
                            "action_cost": cost,
                            "true_next_queue": next_queue,
                            "base_next_queue": base_queue,
                            "action_effect_queue": next_queue - base_queue,
                            "reward": float(reward),
                            "cost": float(metrics["cost"]),
                            "q_lv": float(lv_plan.q_values[action]),
                            "q_star": float(optimum.q_values[t, load, queue, budget, action]),
                            "optimal_action": int(optimum.actions[t, load, queue, budget]),
                            "next_load_prob_0": float(empirical[0]),
                            "next_load_prob_1": float(empirical[1]),
                            "next_load_prob_2": float(empirical[2]),
                            "continuation_value_0": float(continuation[0]),
                            "continuation_value_1": float(continuation[1]),
                            "continuation_value_2": float(continuation[2]),
                            "random_tape_sha256": tape_hash,
                            "sample_count": len(split_loads),
                            "source": "D0_common_random_full_grid",
                        }
                    )
    return BranchDataset(pd.DataFrame(sample_rows), pd.DataFrame(label_rows))


def attach_priority_weights(
    labels: pd.DataFrame,
    mdp: ActionConditionedBudgetMDP,
    old_model: EmpiricalActionModel,
    value: BudgetValueTable,
    optimum: ActionDPResult,
    scenario: str,
    close_gap_ceiling: float,
    weight_ceiling: float,
) -> pd.DataFrame:
    """Attach preregistered state weights without changing any value target."""

    state_weights: dict[tuple[int, int, int, int], dict[str, object]] = {}
    risks = [
        load + queue
        for load in range(mdp.n_loads)
        for queue in range(mdp.config.max_queue + 1)
    ]
    risk_cutoff = float(np.quantile(risks, 0.75))
    positive_regrets: list[float] = []
    interim: list[tuple[tuple[int, int, int, int], dict[str, object]]] = []
    for t, load, queue, budget in np.ndindex(optimum.actions.shape):
        lv = one_step_plan(mdp, value, t, load, queue, budget)
        old = one_step_plan(mdp, value, t, load, queue, budget, learned_model=old_model)
        feasible = np.flatnonzero(np.isfinite(lv.q_values))
        order = feasible[np.argsort(-lv.q_values[feasible], kind="stable")]
        gap = (
            float(lv.q_values[order[0]] - lv.q_values[order[1]])
            if len(order) > 1
            else np.inf
        )
        lv_qstar = float(optimum.q_values[t, load, queue, budget, lv.action])
        old_qstar = float(optimum.q_values[t, load, queue, budget, old.action])
        regret = max(lv_qstar - old_qstar, 0.0)
        if old.action != lv.action and regret > 0:
            positive_regrets.append(regret)
        interim.append(
            (
                (t, load, queue, budget),
                {
                    "old_model_disagrees": bool(old.action != lv.action),
                    "old_flip_q_star_regret": regret,
                    "late_burst_priority": scenario == "late_burst",
                    "tight_budget_priority": budget <= mdp.config.max_budget // 3,
                    "high_risk_optimal_low_cost_priority": bool(
                        load + queue >= risk_cutoff
                        and int(optimum.actions[t, load, queue, budget]) == 0
                    ),
                    "close_gap_priority": gap <= close_gap_ceiling,
                },
            )
        )
    threshold = float(np.quantile(positive_regrets, 0.75)) if positive_regrets else np.inf
    for key, info in interim:
        weight = 1.0
        weight += 1.0 * float(info["old_model_disagrees"])
        weight += 1.5 * float(info["old_flip_q_star_regret"] >= threshold)
        weight += 0.5 * float(info["late_burst_priority"])
        weight += 0.5 * float(info["tight_budget_priority"])
        weight += 1.0 * float(info["high_risk_optimal_low_cost_priority"])
        weight += 0.5 * float(info["close_gap_priority"])
        info["high_regret_threshold"] = threshold
        info["priority_weight"] = min(weight, weight_ceiling)
        state_weights[key] = info
    annotations = pd.DataFrame(
        [
            {
                "t": key[0],
                "load": key[1],
                "queue": key[2],
                "remaining_budget": key[3],
                **value,
            }
            for key, value in state_weights.items()
        ]
    )
    return labels.merge(
        annotations,
        on=["t", "load", "queue", "remaining_budget"],
        how="left",
        validate="many_to_one",
    )
