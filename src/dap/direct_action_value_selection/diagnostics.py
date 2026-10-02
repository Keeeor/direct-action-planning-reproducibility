from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from dap.action_conditioned_budget_advantage.dp import (
    ActionConditionedBudgetMDP,
    ActionDPResult,
)

from .model import DAVSEnsemble, DAVSModel


def _error_auroc(errors: np.ndarray, uncertainty: np.ndarray) -> float:
    errors = np.asarray(errors, dtype=bool)
    uncertainty = np.asarray(uncertainty, dtype=float)
    positives = int(errors.sum())
    negatives = int((~errors).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(uncertainty, method="average")
    statistic = float(ranks[errors].sum() - positives * (positives + 1) / 2.0)
    return statistic / (positives * negatives)


def evaluate_davs_values(
    scorer: DAVSModel | DAVSEnsemble,
    method: str,
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    branch_data: pd.DataFrame,
    scenario: str,
    seed: int,
    test_times: list[int],
    small_gap: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    empirical = branch_data[branch_data.split == "test"].groupby(
        ["t", "load", "queue", "remaining_budget", "remaining_horizon", "action"],
        as_index=False,
    ).agg(Q_branch=("Q_branch", "mean"), Q_star=("Q_star", "first"))
    branch_lookup = empirical.set_index(
        ["t", "load", "queue", "remaining_budget", "action"]
    )
    state_rows: list[dict[str, object]] = []
    q_branch_errors: list[float] = []
    q_star_errors: list[float] = []
    pair_correct: list[float] = []
    variance_values: list[float] = []
    for t in test_times:
        for load, queue, budget in np.ndindex(
            mdp.n_loads, mdp.config.max_queue + 1, mdp.config.max_budget + 1
        ):
            remaining_horizon = mdp.config.horizon - t
            if isinstance(scorer, DAVSEnsemble):
                prediction = scorer.predict(load, queue, budget, remaining_horizon)
                scores = prediction.mean
                uncertainty = prediction.ranking_uncertainty
                variance_values.extend(prediction.variance[np.isfinite(scores)].tolist())
            else:
                scores = scorer.predict_scores(load, queue, budget, remaining_horizon)
                uncertainty = 0.0
            feasible = np.flatnonzero(mdp.action_costs <= budget)
            true = optimum.q_values[t, load, queue, budget, feasible]
            predicted = scores[feasible]
            branch = np.asarray(
                [
                    branch_lookup.loc[(t, load, queue, budget, int(action)), "Q_branch"]
                    for action in feasible
                ],
                dtype=float,
            )
            q_branch_errors.extend(np.abs(predicted - branch).tolist())
            q_star_errors.extend(np.abs(predicted - true).tolist())
            for left_index, left in enumerate(range(len(feasible))):
                for right in range(left + 1, len(feasible)):
                    true_gap = float(true[left] - true[right])
                    if abs(true_gap) <= 1e-12:
                        continue
                    predicted_gap = float(predicted[left] - predicted[right])
                    pair_correct.append(float(np.sign(predicted_gap) == np.sign(true_gap)))
            action = int(feasible[int(np.argmax(predicted))])
            optimal_action = int(optimum.actions[t, load, queue, budget])
            ordered_true = np.sort(true)
            top_gap = (
                float(ordered_true[-1] - ordered_true[-2])
                if len(ordered_true) >= 2
                else float("inf")
            )
            state_rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "seed": seed,
                    "t": t,
                    "load": load,
                    "queue": queue,
                    "budget": budget,
                    "action": action,
                    "optimal_action": optimal_action,
                    "error": float(action != optimal_action),
                    "Q_star_regret": float(
                        optimum.values[t, load, queue, budget]
                        - optimum.q_values[t, load, queue, budget, action]
                    ),
                    "exact_top1_top2_gap": top_gap,
                    "small_gap": bool(top_gap <= small_gap),
                    "ranking_uncertainty": float(uncertainty),
                }
            )
    states = pd.DataFrame(state_rows)
    errors = states.error.to_numpy(float)
    uncertainty_values = states.ranking_uncertainty.to_numpy(float)
    if np.std(uncertainty_values) > 1e-12 and np.std(errors) > 1e-12:
        correlation = float(spearmanr(uncertainty_values, errors).statistic)
    else:
        correlation = float("nan")
    small = states[states.small_gap]
    return (
        {
            "method": method,
            "scenario": scenario,
            "seed": seed,
            "test_states": int(len(states)),
            "action_value_mae_q_branch": float(np.mean(q_branch_errors)),
            "action_value_mae_q_star": float(np.mean(q_star_errors)),
            "pairwise_action_ranking_accuracy": float(np.mean(pair_correct)),
            "small_gap_threshold": float(small_gap),
            "small_gap_states": int(len(small)),
            "small_gap_error_rate": float(small.error.mean()) if len(small) else np.nan,
            "all_state_error_rate": float(states.error.mean()),
            "ensemble_score_variance_mean": (
                float(np.mean(variance_values)) if variance_values else 0.0
            ),
            "error_auroc": _error_auroc(errors, uncertainty_values),
            "error_spearman": correlation,
        },
        states,
    )
