from __future__ import annotations

import numpy as np
import pandas as pd


def _service_guardrail(
    candidate: pd.Series,
    baseline: pd.Series,
    completion_tolerance: float,
    slo_tolerance: float,
    cost_fraction: float,
) -> dict[str, object]:
    completion_delta = float(candidate.completion_rate - baseline.completion_rate)
    slo_delta = float(candidate.slo_violation_rate - baseline.slo_violation_rate)
    cost_delta = float(candidate.total_cost - baseline.total_cost)
    material_service_loss = bool(
        completion_delta < -completion_tolerance or slo_delta > slo_tolerance
    )
    service_gain = bool(
        completion_delta > completion_tolerance or slo_delta < -slo_tolerance
    )
    material_cost_growth = bool(
        cost_delta > cost_fraction * max(float(baseline.total_cost), 1.0)
    )
    cost_saving_with_service_loss = bool(cost_delta < 0.0 and material_service_loss)
    passes = not material_service_loss and not (
        material_cost_growth and not service_gain
    )
    return {
        "completion_delta": completion_delta,
        "slo_violation_delta": slo_delta,
        "total_cost_delta": cost_delta,
        "material_service_loss": material_service_loss,
        "material_uncompensated_cost_growth": material_cost_growth and not service_gain,
        "cost_saving_with_material_service_loss": cost_saving_with_service_loss,
        "passes": bool(passes),
    }


def assess_repair_gate(
    states: pd.DataFrame,
    state_summaries: pd.DataFrame,
    episodes: pd.DataFrame,
    aggregation: pd.DataFrame,
    runtime: pd.DataFrame,
    loso: pd.DataFrame,
    thresholds: dict[str, float | int],
) -> dict[str, object]:
    required = {
        "learned_value_branch",
        "original_learned_model",
        "full_repair",
        "dsp_b",
        "conservative_fallback",
    }
    missing = required - set(episodes.method.unique())
    if missing:
        raise ValueError(f"repair gate is missing methods: {sorted(missing)}")
    state_means = state_summaries.groupby("method").mean(numeric_only=True)
    lv_agreement = float(
        state_means.loc["learned_value_branch", "action_consistency_rate"]
    )
    repair_agreement = float(state_means.loc["full_repair", "action_consistency_rate"])
    agreement_loss = lv_agreement - repair_agreement
    old_regret = float(
        state_means.loc["original_learned_model", "mean_Q_star_regret"]
    )
    repair_regret = float(state_means.loc["full_repair", "mean_Q_star_regret"])
    regret_reduction = (old_regret - repair_regret) / max(old_regret, 1.0e-12)

    burst = (
        episodes[episodes.scenario.isin(["early_burst", "late_burst"])]
        .groupby(["method", "scenario", "seed"], as_index=False)
        .mean_Q_star_regret.mean()
    )
    old_burst = burst[burst.method == "original_learned_model"].set_index(
        ["scenario", "seed"]
    )
    repair_burst = burst[burst.method == "full_repair"].set_index(
        ["scenario", "seed"]
    )
    paired_burst = repair_burst.join(
        old_burst, lsuffix="_repair", rsuffix="_old", how="inner"
    )
    wins = paired_burst.mean_Q_star_regret_repair < paired_burst.mean_Q_star_regret_old
    burst_wins = {
        scenario: int(wins.xs(scenario, level="scenario").sum())
        if scenario in set(wins.index.get_level_values("scenario"))
        else 0
        for scenario in ("early_burst", "late_burst")
    }

    service = episodes.groupby("method").mean(numeric_only=True)
    service_vs_old = _service_guardrail(
        service.loc["full_repair"],
        service.loc["original_learned_model"],
        float(thresholds["material_completion_loss"]),
        float(thresholds["material_slo_increase"]),
        float(thresholds["material_uncompensated_cost_increase_fraction"]),
    )
    service_vs_lv = _service_guardrail(
        service.loc["full_repair"],
        service.loc["learned_value_branch"],
        float(thresholds["material_completion_loss"]),
        float(thresholds["material_slo_increase"]),
        float(thresholds["material_uncompensated_cost_increase_fraction"]),
    )

    keys = ["scenario", "seed", "t", "load", "queue", "remaining_budget"]
    lv = states[states.method == "learned_value_branch"].set_index(keys)
    old = states[states.method == "original_learned_model"].set_index(keys)
    repair = states[states.method == "full_repair"].set_index(keys)
    common = lv[["action", "Q_star_selected"]].join(
        old[["action", "Q_star_selected"]], lsuffix="_lv", rsuffix="_old"
    ).join(repair[["action"]].rename(columns={"action": "action_repair"}))
    common["old_damage_vs_lv"] = (
        common.Q_star_selected_lv - common.Q_star_selected_old
    ).clip(lower=0.0)
    positive = common.loc[
        (common.action_lv != common.action_old) & (common.old_damage_vs_lv > 0.0),
        "old_damage_vs_lv",
    ]
    high_threshold = float(positive.quantile(0.75)) if len(positive) else np.inf
    high = common[common.old_damage_vs_lv >= high_threshold]
    old_high_error = float((high.action_old != high.action_lv).mean()) if len(high) else 0.0
    repair_high_error = (
        float((high.action_repair != high.action_lv).mean()) if len(high) else 0.0
    )
    high_error_reduction = (
        (old_high_error - repair_high_error) / max(old_high_error, 1.0e-12)
    )

    aggregation_means = (
        aggregation.groupby("model_round", as_index=False)
        .rollout_q_star_regret.mean()
        .sort_values("model_round")
    )
    aggregation_deltas = aggregation_means.rollout_q_star_regret.diff().dropna()
    aggregation_continuous = bool(
        len(aggregation_deltas) > 0 and np.all(aggregation_deltas < -1.0e-6)
    )
    loso_reversal = bool(loso.regret_reversal.any()) if len(loso) else True
    fallback = runtime[runtime.method == "conservative_fallback"]
    fallback_rate = float(
        fallback.fallback_decisions.sum() / max(fallback.decisions.sum(), 1)
    )

    checks = {
        "agreement_loss_within_0p03": bool(
            agreement_loss
            <= float(thresholds["agreement_loss_vs_learned_value_ceiling"])
        ),
        "q_star_regret_reduction_at_least_40pct": bool(
            regret_reduction
            >= float(thresholds["regret_reduction_vs_old_model_floor"])
        ),
        "early_burst_improves_4_of_5": bool(
            burst_wins["early_burst"]
            >= int(thresholds["burst_seed_wins_required_each"])
        ),
        "late_burst_improves_4_of_5": bool(
            burst_wins["late_burst"]
            >= int(thresholds["burst_seed_wins_required_each"])
        ),
        "service_cost_guardrail": bool(
            service_vs_old["passes"] and service_vs_lv["passes"]
        ),
        "high_regret_error_reduction": bool(
            high_error_reduction
            >= float(thresholds["high_regret_error_reduction_floor"])
        ),
        "aggregation_continuously_improves": aggregation_continuous,
        "loso_no_material_reversal": not loso_reversal,
        "fallback_not_excessive": bool(
            fallback_rate <= float(thresholds["fallback_rate_ceiling"])
        ),
    }
    return {
        "schema": "direct_action_planning_repair.continuation_gate.v1",
        "decision": "CONTINUE" if all(checks.values()) else "STOP",
        "checks": checks,
        "thresholds": thresholds,
        "agreement": {
            "learned_value": lv_agreement,
            "full_repair": repair_agreement,
            "loss": agreement_loss,
        },
        "q_star_regret": {
            "original_learned_model": old_regret,
            "full_repair": repair_regret,
            "relative_reduction": regret_reduction,
        },
        "burst_seed_wins": burst_wins,
        "service_cost_vs_old_model": service_vs_old,
        "service_cost_vs_learned_value": service_vs_lv,
        "high_regret_errors": {
            "threshold": high_threshold,
            "states": int(len(high)),
            "old_error_rate": old_high_error,
            "repair_error_rate": repair_high_error,
            "relative_reduction": high_error_reduction,
        },
        "aggregation_round_means": aggregation_means.to_dict(orient="records"),
        "aggregation_regret_deltas": [float(value) for value in aggregation_deltas],
        "loso_reversal_cells": int(loso.regret_reversal.sum()) if len(loso) else None,
        "loso_cells": int(len(loso)),
        "fallback_rate": fallback_rate,
    }
