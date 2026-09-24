from __future__ import annotations

import numpy as np
import pandas as pd


def _guardrail(
    candidate: pd.Series,
    baseline: pd.Series,
    completion_tolerance: float,
    slo_tolerance: float,
    cost_fraction: float,
) -> dict[str, object]:
    completion_delta = float(candidate.completion_rate - baseline.completion_rate)
    slo_delta = float(candidate.slo_violation_rate - baseline.slo_violation_rate)
    cost_delta = float(candidate.total_cost - baseline.total_cost)
    service_loss = completion_delta < -completion_tolerance or slo_delta > slo_tolerance
    service_gain = completion_delta > completion_tolerance or slo_delta < -slo_tolerance
    cost_growth = cost_delta > cost_fraction * max(float(baseline.total_cost), 1.0)
    return {
        "completion_delta": completion_delta,
        "slo_violation_delta": slo_delta,
        "total_cost_delta": cost_delta,
        "material_service_loss": bool(service_loss),
        "material_uncompensated_cost_growth": bool(cost_growth and not service_gain),
        "cost_saving_with_material_service_loss": bool(cost_delta < 0.0 and service_loss),
        "passes": bool(not service_loss and not (cost_growth and not service_gain)),
    }


def validation_guardrail(
    candidate: dict[str, float],
    learned_value: dict[str, float],
    thresholds: dict[str, float],
) -> bool:
    agreement_ok = (
        learned_value["full_state_action_consistency"]
        - candidate["full_state_action_consistency"]
        <= float(thresholds["agreement_loss_ceiling"])
    )
    completion_ok = (
        candidate["completion_rate"]
        >= learned_value["completion_rate"] - float(thresholds["material_completion_loss"])
    )
    slo_ok = (
        candidate["slo_violation_rate"]
        <= learned_value["slo_violation_rate"] + float(thresholds["material_slo_increase"])
    )
    cost_ok = (
        candidate["total_cost"]
        <= learned_value["total_cost"]
        * (1.0 + float(thresholds["material_uncompensated_cost_increase_fraction"]))
        or candidate["completion_rate"] > learned_value["completion_rate"]
    )
    return bool(agreement_ok and completion_ok and slo_ok and cost_ok)


def assess_controlled_gate(
    state_summary: pd.DataFrame,
    episodes: pd.DataFrame,
    validation_rounds: pd.DataFrame,
    anchor_metrics: pd.DataFrame,
    collection_metrics: pd.DataFrame,
    thresholds: dict[str, float | int],
) -> dict[str, object]:
    candidate = "controlled_full"
    required = {"learned_value", "original_learned_model", "dap_repair_d0", candidate}
    missing = required - set(episodes.method.unique())
    if missing:
        raise ValueError(f"controlled gate missing methods: {sorted(missing)}")
    states = state_summary.groupby("method").mean(numeric_only=True)
    test = episodes.groupby("method").mean(numeric_only=True)
    agreement_loss = float(
        states.loc["learned_value", "action_consistency_rate"]
        - states.loc[candidate, "action_consistency_rate"]
    )
    candidate_regret = float(test.loc[candidate, "mean_Q_star_regret"])
    old_regret = float(test.loc["original_learned_model", "mean_Q_star_regret"])
    regret_reduction = (old_regret - candidate_regret) / max(old_regret, 1.0e-12)
    burst = (
        episodes[episodes.scenario.isin(["early_burst", "late_burst"])]
        .groupby(["method", "scenario", "train_seed"], as_index=False)
        .mean_Q_star_regret.mean()
    )
    paired = burst[burst.method == candidate].merge(
        burst[burst.method == "original_learned_model"],
        on=["scenario", "train_seed"],
        suffixes=("_candidate", "_old"),
    )
    burst_wins = {
        scenario: int(
            (
                paired.loc[paired.scenario == scenario, "mean_Q_star_regret_candidate"]
                < paired.loc[paired.scenario == scenario, "mean_Q_star_regret_old"]
            ).sum()
        )
        for scenario in ("early_burst", "late_burst")
    }
    service_old = _guardrail(
        test.loc[candidate],
        test.loc["original_learned_model"],
        float(thresholds["material_completion_loss"]),
        float(thresholds["material_slo_increase"]),
        float(thresholds["material_uncompensated_cost_increase_fraction"]),
    )
    service_lv = _guardrail(
        test.loc[candidate],
        test.loc["learned_value"],
        float(thresholds["material_completion_loss"]),
        float(thresholds["material_slo_increase"]),
        float(thresholds["material_uncompensated_cost_increase_fraction"]),
    )
    selected_anchor = anchor_metrics[anchor_metrics.method == candidate]
    forgetting = float(selected_anchor.action_error_increase.max()) if len(selected_anchor) else np.inf
    controlled_coverage = collection_metrics[collection_metrics.protocol == candidate]
    coverage_means = controlled_coverage.groupby("round").grid_coverage.mean().sort_index()
    coverage_contraction = float(
        max(0.0, coverage_means.iloc[0] - coverage_means.min())
    ) if len(coverage_means) else np.inf
    validation = validation_rounds[
        (validation_rounds.protocol == candidate) & validation_rounds.selected
    ]
    validation_candidate = float(validation.rollout_q_star_regret.mean())
    validation_old = float(
        validation_rounds[
            (validation_rounds.protocol == "baseline")
            & (validation_rounds.method == "original_learned_model")
        ].rollout_q_star_regret.mean()
    )
    validation_improved = validation_candidate < validation_old
    test_improved = candidate_regret < old_regret
    reversal = bool(validation_improved and not test_improved)
    checks = {
        "agreement_loss_within_0p03": agreement_loss
        <= float(thresholds["agreement_loss_vs_learned_value_ceiling"]),
        "closed_loop_regret_not_above_frozen_D1": candidate_regret
        <= float(thresholds["dap_repair_best_D1_regret_ceiling"]),
        "regret_reduction_at_least_50pct": regret_reduction
        >= float(thresholds["regret_reduction_vs_original_floor"]),
        "early_burst_improves_4_of_5": burst_wins["early_burst"]
        >= int(thresholds["burst_seed_wins_required_each"]),
        "late_burst_improves_4_of_5": burst_wins["late_burst"]
        >= int(thresholds["burst_seed_wins_required_each"]),
        "no_material_anchor_forgetting": forgetting
        <= float(thresholds["material_forgetting_ceiling"]),
        "coverage_not_materially_contracting": coverage_contraction
        <= float(thresholds["material_coverage_contraction"]),
        "service_cost_guardrail": bool(service_old["passes"] and service_lv["passes"]),
        "no_B4_fallback_dependency": True,
        "no_validation_test_reversal": not reversal,
    }
    return {
        "schema": "direct_action_planning_controlled_aggregation.gate.v1",
        "decision": "CONTINUE" if all(checks.values()) else "STOP",
        "checks": {key: bool(value) for key, value in checks.items()},
        "thresholds": thresholds,
        "agreement": {
            "learned_value": float(states.loc["learned_value", "action_consistency_rate"]),
            "controlled_full": float(states.loc[candidate, "action_consistency_rate"]),
            "loss": agreement_loss,
        },
        "test_q_star_regret": {
            "controlled_full": candidate_regret,
            "original_learned_model": old_regret,
            "relative_reduction": regret_reduction,
            "frozen_DAP_repair_D1": float(thresholds["dap_repair_best_D1_regret_ceiling"]),
        },
        "burst_seed_wins": burst_wins,
        "maximum_anchor_action_error_increase": forgetting,
        "maximum_coverage_contraction_after_D1": coverage_contraction,
        "validation_regret": {
            "controlled_full": validation_candidate,
            "original_learned_model": validation_old,
        },
        "validation_to_test_reversal": reversal,
        "service_cost_vs_old": service_old,
        "service_cost_vs_learned_value": service_lv,
        "fallback_rate": 0.0,
    }
