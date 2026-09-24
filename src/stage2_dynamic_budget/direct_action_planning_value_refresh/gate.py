from __future__ import annotations

import numpy as np
import pandas as pd


def assess_causal_gate(
    state_summary: pd.DataFrame,
    rollout_summary: pd.DataFrame,
    oracle_relative_gain_floor: float,
    repaired_oracle_regret_ceiling: float,
) -> dict[str, object]:
    state_unit = (
        state_summary.groupby(["scenario", "seed", "source", "planner"], as_index=False)
        .q_star_regret.mean()
    )
    state_mean = state_unit.groupby("planner").q_star_regret.mean()
    rollout_unit = rollout_summary.groupby(
        ["scenario", "model_seed", "planner"], as_index=False
    ).rollout_q_star_regret.mean()
    rollout_mean = rollout_unit.groupby("planner").rollout_q_star_regret.mean()
    fixed_name = "repaired_transition_fixed_value"
    oracle_name = "repaired_transition_oracle_value"
    state_fixed = float(state_mean[fixed_name])
    state_oracle = float(state_mean[oracle_name])
    rollout_fixed = float(rollout_mean[fixed_name])
    rollout_oracle = float(rollout_mean[oracle_name])
    state_gain = (state_fixed - state_oracle) / max(state_fixed, 1.0e-15)
    rollout_gain = (rollout_fixed - rollout_oracle) / max(rollout_fixed, 1.0e-15)
    checks = {
        "oracle_state_gain_material": state_gain >= oracle_relative_gain_floor,
        "oracle_rollout_gain_material": rollout_gain >= oracle_relative_gain_floor,
        "repaired_oracle_below_frozen_D1": rollout_oracle
        < repaired_oracle_regret_ceiling,
        "oracle_gain_present_in_early_burst": bool(
            np.all(
                rollout_unit[rollout_unit.scenario == "early_burst"]
                .pivot(index="model_seed", columns="planner", values="rollout_q_star_regret")
                .eval(f"{oracle_name} < {fixed_name}")
            )
        ),
        "oracle_gain_present_in_late_burst": bool(
            np.all(
                rollout_unit[rollout_unit.scenario == "late_burst"]
                .pivot(index="model_seed", columns="planner", values="rollout_q_star_regret")
                .eval(f"{oracle_name} < {fixed_name}")
            )
        ),
    }


def assess_value_refresh_gate(
    state_summaries: pd.DataFrame,
    episodes: pd.DataFrame,
    anchors: pd.DataFrame,
    loso: pd.DataFrame,
    causal_gate: dict[str, object],
    config: dict,
) -> dict[str, object]:
    final_name = "final_value_refresh"
    d1_name = "frozen_best_D1"
    learned_name = "learned_value"
    state_mean = state_summaries.groupby("method").agg(
        action_consistency=("action_consistency_rate", "mean"),
        full_state_regret=("mean_Q_star_regret", "mean"),
    )
    unit = (
        episodes.groupby(["scenario", "train_seed", "method"], as_index=False)
        .agg(
            regret=("mean_Q_star_regret", "mean"),
            completion=("completion_rate", "mean"),
            slo=("slo_violation_rate", "mean"),
            cost=("total_cost", "mean"),
            return_gap=("return_gap_to_paired_optimal", "mean"),
        )
    )
    pivot = unit.pivot(index=["scenario", "train_seed"], columns="method", values="regret")
    wins = pivot[final_name] < pivot[d1_name] - 1.0e-12
    burst_wins = {
        scenario: int(wins.xs(scenario, level="scenario").sum())
        for scenario in ("early_burst", "late_burst")
    }
    mean = unit.groupby("method").mean(numeric_only=True)
    d1_regret = float(mean.loc[d1_name, "regret"])
    final_regret = float(mean.loc[final_name, "regret"])
    oracle_regret = float(
        causal_gate["rollout_regret"]["repaired_oracle"]  # type: ignore[index]
    )
    oracle_gain = max(d1_regret - oracle_regret, 1.0e-15)
    recovered = (d1_regret - final_regret) / oracle_gain
    completion_delta = float(mean.loc[final_name, "completion"] - mean.loc[d1_name, "completion"])
    slo_delta = float(mean.loc[final_name, "slo"] - mean.loc[d1_name, "slo"])
    cost_delta = float(mean.loc[final_name, "cost"] - mean.loc[d1_name, "cost"])
    return_gap_delta = float(
        mean.loc[final_name, "return_gap"] - mean.loc[d1_name, "return_gap"]
    )
    material_service_loss = (
        completion_delta < -float(config["material_completion_loss"])
        or slo_delta > float(config["material_slo_increase"])
    )
    material_cost_growth = cost_delta > max(
        abs(float(mean.loc[d1_name, "cost"]))
        * float(config["material_uncompensated_cost_increase_fraction"]),
        1.0e-12,
    )
    final_anchors = anchors[anchors.method == final_name]
    anchor_value_increase = float(final_anchors.value_mae_increase.max())
    anchor_ranking_loss = float(final_anchors.previously_correct_action_loss.max())
    loso_reversal = bool(
        loso.empty
        or np.any(loso.loso_regret > loso.frozen_D1_regret * (1.0 + config["loso_reversal_fraction"]))
    )
    agreement_loss = float(
        state_mean.loc[learned_name, "action_consistency"]
        - state_mean.loc[final_name, "action_consistency"]
    )
    checks = {
        "oracle_value_causality": causal_gate["decision"] == "CONTINUE",
        "oracle_gain_recovery_at_least_70pct": recovered >= float(
            config["oracle_gain_recovery_floor"]
        ),
        "test_regret_below_frozen_D1_ceiling": final_regret
        < float(config["frozen_D1_regret_ceiling"]),
        "wins_at_least_10_of_15": int(wins.sum()) >= int(config["wins_required_of_15"]),
        "early_burst_improves_4_of_5": burst_wins["early_burst"]
        >= int(config["burst_seed_wins_required_each"]),
        "late_burst_improves_4_of_5": burst_wins["late_burst"]
        >= int(config["burst_seed_wins_required_each"]),
        "no_material_anchor_forgetting": anchor_value_increase
        <= float(config["material_anchor_value_mae_increase"])
        and anchor_ranking_loss <= float(config["material_anchor_ranking_loss"]),
        "loso_no_material_reversal": not loso_reversal,
        "service_cost_guardrail": not material_service_loss and not material_cost_growth,
        "closed_loop_service_cost_benefit": return_gap_delta < -1.0e-12
        and not material_service_loss
        and not material_cost_growth,
        "independent_test_better_than_D1": final_regret < d1_regret,
        "agreement_loss_within_0p03": agreement_loss
        <= float(config["agreement_loss_vs_learned_value_ceiling"]),
    }
    return {
        "schema": "direct_action_planning_value_refresh.continuation_gate.v1",
        "decision": "CONTINUE" if all(checks.values()) else "STOP",
        "checks": checks,
        "test_regret": {
            "frozen_best_D1": d1_regret,
            "final_value_refresh": final_regret,
            "absolute_ceiling": float(config["frozen_D1_regret_ceiling"]),
            "oracle_gain_recovered": recovered,
        },
        "agreement": {
            "learned_value": float(state_mean.loc[learned_name, "action_consistency"]),
            "final_value_refresh": float(state_mean.loc[final_name, "action_consistency"]),
            "loss": agreement_loss,
        },
        "cell_wins": {"total": int(wins.sum()), "burst": burst_wins},
        "anchor_forgetting": {
            "max_value_mae_increase": anchor_value_increase,
            "max_previously_correct_action_loss": anchor_ranking_loss,
        },
        "service_cost_vs_D1": {
            "completion_delta": completion_delta,
            "slo_delta": slo_delta,
            "cost_delta": cost_delta,
            "paired_return_gap_delta": return_gap_delta,
        },
        "loso_material_reversal": loso_reversal,
    }
    return {
        "schema": "direct_action_planning_value_refresh.causal_gate.v1",
        "decision": "CONTINUE" if all(checks.values()) else "STOP",
        "checks": checks,
        "thresholds": {
            "oracle_relative_gain_floor": oracle_relative_gain_floor,
            "repaired_oracle_regret_ceiling": repaired_oracle_regret_ceiling,
        },
        "state_regret": {
            "repaired_fixed": state_fixed,
            "repaired_oracle": state_oracle,
            "relative_gain": state_gain,
        },
        "rollout_regret": {
            "repaired_fixed": rollout_fixed,
            "repaired_oracle": rollout_oracle,
            "relative_gain": rollout_gain,
        },
    }
