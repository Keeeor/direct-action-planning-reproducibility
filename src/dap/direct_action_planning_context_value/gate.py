from __future__ import annotations

import pandas as pd


def assess_oracle_context_gate(cell_metrics: pd.DataFrame) -> dict[str, object]:
    required = {"scenario", "model_seed", "method", "regret"}
    if missing := required - set(cell_metrics.columns):
        raise ValueError(f"oracle metrics missing columns: {sorted(missing)}")
    means = (
        cell_metrics.groupby(["scenario", "method"], as_index=False).regret.mean()
        .pivot(index="scenario", columns="method", values="regret")
    )
    methods = {"frozen_D1", "oracle_scenario_value", "oracle_phase_value"}
    if missing := methods - set(means.columns):
        raise ValueError(f"oracle metrics missing methods: {sorted(missing)}")
    checks: dict[str, bool] = {}
    scenario_rows: dict[str, dict[str, float]] = {}
    for scenario, row in means.iterrows():
        d1 = float(row["frozen_D1"])
        oracle_scenario = float(row["oracle_scenario_value"])
        oracle_phase = float(row["oracle_phase_value"])
        checks[f"oracle_scenario_beats_D1__{scenario}"] = oracle_scenario < d1 - 1.0e-12
        checks[f"oracle_phase_beats_D1__{scenario}"] = oracle_phase < d1 - 1.0e-12
        scenario_rows[str(scenario)] = {
            "frozen_D1_regret": d1,
            "oracle_scenario_regret": oracle_scenario,
            "oracle_phase_regret": oracle_phase,
            "oracle_scenario_gain": d1 - oracle_scenario,
            "oracle_phase_gain": d1 - oracle_phase,
        }
    return {
        "schema": "direct_action_planning_context_value.oracle_gate.v1",
        "decision": "CONTINUE" if checks and all(checks.values()) else "STOP",
        "checks": checks,
        "scenario_means": scenario_rows,
        "oracle_inputs_are_diagnostic_only": True,
    }


def assess_context_gate(
    episodes: pd.DataFrame,
    state_summary: pd.DataFrame,
    oracle_gate: dict[str, object],
    leakage_passed: bool,
    config: dict,
) -> dict[str, object]:
    required = {
        "scenario",
        "model_seed",
        "method",
        "mean_Q_star_regret",
        "completion_rate",
        "slo_violation_rate",
        "total_cost",
        "return_gap_to_paired_optimal",
    }
    if missing := required - set(episodes.columns):
        raise ValueError(f"test metrics missing columns: {sorted(missing)}")
    unit = (
        episodes.groupby(["scenario", "model_seed", "method"], as_index=False)
        .agg(
            regret=("mean_Q_star_regret", "mean"),
            completion=("completion_rate", "mean"),
            slo=("slo_violation_rate", "mean"),
            cost=("total_cost", "mean"),
            return_gap=("return_gap_to_paired_optimal", "mean"),
        )
    )
    methods = {
        "frozen_D1",
        "oracle_scenario_value",
        "final_context_value",
        "domain_value_refresh",
        "in_domain_context_value",
    }
    if missing := methods - set(unit.method):
        raise ValueError(f"test metrics missing methods: {sorted(missing)}")
    regret = unit.pivot(index=["scenario", "model_seed"], columns="method", values="regret")
    wins = regret["final_context_value"] < regret["frozen_D1"] - 1.0e-12
    scenario_means = unit.groupby(["scenario", "method"], as_index=False).regret.mean().pivot(
        index="scenario", columns="method", values="regret"
    )
    no_scenario_reversal = bool(
        (
            scenario_means["final_context_value"]
            <= scenario_means["frozen_D1"] + 1.0e-12
        ).all()
    )
    d1_regret = float(regret["frozen_D1"].mean())
    final_regret = float(regret["final_context_value"].mean())
    oracle_regret = float(regret["oracle_scenario_value"].mean())
    oracle_gain = max(d1_regret - oracle_regret, 1.0e-15)
    recovered = (d1_regret - final_regret) / oracle_gain
    method_means = unit.groupby("method").mean(numeric_only=True)
    completion_delta = float(
        method_means.loc["final_context_value", "completion"]
        - method_means.loc["frozen_D1", "completion"]
    )
    slo_delta = float(
        method_means.loc["final_context_value", "slo"]
        - method_means.loc["frozen_D1", "slo"]
    )
    cost_delta = float(
        method_means.loc["final_context_value", "cost"]
        - method_means.loc["frozen_D1", "cost"]
    )
    return_delta = float(
        method_means.loc["final_context_value", "return_gap"]
        - method_means.loc["frozen_D1", "return_gap"]
    )
    cost_ceiling = max(
        abs(float(method_means.loc["frozen_D1", "cost"]))
        * float(config["material_uncompensated_cost_increase_fraction"]),
        1.0e-12,
    )
    service_cost_guard = (
        completion_delta >= -float(config["material_completion_loss"])
        and slo_delta <= float(config["material_slo_increase"])
        and cost_delta <= cost_ceiling
    )
    in_domain_delta = float(
        method_means.loc["in_domain_context_value", "regret"]
        - method_means.loc["domain_value_refresh", "regret"]
    )

    alias_means = (
        state_summary[
            state_summary.method.isin(["frozen_D1", "final_context_value"])
            & state_summary.alias_region.isin([False, True])
        ]
        .groupby(["method", "alias_region"])
        .mean(numeric_only=True)["mean_Q_star_regret"]
    )
    alias_improvement = float(
        alias_means.loc[("frozen_D1", True)]
        - alias_means.loc[("final_context_value", True)]
    )
    nonalias_improvement = float(
        alias_means.loc[("frozen_D1", False)]
        - alias_means.loc[("final_context_value", False)]
    )
    checks = {
        "oracle_context_all_scenarios": oracle_gate.get("decision") == "CONTINUE",
        "history_recovers_70pct_oracle_gain": recovered
        >= float(config["oracle_gain_recovery_floor"]),
        "no_heldout_scenario_mean_reversal": no_scenario_reversal,
        "wins_at_least_10_of_15": int(wins.sum()) >= int(config["wins_required_of_15"]),
        "improvement_concentrated_in_alias_region": alias_improvement > 0.0
        and alias_improvement >= nonalias_improvement - 1.0e-12,
        "no_material_in_domain_regression": in_domain_delta
        <= float(config["in_domain_regret_increase_ceiling"]),
        "service_cost_guardrail": service_cost_guard,
        "closed_loop_return_benefit": return_delta < -1.0e-12,
        "future_information_leakage_check": bool(leakage_passed),
        "independent_final_test_better_than_D1": final_regret < d1_regret - 1.0e-12,
    }
    return {
        "schema": "direct_action_planning_context_value.continuation_gate.v1",
        "decision": "CONTINUE" if all(checks.values()) else "STOP",
        "checks": checks,
        "regret": {
            "frozen_D1": d1_regret,
            "final_context_value": final_regret,
            "oracle_scenario_value": oracle_regret,
            "oracle_gain_recovered": recovered,
            "in_domain_delta_vs_value_refresh": in_domain_delta,
        },
        "cell_wins": int(wins.sum()),
        "scenario_means": {
            str(scenario): {
                "frozen_D1": float(row["frozen_D1"]),
                "final_context_value": float(row["final_context_value"]),
                "oracle_scenario_value": float(row["oracle_scenario_value"]),
            }
            for scenario, row in scenario_means.iterrows()
        },
        "alias_region": {
            "alias_improvement": alias_improvement,
            "nonalias_improvement": nonalias_improvement,
        },
        "service_cost_vs_D1": {
            "completion_delta": completion_delta,
            "slo_delta": slo_delta,
            "cost_delta": cost_delta,
            "return_gap_delta": return_delta,
        },
    }
