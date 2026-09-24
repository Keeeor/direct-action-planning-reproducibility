from __future__ import annotations

import math


def oracle_recovery(regret_base: float, regret_adapted: float, regret_oracle: float) -> float:
    gain = float(regret_base) - float(regret_oracle)
    if abs(gain) <= 1.0e-12:
        return 1.0 if float(regret_adapted) <= float(regret_base) + 1.0e-12 else 0.0
    return float((float(regret_base) - float(regret_adapted)) / gain)


def assess_support_adaptation_gate(summary: dict[str, object]) -> dict[str, object]:
    scenarios = dict(summary["original_scenario_improvement"])
    checks = {
        "interpolation_stable_vs_D1": float(summary["interpolation_win_fraction"]) >= 0.60,
        "interpolation_better_than_extrapolation": float(summary["interpolation_regret"])
        < float(summary["extrapolation_regret"]),
        "deployable_support_relationship": bool(summary["deployable_support_clear"]),
        "recovery_5pct_at_least_50pct": float(summary["recovery_5pct"]) >= 0.50,
        "recovery_10pct_at_least_70pct": float(summary["recovery_10pct"]) >= 0.70,
        "all_original_loso_scenarios_improve": all(bool(value) for value in scenarios.values())
        and set(scenarios) == {"early_burst", "late_burst", "periodic"},
        "wins_at_least_10_of_15": int(summary["original_cell_wins"]) >= 10,
        "no_material_anchor_forgetting": float(summary["anchor_mae_increase"]) <= 0.02,
        "no_material_in_domain_regression": float(summary["in_domain_regret_increase"])
        <= 0.002,
        "closed_loop_return_improved": bool(summary["closed_loop_return_improved"]),
        "service_cost_guardrail": bool(summary["service_cost_guardrail"]),
        "prefix_suffix_leakage_check": bool(summary["leakage_passed"]),
        "independent_final_test_better": bool(summary["independent_test_better"]),
    }
    finite = all(
        math.isfinite(float(summary[key]))
        for key in (
            "interpolation_win_fraction",
            "interpolation_regret",
            "extrapolation_regret",
            "recovery_5pct",
            "recovery_10pct",
            "anchor_mae_increase",
            "in_domain_regret_increase",
        )
    )
    checks["finite_metrics"] = finite
    return {
        "schema": "direct_action_planning_support_adaptation.continuation_gate.v1",
        "decision": "CONTINUE" if all(checks.values()) else "STOP",
        "checks": checks,
        "summary": summary,
    }
