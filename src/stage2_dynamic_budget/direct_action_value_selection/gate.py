from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = {
    "paired_budget_seed_wins_required": 8,
    "learned_value_agreement_loss_ceiling": 0.02,
    "burst_seed_wins_required_each": 3,
    "material_completion_loss": 0.01,
    "material_slo_increase": 0.01,
    "loso_agreement_deficit_ceiling": 0.02,
    "loso_regret_excess_ceiling": 0.05,
    "uncertainty_error_auroc_floor": 0.60,
    "uncertainty_error_spearman_floor": 0.10,
}

DAVS_METHODS = ("davs_r", "davs_rank", "davs_ensemble")


def _method_mean(frame: pd.DataFrame, method: str) -> dict[str, float]:
    values = frame[frame.method == method][
        ["completion_rate", "slo_violation_rate", "total_cost"]
    ].mean()
    return {key: float(value) for key, value in values.items()}


def _service_guardrail(
    candidate: dict[str, float],
    baseline: dict[str, float],
    completion_loss: float,
    slo_increase: float,
) -> dict[str, object]:
    dominated = bool(
        candidate["completion_rate"] <= baseline["completion_rate"]
        and candidate["slo_violation_rate"] >= baseline["slo_violation_rate"]
        and candidate["total_cost"] >= baseline["total_cost"]
        and (
            candidate["completion_rate"] < baseline["completion_rate"]
            or candidate["slo_violation_rate"] > baseline["slo_violation_rate"]
        )
    )
    material_loss = bool(
        candidate["completion_rate"] < baseline["completion_rate"] - completion_loss
        or candidate["slo_violation_rate"] > baseline["slo_violation_rate"] + slo_increase
    )
    cheaper_with_loss = bool(candidate["total_cost"] < baseline["total_cost"] and material_loss)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "pareto_dominated": dominated,
        "material_service_loss": material_loss,
        "cost_saving_with_material_service_loss": cheaper_with_loss,
        "passes": not dominated and not cheaper_with_loss,
    }


def assess_continuation(
    metrics: pd.DataFrame,
    loso: pd.DataFrame,
    uncertainty: pd.DataFrame,
    thresholds: dict[str, float | int],
) -> dict[str, object]:
    limits = {**DEFAULT_THRESHOLDS, **thresholds}
    required = {"dsp_b", "learned_value_branch", *DAVS_METHODS}
    missing = required - set(metrics.method.unique())
    if missing:
        raise ValueError(f"continuation metrics are missing methods: {sorted(missing)}")
    cells = metrics.groupby(["method", "scenario", "budget", "seed"], as_index=False).mean(
        numeric_only=True
    )
    pooled = cells.groupby(["method", "budget", "seed"], as_index=False)[
        ["action_consistency_rate", "mean_Q_star_regret"]
    ].mean()
    dsp_pooled = pooled[pooled.method == "dsp_b"].set_index(["budget", "seed"])
    learned_agreement = float(
        pooled[pooled.method == "learned_value_branch"].action_consistency_rate.mean()
    )
    loso_available = {
        "method",
        "heldout_scenario",
        "action_consistency_rate",
        "mean_Q_star_regret",
    }.issubset(loso.columns)
    method_checks: dict[str, object] = {}
    for method in DAVS_METHODS:
        candidate = pooled[pooled.method == method].set_index(["budget", "seed"])
        paired = candidate.join(dsp_pooled, lsuffix="_candidate", rsuffix="_dsp", how="inner")
        wins = (
            paired.action_consistency_rate_candidate > paired.action_consistency_rate_dsp
        ) & (paired.mean_Q_star_regret_candidate < paired.mean_Q_star_regret_dsp)
        candidate_agreement = float(candidate.action_consistency_rate.mean())

        burst_counts: dict[str, int] = {}
        for scenario in ("early_burst", "late_burst"):
            scenario_cells = cells[cells.scenario == scenario]
            candidate_seed = scenario_cells[scenario_cells.method == method].groupby("seed")[
                "mean_Q_star_regret"
            ].mean()
            dsp_seed = scenario_cells[scenario_cells.method == "dsp_b"].groupby("seed")[
                "mean_Q_star_regret"
            ].mean()
            shared = candidate_seed.index.intersection(dsp_seed.index)
            burst_counts[scenario] = int(
                (candidate_seed.loc[shared] < dsp_seed.loc[shared]).sum()
            )

        service = _service_guardrail(
            _method_mean(metrics, method),
            _method_mean(metrics, "dsp_b"),
            float(limits["material_completion_loss"]),
            float(limits["material_slo_increase"]),
        )
        if loso_available:
            loso_candidate = loso[loso.method == method].groupby("heldout_scenario")[
                ["action_consistency_rate", "mean_Q_star_regret"]
            ].mean()
            loso_dsp = loso[loso.method == "dsp_b"].groupby("heldout_scenario")[
                ["action_consistency_rate", "mean_Q_star_regret"]
            ].mean()
        else:
            loso_candidate = pd.DataFrame(
                columns=["action_consistency_rate", "mean_Q_star_regret"]
            )
            loso_dsp = loso_candidate.copy()
        shared_scenarios = loso_candidate.index.intersection(loso_dsp.index)
        if len(shared_scenarios) < 3:
            loso_pass = False
            worst_agreement_deficit = float("inf")
            worst_regret_excess = float("inf")
        else:
            agreement_deficits = (
                loso_dsp.loc[shared_scenarios, "action_consistency_rate"]
                - loso_candidate.loc[shared_scenarios, "action_consistency_rate"]
            )
            regret_excesses = (
                loso_candidate.loc[shared_scenarios, "mean_Q_star_regret"]
                - loso_dsp.loc[shared_scenarios, "mean_Q_star_regret"]
            )
            worst_agreement_deficit = float(agreement_deficits.max())
            worst_regret_excess = float(regret_excesses.max())
            loso_pass = bool(
                worst_agreement_deficit <= float(limits["loso_agreement_deficit_ceiling"])
                and worst_regret_excess <= float(limits["loso_regret_excess_ceiling"])
            )
        checks = {
            "stable_over_dsp_b": bool(
                int(wins.sum()) >= int(limits["paired_budget_seed_wins_required"])
            ),
            "learned_value_fidelity": bool(
                learned_agreement - candidate_agreement
                <= float(limits["learned_value_agreement_loss_ceiling"])
            ),
            "early_burst_regret_improves": bool(
                burst_counts["early_burst"] >= int(limits["burst_seed_wins_required_each"])
            ),
            "late_burst_regret_improves": bool(
                burst_counts["late_burst"] >= int(limits["burst_seed_wins_required_each"])
            ),
            "service_cost_guardrail": bool(service["passes"]),
            "leave_one_scenario_no_reversal": loso_pass,
        }
        method_checks[method] = {
            "passes": all(checks.values()),
            "checks": checks,
            "paired_budget_seed_wins": int(wins.sum()),
            "paired_budget_seed_total": int(len(wins)),
            "action_agreement": candidate_agreement,
            "learned_value_action_agreement": learned_agreement,
            "action_agreement_loss": learned_agreement - candidate_agreement,
            "burst_seed_regret_wins": burst_counts,
            "service_cost_guardrail": service,
            "loso": {
                "worst_agreement_deficit_vs_dsp_b": worst_agreement_deficit,
                "worst_regret_excess_vs_dsp_b": worst_regret_excess,
            },
        }

    ensemble_rows = uncertainty[uncertainty.method == "davs_ensemble"]
    mean_auroc = float(ensemble_rows.error_auroc.mean()) if not ensemble_rows.empty else np.nan
    mean_spearman = (
        float(ensemble_rows.error_spearman.mean()) if not ensemble_rows.empty else np.nan
    )
    ensemble_pass = bool(
        np.isfinite(mean_auroc)
        and np.isfinite(mean_spearman)
        and mean_auroc >= float(limits["uncertainty_error_auroc_floor"])
        and mean_spearman >= float(limits["uncertainty_error_spearman_floor"])
    )
    checks = {
        "at_least_one_davs_passes_method_gate": any(
            bool(value["passes"]) for value in method_checks.values()
        ),
        "ensemble_uncertainty_identifies_errors": ensemble_pass,
    }
    return {
        "schema": "direct_action_value_selection.continuation_gate.v1",
        "decision": "CONTINUE" if all(checks.values()) else "STOP",
        "checks": checks,
        "thresholds": limits,
        "methods": method_checks,
        "ensemble_uncertainty": {
            "error_auroc": mean_auroc,
            "error_spearman": mean_spearman,
            "passes": ensemble_pass,
        },
    }
