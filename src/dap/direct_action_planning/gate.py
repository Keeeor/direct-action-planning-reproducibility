from __future__ import annotations

import pandas as pd


DEFAULT_THRESHOLDS = {
    "oracle_action_agreement_floor": 0.999,
    "oracle_q_regret_ceiling": 1.0e-10,
    "paired_budget_seed_wins_required": 8,
    "burst_seed_wins_required_each": 3,
    "material_completion_loss": 0.01,
    "material_slo_increase": 0.01,
    "learned_model_agreement_loss_ceiling": 0.05,
    "learned_model_regret_increase_ceiling": 0.05,
}


def _mean_service(frame: pd.DataFrame, method: str) -> dict[str, float]:
    row = frame[frame.method == method][
        ["completion_rate", "slo_violation_rate", "total_cost"]
    ].mean()
    return {key: float(value) for key, value in row.items()}


def _service_guardrail(
    candidate: dict[str, float],
    baseline: dict[str, float],
    completion_loss: float,
    slo_increase: float,
) -> dict[str, object]:
    completion_worse = candidate["completion_rate"] < baseline["completion_rate"]
    slo_worse = candidate["slo_violation_rate"] > baseline["slo_violation_rate"]
    no_cheaper = candidate["total_cost"] >= baseline["total_cost"]
    dominated = bool(
        no_cheaper
        and candidate["completion_rate"] <= baseline["completion_rate"]
        and candidate["slo_violation_rate"] >= baseline["slo_violation_rate"]
        and (completion_worse or slo_worse)
    )
    material_loss = bool(
        candidate["completion_rate"] < baseline["completion_rate"] - completion_loss
        or candidate["slo_violation_rate"] > baseline["slo_violation_rate"] + slo_increase
    )
    cost_saving_with_material_service_loss = bool(
        candidate["total_cost"] < baseline["total_cost"] and material_loss
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "pareto_dominated": dominated,
        "material_service_loss": material_loss,
        "cost_saving_with_material_service_loss": cost_saving_with_material_service_loss,
        "passes": not dominated and not cost_saving_with_material_service_loss,
    }


def assess_continuation(
    episodes: pd.DataFrame,
    state_summaries: pd.DataFrame,
    thresholds: dict[str, float | int],
) -> dict[str, object]:
    limits = {**DEFAULT_THRESHOLDS, **thresholds}
    required_methods = {
        "dsp_b",
        "oracle_branch",
        "learned_value_branch",
        "learned_model_branch",
    }
    missing = required_methods - set(episodes.method.unique())
    if missing:
        raise ValueError(f"continuation gate is missing methods: {sorted(missing)}")

    oracle = state_summaries[state_summaries.method == "oracle_branch"]
    if oracle.empty:
        raise ValueError("full-state Oracle-Branch summary is required")
    oracle_agreement = float(oracle.action_consistency_rate.mean())
    oracle_regret = float(oracle.mean_Q_star_regret.mean())

    metrics = ["action_consistency_rate", "mean_Q_star_regret"]
    budget_seed = episodes.groupby(["method", "budget", "seed"], as_index=False)[
        metrics
    ].mean()
    dsp_budget = budget_seed[budget_seed.method == "dsp_b"].set_index(["budget", "seed"])
    learned_budget = budget_seed[
        budget_seed.method == "learned_value_branch"
    ].set_index(["budget", "seed"])
    paired_budget = learned_budget.join(
        dsp_budget, lsuffix="_learned", rsuffix="_dsp_b", how="inner"
    )
    paired_wins = (
        (
            paired_budget.action_consistency_rate_learned
            > paired_budget.action_consistency_rate_dsp_b
        )
        & (
            paired_budget.mean_Q_star_regret_learned
            < paired_budget.mean_Q_star_regret_dsp_b
        )
    )

    burst = episodes[episodes.scenario.isin(["early_burst", "late_burst"])]
    burst_seed = burst.groupby(["method", "scenario", "seed"], as_index=False)[
        "mean_Q_star_regret"
    ].mean()
    dsp_burst = burst_seed[burst_seed.method == "dsp_b"].set_index(["scenario", "seed"])
    learned_burst = burst_seed[
        burst_seed.method == "learned_value_branch"
    ].set_index(["scenario", "seed"])
    paired_burst = learned_burst.join(
        dsp_burst, lsuffix="_learned", rsuffix="_dsp_b", how="inner"
    )
    regret_wins = (
        paired_burst.mean_Q_star_regret_learned
        < paired_burst.mean_Q_star_regret_dsp_b
    )
    available_bursts = set(regret_wins.index.get_level_values("scenario"))
    burst_counts = {
        scenario: (
            int(regret_wins.xs(scenario, level="scenario").sum())
            if scenario in available_bursts
            else 0
        )
        for scenario in ("early_burst", "late_burst")
    }

    dsp_service = _mean_service(episodes, "dsp_b")
    learned_service = _mean_service(episodes, "learned_value_branch")
    service_guardrail = _service_guardrail(
        learned_service,
        dsp_service,
        float(limits["material_completion_loss"]),
        float(limits["material_slo_increase"]),
    )

    model_metrics = episodes.groupby("method")[metrics].mean()
    model_agreement_loss = float(
        model_metrics.loc["learned_value_branch", "action_consistency_rate"]
        - model_metrics.loc["learned_model_branch", "action_consistency_rate"]
    )
    model_regret_increase = float(
        model_metrics.loc["learned_model_branch", "mean_Q_star_regret"]
        - model_metrics.loc["learned_value_branch", "mean_Q_star_regret"]
    )
    model_service_guardrail = _service_guardrail(
        _mean_service(episodes, "learned_model_branch"),
        learned_service,
        float(limits["material_completion_loss"]),
        float(limits["material_slo_increase"]),
    )

    checks = {
        "oracle_near_exact": bool(
            oracle_agreement >= float(limits["oracle_action_agreement_floor"])
            and oracle_regret <= float(limits["oracle_q_regret_ceiling"])
        ),
        "learned_value_budget_seed_majority": bool(
            int(paired_wins.sum())
            >= int(limits["paired_budget_seed_wins_required"])
        ),
        "early_burst_regret_improves": bool(
            burst_counts["early_burst"]
            >= int(limits["burst_seed_wins_required_each"])
        ),
        "late_burst_regret_improves": bool(
            burst_counts["late_burst"]
            >= int(limits["burst_seed_wins_required_each"])
        ),
        "learned_value_service_cost_guardrail": bool(service_guardrail["passes"]),
        "learned_model_loss_controllable": bool(
            model_agreement_loss
            <= float(limits["learned_model_agreement_loss_ceiling"])
            and model_regret_increase
            <= float(limits["learned_model_regret_increase_ceiling"])
            and model_service_guardrail["passes"]
        ),
    }
    return {
        "schema": "direct_action_planning.continuation_gate.v1",
        "decision": "CONTINUE" if all(checks.values()) else "STOP",
        "checks": checks,
        "thresholds": limits,
        "oracle": {
            "action_agreement": oracle_agreement,
            "mean_Q_star_regret": oracle_regret,
        },
        "learned_value_vs_dsp_b": {
            "paired_budget_seed_wins": int(paired_wins.sum()),
            "paired_budget_seed_total": int(len(paired_wins)),
            "burst_seed_regret_wins": burst_counts,
            "service_cost_guardrail": service_guardrail,
        },
        "learned_model_vs_learned_value": {
            "action_agreement_loss": model_agreement_loss,
            "mean_Q_star_regret_increase": model_regret_increase,
            "service_cost_guardrail": model_service_guardrail,
        },
    }
