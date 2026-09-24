from __future__ import annotations

import pandas as pd


def assess_continuation(
    episodes: pd.DataFrame,
    state_summaries: pd.DataFrame,
    required_budget_seed_wins: int,
    required_burst_seed_wins: int,
    high_risk_floor: float,
) -> dict[str, object]:
    methods: dict[str, dict[str, object]] = {}
    metric_columns = ["action_consistency_rate", "mean_Q_star_regret"]
    budget_seed = (
        episodes.groupby(["method", "budget", "seed"], as_index=False)[metric_columns]
        .mean()
    )
    burst_seed = (
        episodes[episodes.scenario.isin(["early_burst", "late_burst"])]
        .groupby(["method", "scenario", "seed"], as_index=False)[metric_columns]
        .mean()
    )
    aggregate_service = episodes.groupby("method", as_index=True)[
        ["completion_rate", "slo_violation_rate", "total_cost"]
    ].mean()
    aggregate_high_risk = state_summaries.groupby("method")[
        "high_risk_low_cost_balanced_accuracy"
    ].mean()

    baseline_budget = budget_seed[budget_seed.method == "b4_budget_state"].set_index(
        ["budget", "seed"]
    )
    baseline_burst = burst_seed[burst_seed.method == "b4_budget_state"].set_index(
        ["scenario", "seed"]
    )
    for method in ("acba_a", "acba_b"):
        candidate_budget = budget_seed[budget_seed.method == method].set_index(["budget", "seed"])
        paired_budget = candidate_budget.join(
            baseline_budget, lsuffix="_candidate", rsuffix="_b4", how="inner"
        )
        budget_wins = (
            (paired_budget.action_consistency_rate_candidate > paired_budget.action_consistency_rate_b4)
            & (paired_budget.mean_Q_star_regret_candidate < paired_budget.mean_Q_star_regret_b4)
        )
        candidate_burst = burst_seed[burst_seed.method == method].set_index(
            ["scenario", "seed"]
        )
        paired_burst = candidate_burst.join(
            baseline_burst, lsuffix="_candidate", rsuffix="_b4", how="inner"
        )
        burst_wins = (
            (paired_burst.action_consistency_rate_candidate > paired_burst.action_consistency_rate_b4)
            & (paired_burst.mean_Q_star_regret_candidate < paired_burst.mean_Q_star_regret_b4)
        )
        available_scenarios = set(paired_burst.index.get_level_values("scenario"))
        burst_counts = {
            scenario: (
                int(burst_wins.xs(scenario, level="scenario").sum())
                if scenario in available_scenarios
                else 0
            )
            for scenario in ("early_burst", "late_burst")
        }
        high_risk = float(aggregate_high_risk.loc[method])
        b4_high_risk = float(aggregate_high_risk.loc["b4_budget_state"])
        service = aggregate_service.loc[method]
        b4_service = aggregate_service.loc["b4_budget_state"]
        no_lower_cost = service.total_cost >= b4_service.total_cost
        no_better_completion = service.completion_rate <= b4_service.completion_rate
        no_better_slo = service.slo_violation_rate >= b4_service.slo_violation_rate
        at_least_one_strict_service_loss = (
            service.completion_rate < b4_service.completion_rate
            or service.slo_violation_rate > b4_service.slo_violation_rate
        )
        strict_service_deterioration = bool(
            no_lower_cost
            and no_better_completion
            and no_better_slo
            and at_least_one_strict_service_loss
        )
        checks = {
            "budget_seed_majority": int(budget_wins.sum()) >= required_budget_seed_wins,
            "early_burst_majority": burst_counts["early_burst"] >= required_burst_seed_wins,
            "late_burst_majority": burst_counts["late_burst"] >= required_burst_seed_wins,
            "high_risk_identification": high_risk > high_risk_floor and high_risk > b4_high_risk,
            "not_service_dominated": not strict_service_deterioration,
        }
        methods[method] = {
            "paired_budget_seed_wins": int(budget_wins.sum()),
            "paired_budget_seed_total": int(len(budget_wins)),
            "burst_seed_wins": burst_counts,
            "high_risk_balanced_accuracy": high_risk,
            "b4_high_risk_balanced_accuracy": b4_high_risk,
            "mean_service": {key: float(value) for key, value in service.items()},
            "b4_mean_service": {key: float(value) for key, value in b4_service.items()},
            "checks": checks,
            "passes": all(checks.values()),
        }
    passing = [method for method, audit in methods.items() if audit["passes"]]
    if passing:
        selected = sorted(
            passing,
            key=lambda method: (-int(methods[method]["paired_budget_seed_wins"]), method),
        )[0]
        decision = "CONTINUE"
    else:
        selected = None
        decision = "STOP"
    return {
        "schema": "acba.continuation_gate.v1",
        "decision": decision,
        "selected_method": selected,
        "thresholds": {
            "paired_budget_seed_wins_required": required_budget_seed_wins,
            "burst_seed_wins_required_each": required_burst_seed_wins,
            "high_risk_balanced_accuracy_floor": high_risk_floor,
        },
        "methods": methods,
    }
