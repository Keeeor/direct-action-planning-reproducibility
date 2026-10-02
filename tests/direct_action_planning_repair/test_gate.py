from __future__ import annotations

import pandas as pd

from dap.direct_action_planning_repair.gate import assess_repair_gate


def test_repair_gate_requires_continuous_aggregation_and_both_bursts() -> None:
    state_summaries = pd.DataFrame(
        [
            {"method": "learned_value_branch", "action_consistency_rate": 0.96, "mean_Q_star_regret": 0.02},
            {"method": "original_learned_model", "action_consistency_rate": 0.90, "mean_Q_star_regret": 0.10},
            {"method": "full_repair", "action_consistency_rate": 0.94, "mean_Q_star_regret": 0.04},
        ]
    )
    state_rows = []
    for method, action, q in (
        ("learned_value_branch", 1, 1.0),
        ("original_learned_model", 0, 0.0),
        ("full_repair", 1, 1.0),
    ):
        state_rows.append(
            {
                "method": method,
                "scenario": "early_burst",
                "seed": 0,
                "t": 0,
                "load": 1,
                "queue": 0,
                "remaining_budget": 4,
                "action": action,
                "Q_star_selected": q,
            }
        )
    episode_rows = []
    methods = [
        "learned_value_branch",
        "original_learned_model",
        "full_repair",
        "dsp_b",
        "conservative_fallback",
    ]
    for method in methods:
        for scenario in ("early_burst", "late_burst"):
            for seed in range(5):
                episode_rows.append(
                    {
                        "method": method,
                        "scenario": scenario,
                        "seed": seed,
                        "mean_Q_star_regret": 0.04 if method == "full_repair" else 0.10,
                        "completion_rate": 0.8,
                        "slo_violation_rate": 0.2,
                        "total_cost": 4.0,
                    }
                )
    aggregation = pd.DataFrame(
        {"model_round": [0, 1, 2], "rollout_q_star_regret": [0.10, 0.08, 0.06]}
    )
    runtime = pd.DataFrame(
        [{"method": "conservative_fallback", "fallback_decisions": 2, "decisions": 10}]
    )
    loso = pd.DataFrame([{"regret_reversal": False}])
    thresholds = {
        "agreement_loss_vs_learned_value_ceiling": 0.03,
        "regret_reduction_vs_old_model_floor": 0.40,
        "burst_seed_wins_required_each": 4,
        "material_completion_loss": 0.01,
        "material_slo_increase": 0.01,
        "material_uncompensated_cost_increase_fraction": 0.05,
        "high_regret_error_reduction_floor": 0.30,
        "fallback_rate_ceiling": 0.35,
    }
    gate = assess_repair_gate(
        pd.DataFrame(state_rows),
        state_summaries,
        pd.DataFrame(episode_rows),
        aggregation,
        runtime,
        loso,
        thresholds,
    )
    assert gate["decision"] == "CONTINUE"
    flat = aggregation.copy()
    flat.loc[2, "rollout_q_star_regret"] = 0.08
    stopped = assess_repair_gate(
        pd.DataFrame(state_rows),
        state_summaries,
        pd.DataFrame(episode_rows),
        flat,
        runtime,
        loso,
        thresholds,
    )
    assert stopped["decision"] == "STOP"
    assert not stopped["checks"]["aggregation_continuously_improves"]
