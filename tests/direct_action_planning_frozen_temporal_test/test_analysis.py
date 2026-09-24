from __future__ import annotations

import pandas as pd

from stage2_dynamic_budget.direct_action_planning_frozen_temporal_test.analysis import (
    _integrity,
    _stat_spec,
    build_analysis_tables,
)


def _episodes() -> pd.DataFrame:
    rows = []
    for dataset in ("azure2019", "gentd26"):
        for budget in (48.0, 96.0, 144.0):
            for seed in range(5):
                for episode in range(2):
                    for method, delta, cost in (
                        ("dap_calibrated", 4.0, 8.0),
                        ("double_dqn", 0.0, 9.0),
                        ("dap_immediate", 2.0, 8.5),
                    ):
                        rows.append(
                            {
                                "dataset": dataset,
                                "domain": "d",
                                "budget": budget,
                                "training_seed": seed,
                                "method": method,
                                "episode": episode,
                                "window_seed": seed * 100 + episode,
                                "window_start": seed + episode,
                                "discounted_return": 10.0 + delta,
                                "completion_ratio": 0.90 + delta / 100.0,
                                "slo_violation_rate": 0.20 - delta / 100.0,
                                "total_cost": cost,
                                "queue_area": 10.0 - delta,
                                "decision_ms_mean": 0.1,
                                "decision_ms_p95": 0.2,
                                "budget_overspend": 0.0,
                            }
                        )
    return pd.DataFrame(rows)


def test_build_analysis_tables_uses_seed_blocks_and_complete_pairing() -> None:
    tables = build_analysis_tables(
        _episodes(),
        method_families={"double_dqn": "primary", "dap_immediate": "internal"},
    )
    assert len(tables["unit_metrics"]) == 2 * 3 * 5 * 3
    assert tables["pairing_integrity"]["passed"] is True
    comparison = tables["paired_comparisons"]
    row = comparison[
        comparison.dataset.eq("azure2019")
        & comparison.baseline.eq("double_dqn")
        & comparison.metric.eq("discounted_return")
    ].iloc[0]
    assert row.n_seed_blocks == 5
    assert row.mean_favorable_difference == 4.0
    assert row.unit_wins == 15


def test_pareto_table_distinguishes_external_and_internal_comparisons() -> None:
    tables = build_analysis_tables(
        _episodes(),
        method_families={"double_dqn": "primary", "dap_immediate": "internal"},
    )
    pareto = tables["pareto_counts"]
    external = pareto[
        pareto.dataset.eq("gentd26")
        & pareto.baseline.eq("double_dqn")
        & pareto.relation.eq("return_cost")
    ].iloc[0]
    assert external.dap_dominates == 15
    assert external.baseline_dominates == 0
    internal = tables["internal_ablation"]
    assert set(internal.control) == {"dap_immediate"}


def test_integrity_uses_the_frozen_episode_matrix() -> None:
    integrity = _integrity(_episodes(), pd.DataFrame())
    assert integrity["expected_run_units"] == 30
    assert integrity["expected_methods"] == 10
    assert integrity["expected_episode_rows"] == 3000


def test_stat_spec_marks_only_primary_return_as_hypothesis() -> None:
    tables = build_analysis_tables(
        _episodes(),
        method_families={"double_dqn": "primary", "dap_immediate": "internal"},
    )
    claims = _stat_spec(tables["paired_comparisons"])["claims"]
    hypotheses = [claim for claim in claims if claim["is_hypothesis"]]
    assert len(hypotheses) == 2
    assert {claim["metric"] for claim in hypotheses} == {"discounted_return"}
    assert {claim["target"] for claim in hypotheses} == {
        "dap_calibrated_vs_double_dqn"
    }
