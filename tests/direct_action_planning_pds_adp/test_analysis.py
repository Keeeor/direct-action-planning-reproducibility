from __future__ import annotations

import numpy as np
import pandas as pd

from dap.direct_action_planning_pds_adp.analysis import (
    benjamini_hochberg,
    build_seed_units,
    classify_pareto,
    paired_comparison,
)


def test_seed_unit_averages_domains_and_episodes_without_pseudoreplication():
    rows = []
    for training_seed in (1, 2):
        for method, offset in (("dap_calibrated", 1.0), ("pds_adp", 0.0)):
            for domain in ("a", "b"):
                for episode in (0, 1):
                    rows.append(
                        {
                            "dataset": "trace",
                            "budget": 10.0,
                            "training_seed": training_seed,
                            "method": method,
                            "domain": domain,
                            "episode": episode,
                            "discounted_return": float(training_seed + offset),
                            "completion_ratio": 0.9,
                            "slo_violation_rate": 0.1,
                            "total_cost": 5.0,
                            "decision_ms_mean": 0.2,
                            "decision_ms_p95": 0.3,
                        }
                    )
    units = build_seed_units(pd.DataFrame(rows))
    assert len(units) == 4
    assert set(units.groupby(["dataset", "budget", "method"]).size()) == {2}
    paired = paired_comparison(units, metric="discounted_return", bootstrap_seed=7)
    assert len(paired) == 1
    assert paired.iloc[0]["n"] == 2
    assert paired.iloc[0]["mean_difference_dap_minus_pds"] == 1.0


def test_bh_is_monotone_in_sorted_p_value_order():
    q = benjamini_hochberg(np.asarray([0.04, 0.01, 0.03, 0.20]))
    order = np.argsort([0.04, 0.01, 0.03, 0.20])
    assert np.all(np.diff(q[order]) >= -1.0e-12)
    assert np.all((0.0 <= q) & (q <= 1.0))


def test_pareto_classification_requires_joint_service_and_cost_noninferiority():
    assert classify_pareto(
        dap_completion=0.95, dap_slo=0.05, dap_cost=7,
        pds_completion=0.94, pds_slo=0.06, pds_cost=8,
    ) == "dap_dominates"
    assert classify_pareto(
        dap_completion=0.95, dap_slo=0.05, dap_cost=9,
        pds_completion=0.94, pds_slo=0.06, pds_cost=8,
    ) == "tradeoff"
    assert classify_pareto(
        dap_completion=0.94, dap_slo=0.06, dap_cost=9,
        pds_completion=0.95, pds_slo=0.05, pds_cost=8,
    ) == "pds_dominates"
