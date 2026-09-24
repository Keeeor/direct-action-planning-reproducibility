from __future__ import annotations

import numpy as np

from stage2_dynamic_budget.direct_action_planning_dataset_validation.analysis import (
    _bh_adjust,
    _paired_test,
)


def test_bh_adjust_is_monotone_in_rank() -> None:
    p = np.array([0.04, 0.001, 0.02])
    q = _bh_adjust(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1.0e-12)
    assert np.all(q >= p)


def test_paired_test_reports_directional_counts() -> None:
    result = _paired_test(np.array([1.0, 2.0, -1.0, 0.0]), seed=1, bootstrap_draws=100)
    assert result["wins"] == 2
    assert result["ties"] == 1
    assert result["losses"] == 1
