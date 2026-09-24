from __future__ import annotations

import numpy as np

from stage2_dynamic_budget.direct_action_planning_repair.analysis import _bh_adjust


def test_bh_adjustment_is_monotone_in_sorted_p_values() -> None:
    values = np.asarray([0.04, 0.001, 0.03, 0.2])
    adjusted = _bh_adjust(values)
    order = np.argsort(values)
    assert np.all(np.diff(adjusted[order]) >= -1.0e-12)
    assert np.all(adjusted >= values)
    assert np.all(adjusted <= 1.0)
