from __future__ import annotations

import numpy as np

from dap.direct_action_planning_controlled_aggregation.analysis import (
    _bh_adjust,
)


def test_bh_adjust_is_monotone_and_bounded() -> None:
    values = np.asarray([0.04, 0.001, 0.03, 0.2])
    adjusted = _bh_adjust(values)
    order = np.argsort(values)
    assert np.all(np.diff(adjusted[order]) >= -1.0e-12)
    assert np.all(adjusted >= values)
    assert np.all(adjusted <= 1.0)
