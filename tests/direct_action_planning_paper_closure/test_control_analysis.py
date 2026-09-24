from __future__ import annotations

import numpy as np

from stage2_dynamic_budget.direct_action_planning_paper_closure.analysis import _bh, _pareto


def test_bh_is_monotonic_after_pvalue_sorting():
    raw = np.asarray([0.04, 0.001, 0.02, 0.5])
    adjusted = _bh(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= -1.0e-12)
    assert np.all(adjusted >= raw)


def test_pareto_requires_service_and_cost_improvement():
    assert _pareto(2.0, 1.0, 1.0, 1.0) == "dap_dominates"
    assert _pareto(2.0, 2.0, 1.0, 1.0) == "tradeoff"
