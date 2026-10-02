from __future__ import annotations

import numpy as np

from dap.direct_action_planning_k8s_native_comparison.analysis import (
    bootstrap_ci,
    exact_sign_flip_p,
    holm_adjust,
)


def test_constant_paired_effect_has_exact_interval_and_sign_flip() -> None:
    assert bootstrap_ci([1.0] * 10, seed=1, draws=100) == (1.0, 1.0)
    assert exact_sign_flip_p([1.0] * 10) == 2 / 1024


def test_holm_adjustment_is_monotone_in_p_order() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])
