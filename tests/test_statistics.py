import numpy as np

from dap.analysis.statistics import (
    benjamini_hochberg,
    paired_bootstrap,
    paired_effect_size,
)


def test_paired_bootstrap_is_reproducible_and_uses_direction():
    candidate = np.array([1.0, 2.0, 3.0, 4.0])
    baseline = np.array([2.0, 3.0, 4.0, 5.0])
    first = paired_bootstrap(candidate, baseline, seed=7, n_resamples=2_000)
    second = paired_bootstrap(candidate, baseline, seed=7, n_resamples=2_000)
    assert first == second
    assert first["mean_difference"] == -1.0
    assert first["ci_low"] == -1.0
    assert first["ci_high"] == -1.0


def test_paired_effect_size_handles_zero_variance():
    assert paired_effect_size([1, 1, 1], [2, 2, 2]) == float("-inf")
    assert paired_effect_size([2, 2, 2], [1, 1, 1]) == float("inf")
    assert np.isnan(paired_effect_size([1, 1], [1, 1]))


def test_bh_fdr_is_monotone_in_original_order():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert np.allclose(adjusted, [0.04, 0.0533333333, 0.0533333333, 0.20])

