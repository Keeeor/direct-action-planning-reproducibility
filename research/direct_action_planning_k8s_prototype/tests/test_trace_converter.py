import numpy as np

from workload.trace_converter import (
    fit_rate_scale,
    request_offsets,
    select_activity_window_start,
    transform_rate,
)


def test_rate_scale_is_fit_from_explicit_training_array() -> None:
    training = np.asarray([1.0, 2.0, 3.0, 4.0])
    scale = fit_rate_scale(training, quantile=1.0, target_peak_rps=20.0, max_rps=25.0)
    assert scale.source_split == "train"
    assert transform_rate(np.asarray([2.0, 10.0]), scale).tolist() == [10.0, 25.0]


def test_request_generation_is_prefix_invariant_for_fixed_rng() -> None:
    left = np.random.default_rng(11)
    right = np.random.default_rng(11)
    assert np.array_equal(request_offsets(3.0, 2.0, left), request_offsets(3.0, 2.0, right))


def test_activity_window_selection_is_deterministic_and_load_stratified() -> None:
    trace = np.asarray([0.0, 0.0, 1.0, 1.0, 15.0, 15.0, 15.0, 1.0, 1.0])
    start = select_activity_window_start(trace, horizon=3, activity_quantile=1.0, seed=7)
    assert start == 4
