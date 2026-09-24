import numpy as np

from stage2_dynamic_budget.data.trace_windows import select_trace_window


def test_trace_window_is_deterministic_and_contained() -> None:
    values = np.arange(100, dtype=float)
    first, start = select_trace_window(values, horizon=16, seed=9)
    second, second_start = select_trace_window(values, horizon=16, seed=9)
    assert start == second_start
    assert np.array_equal(first, second)
    assert np.array_equal(first, values[start : start + 16])
    assert 0 <= start <= 84


def test_trace_window_rejects_short_partition() -> None:
    try:
        select_trace_window(np.arange(4), horizon=5, seed=0)
    except ValueError as exc:
        assert "shorter" in str(exc)
    else:
        raise AssertionError("short trace partition should fail")

