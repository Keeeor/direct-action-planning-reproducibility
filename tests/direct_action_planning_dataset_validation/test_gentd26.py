from __future__ import annotations

import numpy as np
import pandas as pd

from stage2_dynamic_budget.data.gentd26 import (
    GenTD26Grid,
    aggregate_request_arrivals,
    aggregate_qps_domain,
    chronological_three_way_split,
    infer_native_grid,
)


def test_native_grid_and_qps_aggregation_are_column_driven() -> None:
    grid = infer_native_grid(
        [np.array([100.0, 157.0, 214.0]), np.array([100.0, 214.0])]
    )
    frame = pd.DataFrame(
        {
            "value": [1.0, 2.0, 4.0],
            "request_type": ["A", "A", "B"],
            "timestamp_anon": [100.0, 100.0, 157.0],
        }
    )
    actual = aggregate_qps_domain(frame, grid, "A")
    np.testing.assert_allclose(actual, [3.0, 0.0, 0.0])


def test_chronological_split_is_contiguous_and_exhaustive() -> None:
    values = np.arange(11, dtype=float)
    parts = chronological_three_way_split(values)
    np.testing.assert_array_equal(parts["train"], values[:6])
    np.testing.assert_array_equal(parts["validation"], values[6:8])
    np.testing.assert_array_equal(parts["test"], values[8:])
    np.testing.assert_array_equal(
        np.concatenate([parts["train"], parts["validation"], parts["test"]]),
        values,
    )


def test_qps_rejects_off_grid_timestamp() -> None:
    grid = GenTD26Grid(origin=100.0, interval_seconds=57.0, size=3)
    frame = pd.DataFrame(
        {"timestamp_anon": [101.0], "value": [1.0], "request_type": ["A"]}
    )
    try:
        aggregate_qps_domain(frame, grid, "A")
    except ValueError as exc:
        assert "align" in str(exc)
    else:
        raise AssertionError("off-grid QPS timestamp was accepted")


def test_request_aggregation_uses_submission_time_not_outcome() -> None:
    frame = pd.DataFrame(
        {
            "gmt_create": [
                "2024-01-01 00:01:00",
                "2024-01-01 00:09:00",
                "2024-01-01 00:11:00",
            ],
            "predict_type": ["A", "A", "B"],
            "predict_status": ["SUCCEED", "FAILED", "SUCCEED"],
        }
    )
    index = pd.date_range("2024-01-01 00:00:00", periods=3, freq="10min")
    actual = aggregate_request_arrivals(frame, ("A",), full_index=index)
    np.testing.assert_allclose(actual.to_numpy(), [2.0, 0.0, 0.0])
