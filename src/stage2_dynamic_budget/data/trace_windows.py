from __future__ import annotations

import numpy as np


def select_trace_window(values, horizon: int, seed: int) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("trace partition must be one-dimensional")
    if horizon <= 0 or len(array) < horizon:
        raise ValueError("trace partition is shorter than the requested positive horizon")
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(array) - horizon + 1))
    return array[start : start + horizon].copy(), start
