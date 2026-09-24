from __future__ import annotations

import numpy as np


def split_by_time(values, train_fraction: float = 0.6, validation_fraction: float = 0.2):
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("values must have a time dimension")
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("invalid split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test segment")
    n = len(array)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    return array[:train_end].copy(), array[train_end:validation_end].copy(), array[validation_end:].copy()
