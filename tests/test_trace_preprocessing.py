from pathlib import Path

import numpy as np
import pandas as pd

from stage2_dynamic_budget.data.azure_functions import (
    aggregate_invocation_file,
    fit_and_scale_chronological,
    select_bursty_functions,
)


def test_aggregate_invocation_file_separates_public_trace_domains(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "HashOwner": ["a", "b", "c"],
            "HashApp": ["x", "y", "z"],
            "HashFunction": ["f", "g", "h"],
            "Trigger": ["http", "queue", "event"],
            "1": [1, 10, 100],
            "2": [2, 20, 200],
        }
    )
    path = tmp_path / "day.csv"
    frame.to_csv(path, index=False)
    result = aggregate_invocation_file(path, chunksize=2)
    assert np.array_equal(result["http"], [1, 2])
    assert np.array_equal(result["async"], [110, 220])
    assert np.array_equal(result["all"], [111, 222])


def test_scaler_is_fit_on_train_only_and_preserves_chronology() -> None:
    train = np.array([1.0, 2.0, 3.0, 4.0])
    validation = np.array([100.0, 200.0])
    test = np.array([300.0, 400.0])
    scaled, metadata = fit_and_scale_chronological(train, validation, test, target_mean=5.0)
    assert np.isclose(scaled["train"].mean(), 5.0)
    assert scaled["validation"][0] <= scaled["validation"][1]
    assert metadata["fit_scope"] == "train_only"
    assert metadata["train_mean_raw"] == 2.5


def test_train_only_scaler_can_bound_extreme_trace_spikes() -> None:
    train = np.array([0.0] * 90 + [1000.0] * 10)
    scaled, metadata = fit_and_scale_chronological(
        train, train.copy(), train.copy(), target_mean=5.0, max_scaled_value=60.0
    )
    assert np.isclose(scaled["train"].mean(), 5.0)
    assert scaled["train"].max() <= 60.0
    assert metadata["max_scaled_value"] == 60.0


def test_bursty_function_selection_uses_training_files_only(tmp_path: Path) -> None:
    first = pd.DataFrame(
        {
            "HashOwner": ["a", "b", "c"],
            "HashApp": ["x", "y", "z"],
            "HashFunction": ["steady", "bursty", "async_one"],
            "Trigger": ["http", "http", "queue"],
            "1": [5, 0, 1],
            "2": [5, 20, 1],
            "3": [5, 0, 1],
        }
    )
    path = tmp_path / "train.csv"
    first.to_csv(path, index=False)
    selected, audit = select_bursty_functions([path], top_k=1, min_total=1)
    assert selected["http"] == ["bursty"]
    assert audit["selection_scope"] == "train_only"
