from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ASYNC_TRIGGERS = {"queue", "event", "storage", "orchestration"}


def aggregate_invocation_file(path: str | Path, chunksize: int = 2_000) -> dict[str, np.ndarray]:
    """Aggregate per-function minute columns into reproducible load domains."""
    totals: dict[str, np.ndarray | None] = {"all": None, "http": None, "async": None}
    for chunk in pd.read_csv(path, chunksize=chunksize):
        minute_values = chunk.iloc[:, 4:].to_numpy(dtype=np.float64, copy=False)
        triggers = chunk["Trigger"].astype(str).str.lower().to_numpy()
        selections = {
            "all": np.ones(len(chunk), dtype=bool),
            "http": triggers == "http",
            "async": np.isin(triggers, list(ASYNC_TRIGGERS)),
        }
        for domain, mask in selections.items():
            partial = minute_values[mask].sum(axis=0) if mask.any() else np.zeros(minute_values.shape[1])
            totals[domain] = partial if totals[domain] is None else totals[domain] + partial
    if totals["all"] is None:
        raise ValueError(f"empty invocation file: {path}")
    return {key: np.asarray(value, dtype=np.float64) for key, value in totals.items()}


def fit_and_scale_chronological(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    *,
    target_mean: float = 6.0,
    train_clip_quantile: float = 0.999,
    max_scaled_value: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | str]]:
    """Fit scale and robust cap on train only, then transform later time blocks."""
    parts = {
        "train": np.asarray(train, dtype=np.float64),
        "validation": np.asarray(validation, dtype=np.float64),
        "test": np.asarray(test, dtype=np.float64),
    }
    if any(values.ndim != 1 or not len(values) for values in parts.values()):
        raise ValueError("all chronological partitions must be non-empty vectors")
    if any(np.any(values < 0) or not np.all(np.isfinite(values)) for values in parts.values()):
        raise ValueError("trace partitions must contain finite non-negative values")
    train_mean = float(parts["train"].mean())
    if train_mean <= 0:
        raise ValueError("training trace has zero mean load")
    train_cap = float(np.quantile(parts["train"], train_clip_quantile))
    clipped_train_mean = float(np.minimum(parts["train"], train_cap).mean())
    if max_scaled_value is None:
        scale = target_mean / clipped_train_mean
    else:
        if max_scaled_value <= target_mean:
            raise ValueError("max_scaled_value must exceed target_mean")
        clipped_train = np.minimum(parts["train"], train_cap)
        low, high = 0.0, target_mean / clipped_train_mean
        while np.minimum(clipped_train * high, max_scaled_value).mean() < target_mean:
            high *= 2.0
            if high > 1e12:
                raise ValueError("cannot attain target mean under the requested scaled cap")
        for _ in range(80):
            middle = 0.5 * (low + high)
            if np.minimum(clipped_train * middle, max_scaled_value).mean() < target_mean:
                low = middle
            else:
                high = middle
        scale = high
    transformed = {
        key: np.minimum(
            np.minimum(values, train_cap) * scale,
            max_scaled_value if max_scaled_value is not None else np.inf,
        )
        for key, values in parts.items()
    }
    metadata: dict[str, float | str] = {
        "fit_scope": "train_only",
        "train_mean_raw": train_mean,
        "train_clip_quantile": train_clip_quantile,
        "train_clip_raw": train_cap,
        "clipped_train_mean_raw": clipped_train_mean,
        "scale_factor": scale,
        "target_train_mean": target_mean,
        "max_scaled_value": max_scaled_value if max_scaled_value is not None else "unbounded",
    }
    return transformed, metadata


def select_bursty_functions(
    training_files: list[str | Path],
    *,
    top_k: int = 32,
    min_total: float = 1_000.0,
    chunksize: int = 2_000,
) -> tuple[dict[str, list[str]], dict]:
    """Select high-CV function IDs using training days only."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    stats: dict[str, list[float | str]] = {}
    for path in training_files:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            values = chunk.iloc[:, 4:].to_numpy(dtype=np.float64, copy=False)
            identifiers = chunk["HashFunction"].astype(str).to_numpy()
            triggers = chunk["Trigger"].astype(str).str.lower().to_numpy()
            sums = values.sum(axis=1)
            sum_squares = np.square(values).sum(axis=1)
            maxima = values.max(axis=1)
            for identifier, trigger, total, squared, maximum in zip(
                identifiers, triggers, sums, sum_squares, maxima
            ):
                if identifier not in stats:
                    stats[identifier] = [0.0, 0.0, 0.0, 0.0, trigger]
                row = stats[identifier]
                row[0] = float(row[0]) + float(total)
                row[1] = float(row[1]) + float(squared)
                row[2] = float(row[2]) + values.shape[1]
                row[3] = max(float(row[3]), float(maximum))
    candidates: dict[str, list[tuple[float, str, float]]] = {"http": [], "async": []}
    for identifier, (total, squared, count, maximum, trigger) in stats.items():
        domain = "http" if trigger == "http" else "async" if trigger in ASYNC_TRIGGERS else None
        if domain is None or float(total) < min_total:
            continue
        mean = float(total) / float(count)
        variance = max(float(squared) / float(count) - mean * mean, 0.0)
        coefficient = float(np.sqrt(variance) / max(mean, 1e-12))
        candidates[domain].append((coefficient, identifier, float(total)))
    selected = {
        domain: [identifier for _, identifier, _ in sorted(rows, reverse=True)[:top_k]]
        for domain, rows in candidates.items()
    }
    if any(not identifiers for identifiers in selected.values()):
        raise ValueError("no eligible bursty functions in one or more domains")
    audit = {
        "selection_scope": "train_only",
        "training_files": [Path(path).name for path in training_files],
        "top_k": top_k,
        "min_total": min_total,
        "eligible_counts": {key: len(value) for key, value in candidates.items()},
    }
    return selected, audit


def aggregate_selected_functions(
    path: str | Path,
    selected: dict[str, list[str]],
    chunksize: int = 2_000,
) -> dict[str, np.ndarray]:
    selected_sets = {key: set(value) for key, value in selected.items()}
    totals: dict[str, np.ndarray | None] = {"http_bursty": None, "async_bursty": None}
    for chunk in pd.read_csv(path, chunksize=chunksize):
        values = chunk.iloc[:, 4:].to_numpy(dtype=np.float64, copy=False)
        identifiers = chunk["HashFunction"].astype(str)
        for base_domain, output_domain in (("http", "http_bursty"), ("async", "async_bursty")):
            mask = identifiers.isin(selected_sets[base_domain]).to_numpy()
            partial = values[mask].sum(axis=0) if mask.any() else np.zeros(values.shape[1])
            totals[output_domain] = partial if totals[output_domain] is None else totals[output_domain] + partial
    return {key: np.asarray(value, dtype=np.float64) for key, value in totals.items()}
