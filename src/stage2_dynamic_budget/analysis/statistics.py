from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import wilcoxon


def _paired(candidate: Iterable[float], baseline: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(list(candidate), dtype=np.float64)
    right = np.asarray(list(baseline), dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired samples must be one-dimensional and have equal shape")
    mask = np.isfinite(left) & np.isfinite(right)
    if not mask.any():
        raise ValueError("paired samples contain no finite pair")
    return left[mask], right[mask]


def paired_bootstrap(
    candidate: Iterable[float],
    baseline: Iterable[float],
    *,
    seed: int = 20260801,
    n_resamples: int = 20_000,
) -> dict[str, float]:
    """Percentile bootstrap CI for candidate minus baseline using paired rows."""
    left, right = _paired(candidate, baseline)
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    differences = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(n_resamples, len(differences)))
    draws = differences[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "n_pairs": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def paired_effect_size(candidate: Iterable[float], baseline: Iterable[float]) -> float:
    """Cohen's dz: paired mean difference divided by paired sample SD."""
    left, right = _paired(candidate, baseline)
    difference = left - right
    mean = float(difference.mean())
    if len(difference) < 2:
        return math.nan
    standard_deviation = float(difference.std(ddof=1))
    if standard_deviation <= 1e-15:
        if mean > 0:
            return math.inf
        if mean < 0:
            return -math.inf
        return math.nan
    return mean / standard_deviation


def paired_wilcoxon(candidate: Iterable[float], baseline: Iterable[float]) -> float:
    left, right = _paired(candidate, baseline)
    if np.allclose(left, right):
        return 1.0
    return float(wilcoxon(left, right, zero_method="pratt", alternative="two-sided").pvalue)


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a one-dimensional array in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted
