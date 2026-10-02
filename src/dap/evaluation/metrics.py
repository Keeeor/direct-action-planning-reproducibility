from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr


def _array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def summarize_episode(
    step_rows: Iterable[dict[str, Any]], budget: float, horizon: int
) -> dict[str, float]:
    rows = list(step_rows)
    if not rows:
        raise ValueError("cannot summarize an empty episode")
    rewards = _array(rows, "reward")
    costs = _array(rows, "resource_cost")
    queues = _array(rows, "queue_length")
    mean_latency = _array(rows, "mean_latency")
    tail_latency = _array(rows, "tail_latency")
    violations = _array(rows, "slo_violation")
    arrivals = _array(rows, "arrivals")
    served = _array(rows, "served")
    total_cost = float(costs.sum())
    cumulative = np.cumsum(costs)
    exhaustion = np.flatnonzero(cumulative >= budget) if budget > 0 else np.array([], dtype=int)
    exhaustion_ratio = float((exhaustion[0] + 1) / horizon) if exhaustion.size else 1.0
    mechanism = _mechanism_metrics(rows)
    return {
        "episode_reward": float(rewards.sum()),
        "total_cost": total_cost,
        "mean_step_cost": float(costs.mean()),
        "over_budget": float(total_cost > budget + 1e-9),
        "over_budget_ratio": float(max(total_cost - budget, 0.0) / max(budget, 1e-8)),
        "budget_utilization": float(total_cost / max(budget, 1e-8)),
        "remaining_budget": float(max(budget - total_cost, 0.0)),
        "budget_exhaustion_ratio": exhaustion_ratio,
        "mean_latency": float(mean_latency.mean()),
        "p95_latency": float(np.quantile(tail_latency, 0.95)),
        "p99_latency": float(np.quantile(tail_latency, 0.99)),
        "slo_violation_rate": float(violations.mean()),
        "completion_rate": float(served.sum() / max(arrivals.sum(), 1.0)),
        "mean_queue": float(queues.mean()),
        "max_queue": float(queues.max()),
        **mechanism,
    }


def _mechanism_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not all("local_budget" in row for row in rows):
        return {
            "risk_budget_correlation": math.nan,
            "budget_reallocation_ratio": math.nan,
            "high_risk_budget_mean": math.nan,
            "low_risk_budget_mean": math.nan,
        }
    risk = _array(rows, "risk_level")
    local = _array(rows, "local_budget")
    if len(risk) < 2 or np.allclose(risk, risk[0]) or np.allclose(local, local[0]):
        correlation = math.nan
    else:
        correlation = float(spearmanr(risk, local).statistic)
    low_cut, high_cut = np.quantile(risk, [0.25, 0.75])
    low_values = local[risk <= low_cut]
    high_values = local[risk >= high_cut]
    low_mean = float(low_values.mean()) if low_values.size else math.nan
    high_mean = float(high_values.mean()) if high_values.size else math.nan
    if not np.isfinite(low_mean):
        ratio = math.nan
    elif low_mean <= 1e-8:
        ratio = math.inf if high_mean > 1e-8 else math.nan
    else:
        ratio = high_mean / low_mean
    return {
        "risk_budget_correlation": correlation,
        "budget_reallocation_ratio": float(ratio),
        "high_risk_budget_mean": high_mean,
        "low_risk_budget_mean": low_mean,
    }
