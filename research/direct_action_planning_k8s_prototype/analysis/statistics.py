from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_METRICS = ("completion_rate", "slo_violation_rate", "ready_replica_seconds")


def _number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _bootstrap(values: list[float], *, seed: int = 20260806, draws: int = 10_000) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_statistics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("run_status") != "completed" or row.get("method") is None:
            continue
        key = (row.get("profile", ""), row.get("budget_seconds", ""), row.get("seed", ""))
        cells[key][row["method"]] = row
    comparisons: dict[tuple[str, str, str], list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for (profile, budget, _seed), by_method in cells.items():
        dap = by_method.get("dap")
        if dap is None:
            continue
        for method, other in by_method.items():
            if method != "dap":
                comparisons[(profile, budget, method)].append((dap, other))
    result: list[dict[str, Any]] = []
    # Positive delta is favorable for completion; negative delta is favorable
    # for SLO violation and resource cost. The direction field makes this
    # explicit instead of treating request samples as independent repetitions.
    directions = {
        "completion_rate": "DAP minus baseline; positive favors DAP",
        "slo_violation_rate": "DAP minus baseline; negative favors DAP",
        "ready_replica_seconds": "DAP minus baseline; negative favors DAP",
    }
    for (profile, budget, baseline), pairs in sorted(comparisons.items()):
        for metric in PRIMARY_METRICS:
            deltas = [_number(dap, metric) - _number(other, metric) for dap, other in pairs]
            deltas = [value for value in deltas if math.isfinite(value)]
            low, high = _bootstrap(deltas)
            result.append({
                "profile": profile,
                "budget_seconds": float(budget),
                "baseline": baseline,
                "metric": metric,
                "direction": directions[metric],
                "paired_runs": len(deltas),
                "mean_delta": float(np.mean(deltas)) if deltas else math.nan,
                "median_delta": float(np.median(deltas)) if deltas else math.nan,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "dap_favorable_fraction": (
                    float(np.mean(np.asarray(deltas) > 0)) if metric == "completion_rate" and deltas
                    else float(np.mean(np.asarray(deltas) < 0)) if deltas else math.nan
                ),
            })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = paired_statistics(list(csv.DictReader(handle)))
    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (args.output / "paired_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} paired comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
