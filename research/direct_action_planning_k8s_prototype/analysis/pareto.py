from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
from typing import Any


METRICS = ("ready_replica_seconds", "completion_rate", "slo_violation_rate")


def _number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except ValueError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    cost_ok = left["ready_replica_seconds"] <= right["ready_replica_seconds"]
    service_ok = left["completion_rate"] >= right["completion_rate"]
    slo_ok = left["slo_violation_rate"] <= right["slo_violation_rate"]
    strict = (
        left["ready_replica_seconds"] < right["ready_replica_seconds"]
        or left["completion_rate"] > right["completion_rate"]
        or left["slo_violation_rate"] < right["slo_violation_rate"]
    )
    return cost_ok and service_ok and slo_ok and strict


def compute_pareto(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("run_status") == "completed":
            grouped[(row.get("profile", ""), row.get("method", ""), row.get("budget_seconds", ""))].append(row)
    points: list[dict[str, Any]] = []
    for (profile, method, budget), group in sorted(grouped.items()):
        point = {
            "profile": profile,
            "method": method,
            "budget_seconds": float(budget),
            "n_runs": len(group),
            **{metric: _mean([_number(row, metric) for row in group]) for metric in METRICS},
        }
        if all(math.isfinite(point[metric]) for metric in METRICS):
            points.append(point)
    for point in points:
        candidates = [candidate for candidate in points if candidate["profile"] == point["profile"]]
        point["pareto_nondominated"] = not any(_dominates(candidate, point) for candidate in candidates if candidate is not point)
        point["pareto_dominates_count"] = sum(
            _dominates(point, candidate) for candidate in candidates if candidate is not point
        )
    return points


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        points = compute_pareto(list(csv.DictReader(handle)))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "pareto_points.csv", points)
    print(f"wrote {len(points)} Pareto points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
