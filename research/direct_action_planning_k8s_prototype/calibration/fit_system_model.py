from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _manifests(root: Path, name: str) -> list[Path]:
    paths = sorted(root.glob(f"{name}/*/manifest.json"))
    return [
        path for path in paths
        if int(json.loads(path.read_text()).get("calibration_protocol_revision", 0)) >= 3
    ]


def _percentile(values: list[float], q: float, fallback: float) -> float:
    return float(np.quantile(values, q)) if values else fallback


def fit(root: Path) -> dict[str, Any]:
    capacity_rows = []
    for path in _manifests(root, "capacity"):
        capacity_rows.extend(json.loads(path.read_text())["rows"])
    startup_rows = []
    for path in _manifests(root, "startup"):
        startup_rows.extend(json.loads(path.read_text())["rows"])
    if not capacity_rows or not startup_rows:
        raise ValueError("both capacity and startup calibration artifacts are required")
    profiles: dict[str, dict[str, Any]] = {}
    for profile in sorted({str(row["profile"]) for row in capacity_rows}):
        selected = [row for row in capacity_rows if row["profile"] == profile]
        raw_capacities: dict[int, float] = {}
        p95: dict[str, float] = {}
        for replicas in sorted({int(row["replicas"]) for row in selected}):
            points = [row for row in selected if int(row["replicas"]) == replicas]
            # Highest steady application completion rate across independent
            # calibration loads estimates the usable processing capacity.
            raw_capacities[replicas] = max(float(row["application_completed_rps"]) for row in points)
            p95[str(replicas)] = float(np.median([float(row["application_p95_seconds"]) for row in points]))
        levels = sorted(raw_capacities)
        monotonic = np.maximum.accumulate([raw_capacities[level] for level in levels])
        capacities = {str(level): float(value) for level, value in zip(levels, monotonic)}
        delays = [float(row["all_ready_latency_seconds"]) for row in startup_rows if row["profile"] == profile]
        profiles[profile] = {
            "capacity_by_ready_replicas_rps": capacities,
            "raw_capacity_by_ready_replicas_rps": {str(key): value for key, value in raw_capacities.items()},
            "monotonic_capacity_projection_applied": bool(
                any(raw_capacities[left] > raw_capacities[right] for left, right in zip(levels, levels[1:]))
            ),
            "p95_latency_by_ready_replicas_seconds": p95,
            "startup_delay_seconds_median": _percentile(delays, 0.5, 3.0),
            "startup_delay_seconds_guard": _percentile(delays, 0.95, 8.0) + 1.0,
            "queue_max": max(float(row["application_queue_depth"]) for row in selected),
            "calibration_points": len(selected),
        }
    return {
        "schema": "dap.k8s.structured_system_model.v1",
        "source": "independent_real_kubernetes_calibration",
        "profiles": profiles,
        "reward": {
            "completion_weight": 1.0, "queue_penalty": 0.002,
            "latency_penalty": 0.1, "slo_penalty": 1.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = fit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
