from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_SLOS = {"azure_http": 0.25, "gentd_inference": 1.0}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _finite(values: Iterable[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else math.nan


def _metric_value(row: dict[str, Any], name: str) -> float | None:
    field = row.get("fields", {}).get(name)
    if isinstance(field, dict):
        try:
            return float(field["raw"])
        except (KeyError, TypeError, ValueError):
            return None
    try:
        return float(row.get(name))
    except (TypeError, ValueError):
        return None


def _timestamp(row: dict[str, Any]) -> float | None:
    value = row.get("collected_at") or row.get("timestamp")
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _area(rows: list[dict[str, Any]], field: str) -> tuple[float, float]:
    samples: list[tuple[float, float]] = []
    for row in rows:
        timestamp = _timestamp(row)
        value = _metric_value(row, field)
        if timestamp is not None and value is not None:
            samples.append((timestamp, value))
    samples.sort()
    if not samples:
        return math.nan, math.nan
    area = 0.0
    for (left_time, left_value), (right_time, right_value) in zip(samples, samples[1:]):
        area += max(right_time - left_time, 0.0) * (left_value + right_value) / 2.0
    return area, samples[-1][1]


def _action_summary(run: Path) -> tuple[int, float]:
    rows = _jsonl(run / "controller" / "controller_actions.jsonl")
    targets = [int(row["target_replicas"]) for row in rows if row.get("target_replicas") is not None]
    if not targets:
        return 0, math.nan
    changes = sum(left != right for left, right in zip(targets, targets[1:]))
    return changes, float(max(targets))


def aggregate_run(run: Path, *, slo_seconds: float | None = None) -> dict[str, Any]:
    """Reduce one raw bundle to one independent experimental observation."""

    manifest_path = run / "run_manifest.json"
    result_path = run / "result.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    result = _read_json(result_path) if result_path.exists() else {}
    profile = str(result.get("profile") or manifest.get("profile") or "unknown")
    slo = float(slo_seconds if slo_seconds is not None else manifest.get("slo_seconds", DEFAULT_SLOS.get(profile, 1.0)))
    replay = result.get("replay", {})
    controller = result.get("controller", {})
    requests = _jsonl(run / "requests.jsonl")
    latencies = _finite(row.get("client_latency_seconds") for row in requests)
    action_count, peak_action_replicas = _action_summary(run)
    live = _jsonl(run / "live_metrics.jsonl")
    states = live or _jsonl(run / "controller" / "controller_states.jsonl")
    queue_area, final_queue = _area(states, "queue_depth")
    _, final_ready = _area(states, "ready_pods")
    cpu_seconds = sum(
        value for value in (_metric_value(row, "cpu_seconds_delta") for row in live) if value is not None
    )
    monitor = result.get("monitor") or manifest.get("monitor") or {}
    samples = int(monitor.get("samples", 0))
    monitor_failures = int(monitor.get("failures", 0))
    measured_duration = float(replay.get("wall_seconds", math.nan))
    horizon = len(_jsonl(run / "controller" / "controller_actions.jsonl"))
    if not math.isfinite(measured_duration) or measured_duration <= 0:
        measured_duration = float(horizon) if horizon else math.nan
    ready_cost = float(controller.get("ready_cost_seconds", math.nan))
    requested_cost = float(controller.get("requested_cost_seconds", math.nan))
    mean_replicas = 1.0 + ready_cost / measured_duration if math.isfinite(ready_cost) and math.isfinite(measured_duration) and measured_duration > 0 else math.nan
    completed = int(replay.get("completed", sum(1 for row in requests if 200 <= int(row.get("http_status", 0)) < 300)))
    scheduled = int(replay.get("scheduled", len(requests)))
    failed = int(replay.get("failed", sum(1 for row in requests if not (200 <= int(row.get("http_status", 0)) < 300))))
    timed_out = int(replay.get("timed_out", sum(1 for row in requests if row.get("error") == "timeout")))
    completed_latency = [
        float(row["client_latency_seconds"])
        for row in requests
        if 200 <= int(row.get("http_status", 0)) < 300 and math.isfinite(float(row.get("client_latency_seconds", math.nan)))
    ]
    all_latency = latencies
    slo_violations = sum(value > slo for value in all_latency)
    status = str(result.get("status") or manifest.get("status") or "missing")
    return {
        "run_directory": str(run),
        "run_id": manifest.get("run_id", run.name),
        "matrix_row_id": manifest.get("matrix_row_id", ""),
        "run_status": status,
        "method": result.get("method", manifest.get("method")),
        "profile": profile,
        "budget_seconds": float(result.get("budget_seconds", manifest.get("budget_seconds", math.nan))),
        "seed": manifest.get("seed", manifest.get("plan_seed")),
        "plan_sha256": manifest.get("plan_sha256"),
        "total_requests": scheduled,
        "completed_requests": completed,
        "failed_requests": failed,
        "timeout_requests": timed_out,
        "completion_rate": completed / scheduled if scheduled else math.nan,
        "throughput_rps": completed / measured_duration if math.isfinite(measured_duration) and measured_duration > 0 else math.nan,
        "mean_latency_seconds": float(np.mean(completed_latency)) if completed_latency else math.nan,
        "p95_latency_seconds": _quantile(completed_latency, 0.95),
        "p99_latency_seconds": _quantile(completed_latency, 0.99),
        "slo_seconds": slo,
        "slo_violation_rate": slo_violations / len(all_latency) if all_latency else math.nan,
        "timeout_rate": timed_out / scheduled if scheduled else math.nan,
        "failure_rate": failed / scheduled if scheduled else math.nan,
        "queue_area": queue_area,
        "final_queue": final_queue,
        "ready_replica_seconds": ready_cost,
        "requested_replica_seconds": requested_cost,
        "estimated_cpu_seconds": cpu_seconds,
        "mean_replicas": mean_replicas,
        "peak_replicas": peak_action_replicas,
        "final_ready_replicas": final_ready,
        "scaling_count": action_count,
        "unused_budget_seconds": float(controller.get("remaining_budget_seconds", math.nan)),
        "budget_violation_seconds": float(controller.get("budget_violation_seconds", math.nan)),
        "controller_deadline_misses": int(controller.get("deadline_misses", 0)),
        "monitor_samples": samples,
        "monitor_failures": monitor_failures,
        "metric_missing_rate": monitor_failures / (samples + monitor_failures) if samples + monitor_failures else math.nan,
        "controller_inference_seconds": _mean_action_metric(run, "dap_inference_latency_seconds"),
        "controller_state_collection_seconds": _mean_action_metric(run, "state_collection_latency_seconds"),
        "controller_api_seconds": _mean_action_metric(run, "kubernetes_api_latency_seconds"),
        "controller_loop_seconds": _mean_action_metric(run, "control_loop_latency_seconds"),
    }


def _mean_action_metric(run: Path, field: str) -> float:
    values = _finite(row.get(field) for row in _jsonl(run / "controller" / "controller_actions.jsonl"))
    return float(np.mean(values)) if values else math.nan


def aggregate(results: Path, output: Path) -> list[dict[str, Any]]:
    rows = [aggregate_run(path.parent) for path in sorted(results.rglob("run_manifest.json"))]
    output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (output / "run_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "dap.k8s.aggregate.v1",
        "runs_discovered": len(rows),
        "completed_runs": sum(row["run_status"] == "completed" for row in rows),
        "failed_runs": sum(row["run_status"] != "completed" for row in rows),
    }
    (output / "aggregate_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = aggregate(args.results.resolve(), args.output.resolve())
    print(json.dumps({"runs": len(rows), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
