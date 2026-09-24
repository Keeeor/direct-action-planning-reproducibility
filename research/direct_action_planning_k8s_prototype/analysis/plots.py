from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"dap": "#0072B2", "hpa": "#D55E00", "keda": "#009E73", "threshold": "#CC79A7", "mpc_4": "#E69F00", "static": "#666666"}


def _float(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("run_status") == "completed"]


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def service_cost_frontier(rows: list[dict[str, str]], output: Path) -> None:
    profiles = sorted({row.get("profile", "") for row in rows})
    if not profiles:
        return
    fig, axes = plt.subplots(1, len(profiles), figsize=(5.2 * len(profiles), 3.8), squeeze=False)
    for axis, profile in zip(axes[0], profiles):
        for method in sorted({row["method"] for row in rows if row.get("profile") == profile}):
            points = [row for row in rows if row.get("profile") == profile and row.get("method") == method]
            x = [_float(row, "ready_replica_seconds") for row in points]
            y = [_float(row, "slo_violation_rate") for row in points]
            valid = [(left, right) for left, right in zip(x, y) if math.isfinite(left) and math.isfinite(right)]
            if valid:
                axis.scatter(*zip(*valid), label=method, s=28, color=COLORS.get(method, "#333333"), alpha=0.72)
        axis.set_title(profile.replace("_", " "))
        axis.set_xlabel("Ready-replica seconds")
        axis.set_ylabel("SLO violation rate")
        axis.grid(alpha=0.25)
    axes[0][0].legend(frameon=False, fontsize=8)
    _save(fig, output, "service_cost_frontier")


def controller_overhead(rows: list[dict[str, str]], output: Path) -> None:
    methods = [method for method in ("dap", "mpc_4", "threshold") if any(row.get("method") == method for row in rows)]
    if not methods:
        return
    means = []
    for method in methods:
        values = [_float(row, "controller_loop_seconds") * 1000.0 for row in rows if row.get("method") == method]
        values = [value for value in values if math.isfinite(value)]
        means.append(float(np.mean(values)) if values else math.nan)
    fig, axis = plt.subplots(figsize=(4.6, 3.5))
    axis.bar(methods, means, color=[COLORS.get(method, "#333333") for method in methods])
    axis.set_ylabel("Mean control-loop latency (ms)")
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output, "controller_overhead")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _relative_seconds(value: str, origin: float) -> float | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() - origin
    except (AttributeError, ValueError):
        return None


def representative_trajectory(rows: list[dict[str, str]], output: Path) -> None:
    candidates = [row for row in rows if row.get("method") == "dap" and row.get("run_directory")]
    if not candidates:
        return
    candidate = candidates[0]
    run = Path(candidate["run_directory"])
    request_rows = _jsonl(run / "requests.jsonl")
    metric_rows = _jsonl(run / "live_metrics.jsonl") or _jsonl(run / "controller" / "controller_states.jsonl")
    action_rows = _jsonl(run / "controller" / "controller_actions.jsonl")
    if not request_rows or not metric_rows:
        return
    figure, axes = plt.subplots(6, 1, figsize=(8.2, 9.6), sharex=True)
    bins: dict[int, int] = {}
    for row in request_rows:
        second = int(float(row.get("scheduled_offset_seconds", 0.0)))
        bins[second] = bins.get(second, 0) + 1
    axes[0].step(sorted(bins), [bins[key] for key in sorted(bins)], where="post", color="#333333")
    axes[0].set_ylabel("requests/s")
    timestamps = [row.get("collected_at") or row.get("timestamp") for row in metric_rows]
    valid_times = [datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() for value in timestamps if isinstance(value, str)]
    if not valid_times:
        return
    origin = min(valid_times)
    x = [_relative_seconds(row.get("collected_at") or row.get("timestamp"), origin) for row in metric_rows]
    pairs = [(time, row) for time, row in zip(x, metric_rows) if time is not None]
    x = [time for time, _ in pairs]
    def series(name: str) -> list[float]:
        values = []
        for _, row in pairs:
            field = row.get("fields", {}).get(name, {})
            values.append(float(field.get("raw", math.nan)) if isinstance(field, dict) else math.nan)
        return values
    axes[1].plot(x, series("queue_depth"), color="#D55E00")
    axes[1].set_ylabel("queue")
    axes[2].step(x, series("ready_pods"), where="post", color="#009E73")
    axes[2].set_ylabel("Ready Pods")
    axes[3].plot(x, series("p95_latency_seconds"), color="#CC79A7")
    axes[3].set_ylabel("P95 (s)")
    budget_x = []
    budget_y = []
    action_x = []
    action_y = []
    for index, row in enumerate(action_rows):
        timestamp = row.get("timestamp")
        time_value = _relative_seconds(timestamp, origin) if timestamp else float(index)
        if time_value is None:
            time_value = float(index)
        budget_x.append(time_value)
        budget_y.append(float(row.get("budget_remaining_seconds", math.nan)))
        action_x.append(time_value)
        action_y.append(float(row.get("target_replicas", math.nan)))
    axes[4].step(budget_x, budget_y, where="post", color="#0072B2")
    axes[4].set_ylabel("budget left")
    axes[5].step(action_x, action_y, where="post", color="#E69F00")
    axes[5].set_ylabel("target Pods")
    axes[5].set_xlabel("elapsed seconds")
    for axis in axes:
        axis.grid(alpha=0.22)
    _save(figure, output, "dap_representative_trajectory")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _load_csv(args.input / "run_metrics.csv")
    args.output.mkdir(parents=True, exist_ok=True)
    service_cost_frontier(rows, args.output)
    controller_overhead(rows, args.output)
    representative_trajectory(rows, args.output)
    print(f"rendered figures from {len(rows)} completed runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
