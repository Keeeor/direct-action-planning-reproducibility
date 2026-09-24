from __future__ import annotations

"""Independent live-metric recorder for one real Kubernetes trial.

The controller's state samples are decision-time observations. This monitor is
separate so the analysis bundle also contains a denser, controller-independent
record of live application and Kubernetes measurements.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any

from controller.kube_client import KubectlClient
from controller.state_collector import StateCollector
from controller.system_model import StructuredSystemModel


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


@dataclass(frozen=True)
class MonitorSummary:
    samples: int
    failures: int
    missing_pod_observations: int
    estimated_cpu_seconds: float
    started_at: str
    ended_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "failures": self.failures,
            "missing_pod_observations": self.missing_pod_observations,
            "estimated_cpu_seconds": self.estimated_cpu_seconds,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


class LiveMetricMonitor:
    """Poll live worker metrics without providing information to a controller."""

    def __init__(
        self,
        *,
        kube: KubectlClient,
        system_model: StructuredSystemModel,
        capacity_per_pod_rps: float,
        output: Path,
        interval_seconds: float = 1.0,
    ):
        if interval_seconds <= 0:
            raise ValueError("monitor interval must be positive")
        self.kube = kube
        self.output = output
        self.interval_seconds = float(interval_seconds)
        self.collector = StateCollector(
            kube,
            capacity_per_pod_rps=capacity_per_pod_rps,
            capacity_by_ready_replicas=system_model.capacity_by_replicas,
        )

    def run(self, stop: threading.Event) -> MonitorSummary:
        started_at = datetime.now(timezone.utc).isoformat()
        samples = failures = missing = 0
        cpu_seconds = 0.0
        next_tick = time.monotonic()
        while not stop.is_set():
            next_tick += self.interval_seconds
            try:
                snapshot = self.collector.collect(
                    remaining_budget_ratio=0.0,
                    remaining_horizon_ratio=0.0,
                )
                pods = self.kube.worker_pods()
                missing += len(snapshot.missing_pods)
                cpu_seconds += snapshot.fields["cpu_seconds_delta"].raw
                _append(self.output, {
                    "schema": "dap.k8s.live_metric_sample.v1",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "pods": pods,
                    **snapshot.as_dict(),
                })
                samples += 1
            except Exception as exc:  # pragma: no cover - live infrastructure failure path
                failures += 1
                _append(self.output, {
                    "schema": "dap.k8s.live_metric_failure.v1",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                })
            stop.wait(max(0.0, next_tick - time.monotonic()))
        return MonitorSummary(
            samples=samples,
            failures=failures,
            missing_pod_observations=missing,
            estimated_cpu_seconds=cpu_seconds,
            started_at=started_at,
            ended_at=datetime.now(timezone.utc).isoformat(),
        )
