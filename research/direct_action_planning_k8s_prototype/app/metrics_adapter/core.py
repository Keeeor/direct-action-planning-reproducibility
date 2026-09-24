from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any


_PROCESS_METRIC = re.compile(
    r"^(process_cpu_seconds_total|process_resident_memory_bytes)(?:\{[^}]*\})?\s+([^\s]+)"
)


@dataclass(frozen=True)
class ProcessSample:
    pod: str
    observed_seconds: float
    cpu_seconds: float
    memory_bytes: float


@dataclass(frozen=True)
class PodUsage:
    pod: str
    observed_seconds: float
    window_seconds: float
    cpu_cores: float
    memory_bytes: float


def parse_process_metrics(payload: str) -> tuple[float, float]:
    """Extract the real Python worker process CPU and resident memory values."""
    values: dict[str, float] = {}
    for line in payload.splitlines():
        match = _PROCESS_METRIC.match(line)
        if match is None:
            continue
        try:
            values[match.group(1)] = float(match.group(2))
        except ValueError:
            continue
    if "process_cpu_seconds_total" not in values:
        raise ValueError("worker metrics lack process_cpu_seconds_total")
    return values["process_cpu_seconds_total"], values.get("process_resident_memory_bytes", 0.0)


class UsageCache:
    """Convert cumulative process CPU time into a Kubernetes PodMetrics usage rate."""

    def __init__(self) -> None:
        self._previous: dict[str, ProcessSample] = {}
        self._usage: dict[str, PodUsage] = {}

    def record(self, sample: ProcessSample) -> PodUsage:
        previous = self._previous.get(sample.pod)
        if previous is None:
            usage = PodUsage(
                pod=sample.pod,
                observed_seconds=sample.observed_seconds,
                window_seconds=0.0,
                cpu_cores=0.0,
                memory_bytes=max(sample.memory_bytes, 0.0),
            )
        else:
            window = max(sample.observed_seconds - previous.observed_seconds, 1.0e-6)
            cpu_delta = max(sample.cpu_seconds - previous.cpu_seconds, 0.0)
            usage = PodUsage(
                pod=sample.pod,
                observed_seconds=sample.observed_seconds,
                window_seconds=window,
                cpu_cores=cpu_delta / window,
                memory_bytes=max(sample.memory_bytes, 0.0),
            )
        self._previous[sample.pod] = sample
        self._usage[sample.pod] = usage
        return usage

    def remove_except(self, pods: set[str]) -> None:
        for pod in tuple(self._previous):
            if pod not in pods:
                self._previous.pop(pod, None)
                self._usage.pop(pod, None)

    def usages(self) -> list[PodUsage]:
        return [self._usage[name] for name in sorted(self._usage)]


def _rfc3339(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def pod_metrics_list(namespace: str, usages: list[PodUsage]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for usage in usages:
        cpu_nanocores = max(0, int(round(usage.cpu_cores * 1_000_000_000)))
        memory_bytes = max(0, int(round(usage.memory_bytes)))
        items.append(
            {
                "metadata": {"name": usage.pod, "namespace": namespace},
                "timestamp": _rfc3339(usage.observed_seconds),
                "window": f"{max(1, int(round(usage.window_seconds)))}s",
                "containers": [
                    {
                        "name": "worker",
                        "usage": {"cpu": f"{cpu_nanocores}n", "memory": str(memory_bytes)},
                    }
                ],
            }
        )
    return {
        "kind": "PodMetricsList",
        "apiVersion": "metrics.k8s.io/v1beta1",
        "metadata": {},
        "items": items,
    }
