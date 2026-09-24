from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import re
import time
from typing import Any, Iterable, Mapping

import numpy as np

from .kube_client import KubectlClient


_SAMPLE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+'
    r'(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Inf|NaN)(?:\s+\d+)?$'
)
_LABEL = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:\\.|[^"\\])*)"')


@dataclass(frozen=True)
class PrometheusSample:
    name: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class FieldEvidence:
    raw: float
    normalized: float
    timestamp: str
    freshness_seconds: float
    missing_policy: str


@dataclass(frozen=True)
class StateSnapshot:
    collected_at: str
    monotonic_seconds: float
    collection_latency_seconds: float
    pod_metric_latency_seconds: float
    desired_replicas: int
    ready_replicas: int
    missing_pods: tuple[str, ...]
    fields: dict[str, FieldEvidence]
    dap_observation: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dap_observation"] = list(self.dap_observation)
        return data


def parse_prometheus_text(text: str) -> list[PrometheusSample]:
    samples: list[PrometheusSample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        labels: dict[str, str] = {}
        for label in _LABEL.finditer(match.group("labels") or ""):
            value = bytes(label.group("value"), "utf-8").decode("unicode_escape")
            labels[label.group("key")] = value
        samples.append(
            PrometheusSample(
                name=match.group("name"), labels=labels, value=float(match.group("value"))
            )
        )
    return samples


def histogram_quantile(cumulative_buckets: dict[float, float], quantile: float) -> float:
    if not cumulative_buckets:
        return 0.0
    ordered = sorted((bound, max(count, 0.0)) for bound, count in cumulative_buckets.items())
    total = ordered[-1][1]
    if total <= 0:
        return 0.0
    target = min(max(float(quantile), 0.0), 1.0) * total
    previous_bound = 0.0
    previous_count = 0.0
    for bound, count in ordered:
        if count >= target:
            if math.isinf(bound):
                return previous_bound
            width = max(bound - previous_bound, 0.0)
            bucket_count = max(count - previous_count, 0.0)
            fraction = 0.0 if bucket_count <= 0 else (target - previous_count) / bucket_count
            return previous_bound + min(max(fraction, 0.0), 1.0) * width
        previous_bound, previous_count = bound, count
    return previous_bound


class StateCollector:
    """Collect only live Kubernetes/application measurements and retain provenance."""

    COUNTERS = (
        "requests_received_total",
        "requests_completed_total",
        "requests_failed_total",
        "request_latency_seconds_count",
        "request_latency_seconds_sum",
        "request_slo_violations_total",
        "process_cpu_seconds_total",
    )

    def __init__(
        self,
        kube: KubectlClient,
        *,
        capacity_per_pod_rps: float,
        capacity_by_ready_replicas: Mapping[int, float] | None = None,
        rolling_window: int = 8,
        max_replicas: int = 5,
        normalization: dict[str, tuple[float, float]] | None = None,
    ):
        self.kube = kube
        self.capacity_per_pod_rps = float(capacity_per_pod_rps)
        if self.capacity_per_pod_rps <= 0:
            raise ValueError("capacity_per_pod_rps must be positive")
        self.capacity_by_ready_replicas = {
            int(key): float(value) for key, value in (capacity_by_ready_replicas or {}).items()
        }
        if self.capacity_by_ready_replicas and (
            1 not in self.capacity_by_ready_replicas
            or any(value <= 0 for value in self.capacity_by_ready_replicas.values())
        ):
            raise ValueError("calibrated capacity must include positive one-Pod capacity")
        self.max_replicas = int(max_replicas)
        self.normalization = normalization or {}
        self._previous: dict[str, dict[tuple[str, tuple[tuple[str, str], ...]], float]] = {}
        self._last_monotonic: float | None = None
        self._previous_queue = 0.0
        self._rates: deque[float] = deque(maxlen=rolling_window)
        self._latencies: deque[float] = deque(maxlen=rolling_window)
        self._tails: deque[float] = deque(maxlen=rolling_window)
        self._slo: deque[float] = deque(maxlen=rolling_window)

    def capacity_for_ready_replicas(self, ready_replicas: int) -> float:
        """Return the calibrated capacity observable at the current Ready level.

        The structured action model is fitted on a potentially non-linear
        replicas-to-capacity curve. Reusing that curve for online state
        construction prevents a hidden state-definition mismatch between the
        planner's counterfactual branches and its live observations.
        """

        ready = max(int(ready_replicas), 1)
        if not self.capacity_by_ready_replicas:
            return float(ready * self.capacity_per_pod_rps)
        keys = np.asarray(sorted(self.capacity_by_ready_replicas), dtype=np.float64)
        values = np.asarray(
            [self.capacity_by_ready_replicas[int(key)] for key in keys], dtype=np.float64
        )
        return float(np.interp(float(ready), keys, values))

    def _normalize(self, name: str, value: float) -> float:
        center, scale = self.normalization.get(name, (0.0, 1.0))
        return float(np.clip((value - center) / max(scale, 1.0e-9), -10.0, 10.0))

    @staticmethod
    def _index(samples: Iterable[PrometheusSample]) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
        indexed: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        for sample in samples:
            key = (sample.name, tuple(sorted(sample.labels.items())))
            indexed[key] = sample.value
        return indexed

    @staticmethod
    def _sum_metric(index: dict, name: str) -> float:
        return float(sum(value for (metric, _), value in index.items() if metric == name))

    def collect(
        self,
        *,
        remaining_budget_ratio: float,
        remaining_horizon_ratio: float,
    ) -> StateSnapshot:
        started = time.perf_counter()
        now = started
        timestamp = datetime.now(timezone.utc).isoformat()
        status = self.kube.deployment_status()
        pods = self.kube.worker_pods()
        active_names = {pod["name"] for pod in pods if pod["ready"] and not pod["deleting"]}
        missing: list[str] = []
        current_by_pod: dict[str, dict] = {}
        metric_latency = 0.0
        for pod in pods:
            if not pod["ready"] or pod["deleting"]:
                continue
            evidence = None
            for attempt in range(3):
                try:
                    evidence = self.kube.pod_metrics(pod["name"])
                    metric_latency += evidence.latency_seconds
                    current_by_pod[pod["name"]] = self._index(parse_prometheus_text(evidence.stdout))
                    break
                except RuntimeError:
                    if attempt < 2:
                        time.sleep(0.05 * (attempt + 1))
            if evidence is None:
                missing.append(pod["name"])

        interval = 0.0 if self._last_monotonic is None else max(now - self._last_monotonic, 1.0e-9)
        deltas: dict[str, float] = {name: 0.0 for name in self.COUNTERS}
        bucket_deltas: dict[float, float] = {}
        queue = inflight = 0.0
        for pod, current in current_by_pod.items():
            previous = self._previous.get(pod)
            queue += self._sum_metric(current, "queue_depth")
            inflight += self._sum_metric(current, "inflight_requests")
            if previous is None:
                continue
            for key, value in current.items():
                name, labels = key
                increment = max(value - previous.get(key, value), 0.0)
                if name in deltas:
                    deltas[name] += increment
                elif name == "request_latency_seconds_bucket":
                    le = dict(labels).get("le")
                    if le is not None:
                        bound = math.inf if le == "+Inf" else float(le)
                        bucket_deltas[bound] = bucket_deltas.get(bound, 0.0) + increment

        self._previous = {pod: values for pod, values in current_by_pod.items() if pod in active_names}
        self._last_monotonic = now
        arrival_rate = deltas["requests_received_total"] / interval if interval else 0.0
        completed_rate = deltas["requests_completed_total"] / interval if interval else 0.0
        latency_count = deltas["request_latency_seconds_count"]
        mean_latency = deltas["request_latency_seconds_sum"] / max(latency_count, 1.0)
        p95_latency = histogram_quantile(bucket_deltas, 0.95)
        slo_rate = deltas["request_slo_violations_total"] / max(latency_count, 1.0)
        cpu_utilization = deltas["process_cpu_seconds_total"] / max(
            interval * max(status["ready_replicas"], 1), 1.0e-9
        )
        busy_pod_ratio = min(inflight / max(status["ready_replicas"], 1), 1.0)
        queue_growth = queue - self._previous_queue
        self._previous_queue = queue
        if interval:
            self._rates.append(arrival_rate)
            self._latencies.append(mean_latency)
            self._tails.append(p95_latency)
            self._slo.append(slo_rate)
        recent_rate = float(np.mean(self._rates)) if self._rates else arrival_rate
        recent_latency = float(np.mean(self._latencies)) if self._latencies else mean_latency
        recent_tail = float(np.mean(self._tails)) if self._tails else p95_latency
        recent_slo = float(np.mean(self._slo)) if self._slo else slo_rate
        capacity = self.capacity_for_ready_replicas(status["ready_replicas"])
        burst = arrival_rate / max(recent_rate, 1.0e-9) if recent_rate > 0 else 0.0
        pressure = (arrival_rate + queue / max(interval, 1.0)) / max(capacity, 1.0e-9)
        values = {
            "current_request_rate": arrival_rate,
            "recent_request_rate": recent_rate,
            "request_rate_change": arrival_rate - (self._rates[-2] if len(self._rates) > 1 else arrival_rate),
            "queue_depth": queue,
            "queue_growth_rate": queue_growth / max(interval, 1.0),
            "ready_pods": float(status["ready_replicas"]),
            "busy_pod_ratio": busy_pod_ratio,
            "cpu_utilization": cpu_utilization,
            "mean_latency_seconds": recent_latency,
            "p95_latency_seconds": recent_tail,
            "slo_violation_rate": recent_slo,
            "burst_intensity": burst,
            "queue_to_capacity": pressure,
            "remaining_budget_ratio": float(np.clip(remaining_budget_ratio, 0.0, 1.0)),
            "remaining_horizon_ratio": float(np.clip(remaining_horizon_ratio, 0.0, 1.0)),
            "completed_request_rate": completed_rate,
            "failure_rate": deltas["requests_failed_total"] / max(deltas["requests_received_total"], 1.0),
            "active_capacity_rps": capacity,
            "cpu_seconds_delta": deltas["process_cpu_seconds_total"],
            "missing_pod_count": float(len(missing)),
        }
        age = time.perf_counter() - now
        policy = "counter_delta; zero on first/new-pod sample" if interval else "initial sample set to zero"
        fields = {
            name: FieldEvidence(
                raw=float(value), normalized=self._normalize(name, float(value)),
                timestamp=timestamp, freshness_seconds=age,
                missing_policy=policy if name not in {"queue_depth", "ready_pods"} else "live gauge; missing pod omitted",
            )
            for name, value in values.items()
        }
        observation = (
            arrival_rate, recent_rate, queue, queue_growth, capacity,
            max(status["ready_replicas"] - 1, 0) / max(self.max_replicas - 1, 1),
            cpu_utilization, recent_latency, recent_tail, recent_slo,
            burst, pressure, values["remaining_budget_ratio"], values["remaining_horizon_ratio"],
        )
        return StateSnapshot(
            collected_at=timestamp, monotonic_seconds=now,
            collection_latency_seconds=time.perf_counter() - started,
            pod_metric_latency_seconds=metric_latency,
            desired_replicas=status["desired_replicas"], ready_replicas=status["ready_replicas"],
            missing_pods=tuple(missing), fields=fields,
            dap_observation=tuple(float(value) for value in observation),
        )
