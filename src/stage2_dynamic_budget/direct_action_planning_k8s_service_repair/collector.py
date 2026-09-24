"""Live observation adapter with training/runtime semantic parity."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .prototype_api import FieldEvidence, StateSnapshot


def _derived_field(template: FieldEvidence, value: float, policy: str) -> FieldEvidence:
    return FieldEvidence(
        raw=float(value), normalized=float(value), timestamp=template.timestamp,
        freshness_seconds=template.freshness_seconds, missing_policy=policy,
    )


class RuntimeSemanticCollector:
    """Replace legacy CPU-as-feature-6 and rolling-mean semantics explicitly."""

    def __init__(self, collector: Any):
        self.collector = collector
        self._recent_rate: float | None = None

    def collect(
        self, *, remaining_budget_ratio: float, remaining_horizon_ratio: float
    ) -> StateSnapshot:
        snapshot = self.collector.collect(
            remaining_budget_ratio=remaining_budget_ratio,
            remaining_horizon_ratio=remaining_horizon_ratio,
        )
        fields = dict(snapshot.fields)
        current_rate = float(fields["current_request_rate"].raw)
        self._recent_rate = (
            current_rate
            if self._recent_rate is None
            else 0.75 * self._recent_rate + 0.25 * current_rate
        )
        completed = float(fields["completed_request_rate"].raw)
        capacity = max(float(fields["active_capacity_rps"].raw), 1.0e-9)
        service_utilization = completed / capacity
        template = fields["current_request_rate"]
        fields["recent_request_rate"] = _derived_field(
            template, self._recent_rate,
            "EWMA recurrence: 0.75 previous + 0.25 current; matches branch model",
        )
        fields["service_utilization"] = _derived_field(
            template, service_utilization,
            "completed request-rate / calibrated current Ready capacity",
        )
        observation = list(snapshot.dap_observation)
        observation[1] = self._recent_rate
        observation[6] = service_utilization
        return replace(snapshot, fields=fields, dap_observation=tuple(observation))


def stale_system_with_current_safety(
    stale: StateSnapshot, current: StateSnapshot
) -> StateSnapshot:
    """Sample-and-hold learned state while preserving independent safety state.

    `fields["ready_pods"]` and observation components 0:12 intentionally remain
    stale because they belong to the learned/model state. Top-level Ready and
    desired counts, plus budget/horizon components, stay current and feed the
    hard feasibility channel and ledger.
    """

    fields = dict(stale.fields)
    for name in ("remaining_budget_ratio", "remaining_horizon_ratio"):
        fields[name] = current.fields[name]
    return replace(
        stale,
        collected_at=current.collected_at,
        monotonic_seconds=current.monotonic_seconds,
        collection_latency_seconds=current.collection_latency_seconds,
        pod_metric_latency_seconds=current.pod_metric_latency_seconds,
        desired_replicas=current.desired_replicas,
        ready_replicas=current.ready_replicas,
        missing_pods=current.missing_pods,
        fields=fields,
        dap_observation=tuple(stale.dap_observation[:12])
        + tuple(current.dap_observation[12:14]),
    )

