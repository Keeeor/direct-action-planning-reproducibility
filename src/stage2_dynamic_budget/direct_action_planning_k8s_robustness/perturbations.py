"""Controller-visible observation perturbations with unperturbed budget accounting."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from .prototype_api import StateSnapshot


OBSERVATION_CONDITIONS = ("observation_lag_1", "metric_dropout_10pct")


def dropout_steps(*, horizon: int, fraction: float, seed: int) -> frozenset[int]:
    if int(horizon) < 2:
        raise ValueError("dropout schedule needs at least two control steps")
    if not 0.0 <= float(fraction) < 1.0:
        raise ValueError("dropout fraction must lie in [0, 1)")
    count = int(round(int(horizon) * float(fraction)))
    count = min(count, int(horizon) - 1)
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(np.arange(1, int(horizon)), size=count, replace=False)
    return frozenset(int(value) for value in selected)


def _current_budget_after_stale_system(
    stale: StateSnapshot, current: StateSnapshot
) -> StateSnapshot:
    fields = dict(stale.fields)
    for name in ("remaining_budget_ratio", "remaining_horizon_ratio"):
        fields[name] = current.fields[name]
    return replace(
        stale,
        collected_at=current.collected_at,
        monotonic_seconds=current.monotonic_seconds,
        collection_latency_seconds=current.collection_latency_seconds,
        pod_metric_latency_seconds=current.pod_metric_latency_seconds,
        missing_pods=current.missing_pods,
        fields=fields,
        dap_observation=tuple(stale.dap_observation[:12])
        + tuple(current.dap_observation[12:14]),
    )


class PerturbedCollector:
    """Wrap a real collector and perturb only the observation seen by DAP.

    The wrapped collector still samples the live system every cycle.  Remaining
    budget and horizon are always current, so the perturbation never weakens the
    independent Ready-Pod budget ledger or the hard action-feasibility check.
    """

    def __init__(
        self,
        collector: Any,
        *,
        condition: str,
        horizon: int,
        seed: int,
        event_path: Path | None,
        forced_dropout_steps: set[int] | frozenset[int] | None = None,
    ):
        if condition not in OBSERVATION_CONDITIONS:
            raise ValueError(f"unknown observation perturbation: {condition}")
        self.collector = collector
        self.condition = condition
        self.event_path = event_path
        self.step = 0
        self.previous_actual: StateSnapshot | None = None
        self.last_delivered: StateSnapshot | None = None
        self.schedule = (
            frozenset(forced_dropout_steps)
            if forced_dropout_steps is not None
            else dropout_steps(horizon=int(horizon), fraction=0.10, seed=int(seed))
        )
        self.applied_events = 0

    def _record(
        self, *, applied: bool, actual: StateSnapshot, delivered: StateSnapshot
    ) -> None:
        if self.event_path is None:
            return
        row = {
            "schema": "dap.k8s.perturbation_event.v1",
            "condition": self.condition,
            "step": self.step,
            "applied": bool(applied),
            "actual_collected_at": actual.collected_at,
            "delivered_system_observation": list(delivered.dap_observation[:12]),
            "actual_system_observation": list(actual.dap_observation[:12]),
            "current_budget_horizon": list(delivered.dap_observation[12:14]),
            "actual_ready_replicas": actual.ready_replicas,
            "delivered_ready_replicas": delivered.ready_replicas,
        }
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def collect(
        self, *, remaining_budget_ratio: float, remaining_horizon_ratio: float
    ) -> StateSnapshot:
        actual = self.collector.collect(
            remaining_budget_ratio=remaining_budget_ratio,
            remaining_horizon_ratio=remaining_horizon_ratio,
        )
        applied = False
        if self.condition == "observation_lag_1":
            if self.previous_actual is None:
                delivered = actual
            else:
                delivered = _current_budget_after_stale_system(
                    self.previous_actual, actual
                )
                applied = True
            self.previous_actual = actual
        else:
            if self.step in self.schedule and self.last_delivered is not None:
                delivered = _current_budget_after_stale_system(
                    self.last_delivered, actual
                )
                applied = True
            else:
                delivered = actual
                self.last_delivered = actual
        if applied:
            self.applied_events += 1
        self._record(applied=applied, actual=actual, delivered=delivered)
        self.step += 1
        return delivered
