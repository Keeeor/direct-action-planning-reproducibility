"""Registered learned-observation perturbations with a current safety channel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .collector import stale_system_with_current_safety
from .prototype_api import StateSnapshot


OBSERVATION_CONDITIONS = ("observation_lag_1", "metric_dropout_10pct")


def dropout_steps(*, horizon: int, fraction: float, seed: int) -> frozenset[int]:
    if int(horizon) < 2:
        raise ValueError("dropout schedule needs at least two cycles")
    if not 0.0 <= float(fraction) < 1.0:
        raise ValueError("dropout fraction must lie in [0, 1)")
    count = min(int(round(int(horizon) * float(fraction))), int(horizon) - 1)
    selected = np.random.default_rng(int(seed)).choice(
        np.arange(1, int(horizon)), size=count, replace=False
    )
    return frozenset(int(value) for value in selected)


class PerturbedSemanticCollector:
    def __init__(
        self, collector: Any, *, condition: str, horizon: int, seed: int,
        event_path: Path | None,
    ):
        if condition not in OBSERVATION_CONDITIONS:
            raise ValueError(f"unknown observation perturbation: {condition}")
        self.collector = collector
        self.condition = condition
        self.event_path = event_path
        self.step = 0
        self.previous_actual: StateSnapshot | None = None
        self.last_delivered: StateSnapshot | None = None
        self.schedule = dropout_steps(
            horizon=int(horizon), fraction=0.10, seed=int(seed)
        )
        self.applied_events = 0

    def _record(
        self, *, applied: bool, actual: StateSnapshot, delivered: StateSnapshot
    ) -> None:
        if self.event_path is None:
            return
        row = {
            "schema": "dap.k8s.service_repair_perturbation_event.v1",
            "condition": self.condition,
            "step": self.step,
            "applied": bool(applied),
            "actual_system_observation": list(actual.dap_observation[:12]),
            "delivered_system_observation": list(delivered.dap_observation[:12]),
            "actual_ready_replicas": actual.ready_replicas,
            "safety_ready_replicas": delivered.ready_replicas,
            "model_ready_replicas": int(round(delivered.fields["ready_pods"].raw)),
            "current_budget_horizon": list(delivered.dap_observation[12:14]),
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
                delivered = stale_system_with_current_safety(
                    self.previous_actual, actual
                )
                applied = True
            self.previous_actual = actual
        elif self.step in self.schedule and self.last_delivered is not None:
            delivered = stale_system_with_current_safety(
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

