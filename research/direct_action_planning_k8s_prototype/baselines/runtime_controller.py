from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from controller.budget_tracker import BudgetTracker
from controller.kube_client import KubectlClient
from controller.state_collector import StateCollector
from controller.system_model import StructuredSystemModel

from .policies import ReplicaPolicy


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


class BaselineRuntimeController:
    def __init__(
        self, *, kube: KubectlClient, policy: ReplicaPolicy, system_model: StructuredSystemModel,
        profile: str, result_directory: Path, total_budget_seconds: float, horizon_steps: int,
        control_interval_seconds: float, capacity_per_pod_rps: float,
    ):
        self.kube = kube
        self.policy = policy
        self.model = system_model
        self.profile = profile
        self.result_directory = result_directory
        self.total_budget = float(total_budget_seconds)
        self.horizon = int(horizon_steps)
        self.interval = float(control_interval_seconds)
        self.tracker = BudgetTracker(self.total_budget)
        self.collector = StateCollector(
            kube,
            capacity_per_pod_rps=capacity_per_pod_rps,
            capacity_by_ready_replicas=system_model.capacity_by_replicas,
        )

    def _event(self, event: str, **values: Any) -> None:
        _append(self.result_directory / "controller_events.jsonl", {
            "event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **values,
        })

    def _account_until(self, deadline: float) -> None:
        while time.monotonic() < deadline:
            status = self.kube.deployment_status()
            self.tracker.update(
                ready_replicas=status["ready_replicas"], desired_replicas=status["desired_replicas"],
                monotonic_seconds=time.monotonic(), wall_time=datetime.now(timezone.utc).isoformat(),
            )
            time.sleep(min(0.25, max(deadline - time.monotonic(), 0.0)))

    def _base_and_account(self) -> None:
        self.kube.scale(1)
        deadline = time.monotonic() + max(10.0, self.model.scale_down_guard_seconds * 3)
        while True:
            status = self.kube.deployment_status()
            self.tracker.update(
                ready_replicas=status["ready_replicas"], desired_replicas=status["desired_replicas"],
                monotonic_seconds=time.monotonic(), wall_time=datetime.now(timezone.utc).isoformat(),
            )
            if status["ready_replicas"] <= 1 or time.monotonic() >= deadline:
                return
            time.sleep(0.25)

    def run(self, *, prepare: bool = True) -> dict[str, Any]:
        if self.kube.autoscaler_conflicts():
            raise RuntimeError("external baseline cannot share deployment with HPA/KEDA")
        self.result_directory.mkdir(parents=True, exist_ok=False)
        if prepare:
            self.kube.set_profile(self.profile)
            self.kube.scale(1)
            self.kube.rollout_restart()
            self.kube.rollout_status()
        started = time.monotonic()
        missed = 0
        try:
            for step in range(self.horizon):
                cycle_started = time.monotonic()
                status = self.kube.deployment_status()
                budget_sample = self.tracker.update(
                    ready_replicas=status["ready_replicas"], desired_replicas=status["desired_replicas"],
                    monotonic_seconds=cycle_started, wall_time=datetime.now(timezone.utc).isoformat(),
                )
                snapshot = self.collector.collect(
                    remaining_budget_ratio=self.tracker.remaining / max(self.total_budget, 1.0),
                    remaining_horizon_ratio=(self.horizon - step) / self.horizon,
                )
                _append(self.result_directory / "controller_states.jsonl", {
                    "step": step, "budget": budget_sample.__dict__, **snapshot.as_dict(),
                })
                decision = self.policy.select(
                    snapshot=snapshot, budget=self.tracker, remaining_horizon_steps=self.horizon - step
                )
                api = self.kube.scale(self.policy.mapper.replicas(decision.action)) if hasattr(self.policy, "mapper") else None
                _append(self.result_directory / "controller_actions.jsonl", {
                    "step": step, "action": decision.action,
                    "target_replicas": self.policy.mapper.replicas(decision.action) if hasattr(self.policy, "mapper") else 1,
                    "scores": decision.scores, "details": decision.details,
                    "state_collection_latency_seconds": snapshot.collection_latency_seconds,
                    "kubernetes_api_latency_seconds": None if api is None else api.latency_seconds,
                    "control_loop_latency_seconds": time.perf_counter() - cycle_started,
                    "budget_remaining_seconds": self.tracker.remaining,
                })
                deadline = started + (step + 1) * self.interval
                if deadline > time.monotonic():
                    self._account_until(deadline)
                else:
                    missed += 1
            self._base_and_account()
            result = {
                "schema": "dap.k8s.baseline_controller_result.v1", "status": "completed",
                "ready_cost_seconds": self.tracker.ready_cost,
                "requested_cost_seconds": self.tracker.requested_cost,
                "remaining_budget_seconds": self.tracker.remaining,
                "budget_violation_seconds": max(self.tracker.ready_cost - self.total_budget, 0.0),
                "deadline_misses": missed, "budget_ledger": self.tracker.as_records(),
            }
            (self.result_directory / "controller_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            return result
        finally:
            try:
                self.kube.scale(1)
            except Exception:
                pass
