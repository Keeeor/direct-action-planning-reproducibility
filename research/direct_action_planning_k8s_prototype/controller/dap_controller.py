from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from .budget_tracker import BudgetTracker
from .checkpoint_loader import load_checkpoint
from .config import ControllerConfig, load_controller_config
from .kube_client import KubectlClient
from .planner import DirectActionPlanner
from .state_collector import StateCollector
from .system_model import StructuredSystemModel


def _append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


class DAPController:
    """External DAP controller issuing absolute Deployment replica targets."""

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.kube = KubectlClient(
            context=config.context, namespace=config.namespace, deployment=config.deployment
        )
        self.checkpoint = load_checkpoint(config.checkpoint_path)
        self.system_model = StructuredSystemModel.load(
            config.system_model_path, config.profile, slo_seconds=config.slo_seconds
        )
        self.collector = StateCollector(
            self.kube, capacity_per_pod_rps=config.capacity_per_pod_rps,
            capacity_by_ready_replicas=self.system_model.capacity_by_replicas,
            max_replicas=max(config.action_mapper.targets.values()),
        )
        self.tracker = BudgetTracker(config.total_budget_seconds, base_replicas=config.base_replicas)
        self.planner = DirectActionPlanner(
            checkpoint=self.checkpoint, system_model=self.system_model,
            mapper=config.action_mapper, control_interval_seconds=config.control_interval_seconds,
            horizon_steps=config.horizon_steps, total_budget_seconds=config.total_budget_seconds,
        )
        self.actions_path = config.result_directory / "controller_actions.jsonl"
        self.states_path = config.result_directory / "controller_states.jsonl"
        self.events_path = config.result_directory / "controller_events.jsonl"

    def _event(self, event: str, **values: Any) -> None:
        _append(
            self.events_path,
            {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **values},
        )

    def _return_to_base_and_account(self) -> None:
        evidence = self.kube.scale(self.config.base_replicas)
        self._event("safe_fallback_to_base", api_latency_seconds=evidence.latency_seconds)
        timeout = time.monotonic() + max(10.0, self.system_model.scale_down_guard_seconds * 3.0)
        while True:
            status = self.kube.deployment_status()
            self.tracker.update(
                ready_replicas=status["ready_replicas"], desired_replicas=status["desired_replicas"],
                monotonic_seconds=time.monotonic(), wall_time=datetime.now(timezone.utc).isoformat(),
            )
            if status["ready_replicas"] <= self.config.base_replicas:
                return
            if time.monotonic() >= timeout:
                self._event("scale_down_accounting_timeout", status=status)
                return
            time.sleep(0.25)

    def _account_until(self, deadline: float) -> None:
        """Integrate observed Ready replicas between control decisions."""
        last_ready: int | None = None
        while True:
            now = time.monotonic()
            if now >= deadline:
                return
            status = self.kube.deployment_status()
            sample = self.tracker.update(
                ready_replicas=status["ready_replicas"], desired_replicas=status["desired_replicas"],
                monotonic_seconds=now, wall_time=datetime.now(timezone.utc).isoformat(),
            )
            if last_ready is None or last_ready != status["ready_replicas"]:
                self._event(
                    "ready_replica_observed", ready_replicas=status["ready_replicas"],
                    desired_replicas=status["desired_replicas"], cumulative_ready_cost=sample.cumulative_ready_cost,
                )
                last_ready = status["ready_replicas"]
            time.sleep(min(0.25, max(deadline - time.monotonic(), 0.0)))

    def run(self, *, prepare: bool = True) -> dict[str, Any]:
        if self.kube.autoscaler_conflicts():
            raise RuntimeError(f"DAP cannot share deployment with autoscaler: {self.kube.autoscaler_conflicts()}")
        self.config.result_directory.mkdir(parents=True, exist_ok=False)
        if prepare:
            self.kube.set_profile(self.config.profile)
            self.kube.scale(self.config.base_replicas)
            self.kube.rollout_restart()
            self.kube.rollout_status()
        started = time.monotonic()
        self._event(
            "controller_started", checkpoint_sha256=self.checkpoint.sha256,
            total_budget_seconds=self.config.total_budget_seconds,
        )
        deadline_misses = 0
        try:
            for step in range(self.config.horizon_steps):
                cycle_started = time.monotonic()
                status = self.kube.deployment_status()
                budget_sample = self.tracker.update(
                    ready_replicas=status["ready_replicas"], desired_replicas=status["desired_replicas"],
                    monotonic_seconds=cycle_started, wall_time=datetime.now(timezone.utc).isoformat(),
                )
                remaining_horizon = self.config.horizon_steps - step
                snapshot = self.collector.collect(
                    remaining_budget_ratio=self.tracker.remaining / max(self.tracker.total_budget, 1.0),
                    remaining_horizon_ratio=remaining_horizon / self.config.horizon_steps,
                )
                _append(self.states_path, {"step": step, **snapshot.as_dict(), "budget": budget_sample.__dict__})
                raw_state = {name: field.raw for name, field in snapshot.fields.items()}
                decision_started = time.perf_counter()
                decision = self.planner.select(
                    observation=snapshot.dap_observation, state=raw_state, budget=self.tracker,
                    current_ready=snapshot.ready_replicas, remaining_horizon_steps=remaining_horizon,
                )
                inference_latency = time.perf_counter() - decision_started
                api = self.kube.scale(decision.target_replicas)
                pods = self.kube.worker_pods()
                action_row = {
                    "step": step, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": decision.action, "target_replicas": decision.target_replicas,
                    "q_values": decision.q_values, "feasible": decision.feasible,
                    "predicted_load_rps": decision.predicted_load_rps,
                    "branches": {
                        name: {
                            "reward": branch.reward, "expected_cost_seconds": branch.expected_cost_seconds,
                            "effective_replicas": branch.effective_replicas, **branch.details,
                        }
                        for name, branch in decision.branches.items()
                    },
                    "budget_remaining_seconds": self.tracker.remaining,
                    "budget_sample": budget_sample.__dict__,
                    "state_collection_latency_seconds": snapshot.collection_latency_seconds,
                    "dap_inference_latency_seconds": inference_latency,
                    "kubernetes_api_latency_seconds": api.latency_seconds,
                    "control_loop_latency_seconds": time.perf_counter() - cycle_started,
                    "pods": pods,
                }
                _append(self.actions_path, action_row)
                next_deadline = started + (step + 1) * self.config.control_interval_seconds
                remaining_sleep = next_deadline - time.monotonic()
                if remaining_sleep > 0:
                    self._account_until(next_deadline)
                else:
                    deadline_misses += 1
                    self._event("control_deadline_missed", step=step, lateness_seconds=-remaining_sleep)
            self._return_to_base_and_account()
            final = self.tracker.samples[-1]
            result = {
                "schema": "dap.k8s.controller_result.v1", "status": "completed",
                "steps": self.config.horizon_steps, "ready_cost_seconds": self.tracker.ready_cost,
                "requested_cost_seconds": self.tracker.requested_cost,
                "remaining_budget_seconds": self.tracker.remaining,
                "budget_violation_seconds": max(self.tracker.ready_cost - self.tracker.total_budget, 0.0),
                "deadline_misses": deadline_misses, "final_budget_sample": final.__dict__,
                "budget_ledger": self.tracker.as_records(),
            }
            (self.config.result_directory / "controller_result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            self._event("controller_completed", **result)
            return result
        except Exception as exc:
            self._event("controller_exception", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            # A controller fault must leave the service at its base footprint.
            try:
                if not self.tracker.samples or self.tracker.samples[-1].ready_replicas > self.config.base_replicas:
                    self._return_to_base_and_account()
            except Exception as cleanup_error:  # pragma: no cover - external API failure
                self._event("safe_fallback_failed", error=f"{type(cleanup_error).__name__}: {cleanup_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = DAPController(load_controller_config(args.config)).run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["budget_violation_seconds"] <= 1.0e-9 else 2


if __name__ == "__main__":
    raise SystemExit(main())
