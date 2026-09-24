"""Real Kubernetes controller for the audited service-repair checkpoint."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from .audit import verify_contract
from .collector import RuntimeSemanticCollector
from .planner import RuntimeConsistentPlanner
from .prototype_api import (
    BudgetTracker,
    ControllerConfig,
    KubectlClient,
    PROJECT_ROOT,
    StateCollector,
    StateSnapshot,
    load_checkpoint,
)
from .transition import RuntimeConsistentSystemModel


def _append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def decision_ready_inputs(snapshot: StateSnapshot) -> tuple[int, int]:
    """Return (learned/model Ready, independent current safety Ready)."""

    model_ready = max(int(round(snapshot.fields["ready_pods"].raw)), 1)
    safety_ready = max(int(snapshot.ready_replicas), 1)
    return model_ready, safety_ready


class ServiceRepairDAPController:
    """External DAP controller with runtime-consistent observation semantics."""

    def __init__(self, config: ControllerConfig, *, audit_contract: Path):
        self.config = config
        self.audit = verify_contract(
            project_root=PROJECT_ROOT,
            contract_path=Path(audit_contract),
            expected_config_path=None,
        )
        self.audit_contract_path = Path(audit_contract).resolve()
        self.kube = KubectlClient(
            context=config.context, namespace=config.namespace,
            deployment=config.deployment,
        )
        self.checkpoint = load_checkpoint(config.checkpoint_path)
        if self.checkpoint.metadata.get("schema") != "dap.k8s.service_repair_selection.v1":
            raise ValueError("runtime requires a service-repair checkpoint")
        if self.checkpoint.metadata.get("development_contract_sha256") != (
            "sha256:" + __import__("hashlib").sha256(self.audit_contract_path.read_bytes()).hexdigest()
        ):
            raise ValueError("checkpoint is not bound to the supplied development audit")
        self.system_model = RuntimeConsistentSystemModel.load(
            config.system_model_path, config.profile, slo_seconds=config.slo_seconds
        )
        base_collector = StateCollector(
            self.kube, capacity_per_pod_rps=config.capacity_per_pod_rps,
            capacity_by_ready_replicas=self.system_model.capacity_by_replicas,
            max_replicas=max(config.action_mapper.targets.values()),
        )
        self.collector: Any = RuntimeSemanticCollector(base_collector)
        self.tracker = BudgetTracker(
            config.total_budget_seconds, base_replicas=config.base_replicas
        )
        self.planner = RuntimeConsistentPlanner(
            checkpoint=self.checkpoint, system_model=self.system_model,
            mapper=config.action_mapper,
            control_interval_seconds=config.control_interval_seconds,
            horizon_steps=config.horizon_steps,
            total_budget_seconds=config.total_budget_seconds,
            tie_margin=float(self.checkpoint.metadata["tie_margin"]),
        )
        self.actions_path = config.result_directory / "controller_actions.jsonl"
        self.states_path = config.result_directory / "controller_states.jsonl"
        self.events_path = config.result_directory / "controller_events.jsonl"

    def _event(self, event: str, **values: Any) -> None:
        _append(self.events_path, {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **values,
        })

    def _account_until(self, deadline: float) -> None:
        last_ready: int | None = None
        while time.monotonic() < deadline:
            now = time.monotonic()
            status = self.kube.deployment_status()
            sample = self.tracker.update(
                ready_replicas=status["ready_replicas"],
                desired_replicas=status["desired_replicas"],
                monotonic_seconds=now,
                wall_time=datetime.now(timezone.utc).isoformat(),
            )
            if last_ready != status["ready_replicas"]:
                self._event(
                    "ready_replica_observed",
                    ready_replicas=status["ready_replicas"],
                    desired_replicas=status["desired_replicas"],
                    cumulative_ready_cost=sample.cumulative_ready_cost,
                )
                last_ready = status["ready_replicas"]
            time.sleep(min(0.25, max(deadline - time.monotonic(), 0.0)))

    def _return_to_base_and_account(self) -> None:
        evidence = self.kube.scale(self.config.base_replicas)
        self._event("safe_fallback_to_base", api_latency_seconds=evidence.latency_seconds)
        deadline = time.monotonic() + max(
            10.0, self.system_model.scale_down_guard_seconds * 3.0
        )
        while True:
            status = self.kube.deployment_status()
            self.tracker.update(
                ready_replicas=status["ready_replicas"],
                desired_replicas=status["desired_replicas"],
                monotonic_seconds=time.monotonic(),
                wall_time=datetime.now(timezone.utc).isoformat(),
            )
            if status["ready_replicas"] <= self.config.base_replicas:
                return
            if time.monotonic() >= deadline:
                self._event("scale_down_accounting_timeout", status=status)
                return
            time.sleep(0.25)

    def run(self, *, prepare: bool = True) -> dict[str, Any]:
        conflicts = self.kube.autoscaler_conflicts()
        if conflicts:
            raise RuntimeError(f"repaired DAP cannot share deployment with autoscaler: {conflicts}")
        self.config.result_directory.mkdir(parents=True, exist_ok=False)
        if prepare:
            self.kube.set_profile(self.config.profile)
            self.kube.scale(self.config.base_replicas)
            self.kube.rollout_restart()
            self.kube.rollout_status()
        started = time.monotonic()
        self._event(
            "controller_started",
            checkpoint_sha256=self.checkpoint.sha256,
            development_audit_sha256=(
                "sha256:" + __import__("hashlib").sha256(
                    self.audit_contract_path.read_bytes()
                ).hexdigest()
            ),
            total_budget_seconds=self.config.total_budget_seconds,
        )
        deadline_misses = 0
        current_target = self.config.base_replicas
        try:
            for step in range(self.config.horizon_steps):
                cycle_started = time.monotonic()
                status = self.kube.deployment_status()
                budget_sample = self.tracker.update(
                    ready_replicas=status["ready_replicas"],
                    desired_replicas=status["desired_replicas"],
                    monotonic_seconds=cycle_started,
                    wall_time=datetime.now(timezone.utc).isoformat(),
                )
                current_target = int(status["desired_replicas"])
                remaining_horizon = self.config.horizon_steps - step
                snapshot = self.collector.collect(
                    remaining_budget_ratio=(
                        self.tracker.remaining / max(self.tracker.total_budget, 1.0)
                    ),
                    remaining_horizon_ratio=(
                        remaining_horizon / self.config.horizon_steps
                    ),
                )
                model_ready, safety_ready = decision_ready_inputs(snapshot)
                _append(self.states_path, {
                    "step": step,
                    **snapshot.as_dict(),
                    "budget": budget_sample.__dict__,
                    "model_ready_replicas": model_ready,
                    "safety_ready_replicas": safety_ready,
                })
                decision_started = time.perf_counter()
                decision = self.planner.select(
                    observation=snapshot.dap_observation,
                    model_ready=model_ready,
                    safety_ready=safety_ready,
                    remaining_budget_seconds=self.tracker.remaining,
                    remaining_horizon_steps=remaining_horizon,
                    current_target_replicas=current_target,
                )
                inference_latency = time.perf_counter() - decision_started
                api = self.kube.scale(decision.target_replicas)
                _append(self.actions_path, {
                    "step": step,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": decision.action,
                    "greedy_action": decision.greedy_action,
                    "target_replicas": decision.target_replicas,
                    "q_values": decision.q_values,
                    "feasible": decision.feasible,
                    "predicted_load_rps": decision.predicted_load_rps,
                    "model_ready_replicas": decision.model_ready_replicas,
                    "hard_mask_current_ready_replicas": decision.safety_ready_replicas,
                    "tie_retained_current_target": decision.tie_retained_current_target,
                    "branches": {
                        name: {
                            "reward": branch.reward,
                            "expected_cost_seconds": branch.expected_cost_seconds,
                            "effective_replicas": branch.effective_replicas,
                            "next_ready_replicas": branch.next_ready_replicas,
                            **branch.details,
                        }
                        for name, branch in decision.branches.items()
                    },
                    "budget_remaining_seconds": self.tracker.remaining,
                    "budget_sample": budget_sample.__dict__,
                    "state_collection_latency_seconds": snapshot.collection_latency_seconds,
                    "dap_inference_latency_seconds": inference_latency,
                    "kubernetes_api_latency_seconds": api.latency_seconds,
                    "control_loop_latency_seconds": time.perf_counter() - cycle_started,
                    "pods": self.kube.worker_pods(),
                })
                next_deadline = started + (
                    step + 1
                ) * self.config.control_interval_seconds
                if next_deadline > time.monotonic():
                    self._account_until(next_deadline)
                else:
                    deadline_misses += 1
                    self._event(
                        "control_deadline_missed", step=step,
                        lateness_seconds=time.monotonic() - next_deadline,
                    )
            self._return_to_base_and_account()
            result = {
                "schema": "dap.k8s.service_repair_controller_result.v1",
                "status": "completed",
                "steps": self.config.horizon_steps,
                "ready_cost_seconds": self.tracker.ready_cost,
                "requested_cost_seconds": self.tracker.requested_cost,
                "remaining_budget_seconds": self.tracker.remaining,
                "budget_violation_seconds": max(
                    self.tracker.ready_cost - self.tracker.total_budget, 0.0
                ),
                "deadline_misses": deadline_misses,
                "budget_ledger": self.tracker.as_records(),
            }
            (self.config.result_directory / "controller_result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._event("controller_completed", **result)
            return result
        except Exception as exc:
            self._event("controller_exception", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            try:
                if (
                    not self.tracker.samples
                    or self.tracker.samples[-1].ready_replicas > self.config.base_replicas
                ):
                    self._return_to_base_and_account()
            except Exception as cleanup_error:  # pragma: no cover - external failure
                self._event(
                    "safe_fallback_failed",
                    error=f"{type(cleanup_error).__name__}: {cleanup_error}",
                )

