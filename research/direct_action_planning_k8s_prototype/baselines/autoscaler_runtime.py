from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import yaml

from controller.budget_tracker import BudgetTracker
from controller.kube_client import KubectlClient
from controller.state_collector import StateCollector
from controller.system_model import StructuredSystemModel


ROOT = Path(__file__).resolve().parents[1]


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def native_budget_cap(
    *,
    remaining_budget_seconds: float,
    current_ready_replicas: int,
    min_replicas: int,
    max_replicas: int,
    control_interval_seconds: float,
    native_scale_down_reserve_seconds: float,
) -> int:
    """Largest native-autoscaler target that is safe before it can scale.

    A native controller does not immediately obey an upper-bound patch. The
    feasibility calculation therefore includes one control interval at the
    candidate target and a control-plane scale-down reserve at the larger of
    the observed and candidate Ready footprints.
    """

    if min_replicas < 1 or max_replicas < min_replicas:
        raise ValueError("invalid native replica bounds")
    current_extra = max(int(current_ready_replicas) - min_replicas, 0)
    for target in range(int(max_replicas), int(min_replicas) - 1, -1):
        target_extra = max(target - min_replicas, 0)
        commitment = (
            target_extra * max(float(control_interval_seconds), 0.0)
            + max(current_extra, target_extra) * max(float(native_scale_down_reserve_seconds), 0.0)
        )
        if commitment <= float(remaining_budget_seconds) + 1.0e-9:
            return target
    return int(min_replicas)


class NativeAutoscalerRuntime:
    """Run one native autoscaler with an external hard-budget governor.

    The governor never writes Deployment replicas while HPA/KEDA owns that
    scale target. It only reduces the autoscaler's declared upper bound when
    the measured Ready-replica ledger no longer permits another interval.
    """

    SPECS = {
        "hpa": {
            "manifest": ROOT / "kubernetes" / "hpa" / "hpa.yaml",
            "resource": "hpa", "name": "dap-worker-hpa", "max_key": "maxReplicas",
        },
        "keda": {
            "manifest": ROOT / "kubernetes" / "keda" / "scaledobject.yaml",
            "resource": "scaledobjects.keda.sh", "name": "dap-worker-keda", "max_key": "maxReplicaCount",
        },
    }

    def __init__(
        self, *, kind: str, kube: KubectlClient, system_model: StructuredSystemModel,
        result_directory: Path, total_budget_seconds: float, horizon_steps: int,
        control_interval_seconds: float, capacity_per_pod_rps: float,
        native_scale_down_reserve_seconds: float | None = None,
    ):
        if kind not in self.SPECS:
            raise ValueError(kind)
        self.kind = kind
        self.spec = self.SPECS[kind]
        self.kube = kube
        self.model = system_model
        self.result_directory = result_directory
        self.total_budget = float(total_budget_seconds)
        self.horizon = int(horizon_steps)
        self.interval = float(control_interval_seconds)
        self.native_scale_down_reserve_seconds = max(
            float(native_scale_down_reserve_seconds or 0.0),
            float(system_model.scale_down_guard_seconds),
        )
        self.tracker = BudgetTracker(self.total_budget)
        self.min_replicas = 1
        self.max_replicas = 5
        self.applied_max_replicas: int | None = None
        self.runtime_manifest_path: Path | None = None
        self.collector = StateCollector(
            kube,
            capacity_per_pod_rps=capacity_per_pod_rps,
            capacity_by_ready_replicas=system_model.capacity_by_replicas,
        )

    def _validate_prerequisites(self) -> None:
        if self.kind == "hpa" and not self.kube.api_available("metrics.k8s.io/v1beta1"):
            raise RuntimeError("Kubernetes Metrics API is unavailable; native HPA cannot be evaluated")
        if self.kind == "keda":
            crd = self.kube.run(["get", "scaledobjects.keda.sh", "-o", "json"], check=False)
            if crd.returncode != 0:
                raise RuntimeError("KEDA ScaledObject CRD is unavailable; native KEDA cannot be evaluated")
            probe = self.kube.run(["get", "service", "prometheus"], check=False)
            if probe.returncode != 0:
                raise RuntimeError("Prometheus service is unavailable; configured KEDA trigger cannot be evaluated")

    def _account(self) -> None:
        status = self.kube.deployment_status()
        self.tracker.update(
            ready_replicas=status["ready_replicas"], desired_replicas=status["desired_replicas"],
            monotonic_seconds=time.monotonic(), wall_time=datetime.now(timezone.utc).isoformat(),
        )

    def _write_initially_capped_manifest(self, maximum: int) -> Path:
        payload = yaml.safe_load(Path(self.spec["manifest"]).read_text(encoding="utf-8"))
        payload["spec"][self.spec["max_key"]] = int(maximum)
        # The min/max bounds are fixed before native controller creation, so
        # KEDA cannot briefly create an HPA with an unaffordable maximum.
        if self.kind == "keda":
            payload["spec"]["minReplicaCount"] = self.min_replicas
        else:
            payload["spec"]["minReplicas"] = self.min_replicas
        path = self.result_directory / f"{self.kind}_runtime_manifest.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def _govern(self, ready: int) -> bool:
        maximum = native_budget_cap(
            remaining_budget_seconds=self.tracker.remaining,
            current_ready_replicas=ready,
            min_replicas=self.min_replicas,
            max_replicas=self.max_replicas,
            control_interval_seconds=self.interval,
            native_scale_down_reserve_seconds=self.native_scale_down_reserve_seconds,
        )
        if self.applied_max_replicas is not None and maximum >= self.applied_max_replicas:
            return False
        self.kube.patch(
            self.spec["resource"], self.spec["name"], {"spec": {self.spec["max_key"]: maximum}}
        )
        self.applied_max_replicas = maximum
        _append(self.result_directory / "controller_events.jsonl", {
            "event": "native_budget_cap_reduced",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ready_replicas": ready,
            "remaining_budget_seconds": self.tracker.remaining,
            "max_replicas": maximum,
            "native_scale_down_reserve_seconds": self.native_scale_down_reserve_seconds,
        })
        return True

    def run(self, *, prepare: bool = False) -> dict[str, Any]:
        del prepare
        self._validate_prerequisites()
        if self.kube.autoscaler_conflicts():
            raise RuntimeError(f"autoscaler conflict: {self.kube.autoscaler_conflicts()}")
        self.result_directory.mkdir(parents=True, exist_ok=False)
        initial_maximum = native_budget_cap(
            remaining_budget_seconds=self.tracker.remaining,
            current_ready_replicas=self.min_replicas,
            min_replicas=self.min_replicas,
            max_replicas=self.max_replicas,
            control_interval_seconds=self.interval,
            native_scale_down_reserve_seconds=self.native_scale_down_reserve_seconds,
        )
        self.runtime_manifest_path = self._write_initially_capped_manifest(initial_maximum)
        self.applied_max_replicas = initial_maximum
        self.kube.apply_file(str(self.runtime_manifest_path))
        _append(self.result_directory / "controller_events.jsonl", {
            "event": "native_budget_cap_initialized",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "max_replicas": initial_maximum,
            "native_scale_down_reserve_seconds": self.native_scale_down_reserve_seconds,
        })
        started = time.monotonic()
        governor_activations = 0
        try:
            for step in range(self.horizon):
                cycle_started = time.monotonic()
                while time.monotonic() < started + (step + 1) * self.interval:
                    self._account()
                    status = self.kube.deployment_status()
                    if self._govern(status["ready_replicas"]):
                        governor_activations += 1
                    time.sleep(0.25)
                status = self.kube.deployment_status()
                snapshot = self.collector.collect(
                    remaining_budget_ratio=self.tracker.remaining / max(self.total_budget, 1.0),
                    remaining_horizon_ratio=(self.horizon - step - 1) / self.horizon,
                )
                _append(self.result_directory / "controller_states.jsonl", {
                    "step": step, **snapshot.as_dict(), "budget": self.tracker.samples[-1].__dict__,
                })
                _append(self.result_directory / "controller_actions.jsonl", {
                    "step": step, "action": "native_autoscaler", "target_replicas": status["desired_replicas"],
                    "ready_replicas": status["ready_replicas"], "budget_remaining_seconds": self.tracker.remaining,
                    "control_loop_latency_seconds": time.perf_counter() - cycle_started,
                })
            self.kube.delete_file(str(self.runtime_manifest_path))
            self.kube.scale(1)
            deadline = time.monotonic() + max(10.0, 3 * self.model.scale_down_guard_seconds)
            while time.monotonic() < deadline:
                self._account()
                if self.kube.deployment_status()["ready_replicas"] <= 1:
                    break
                time.sleep(0.25)
            result = {
                "schema": "dap.k8s.native_autoscaler_result.v1", "status": "completed", "kind": self.kind,
                "ready_cost_seconds": self.tracker.ready_cost,
                "requested_cost_seconds": self.tracker.requested_cost,
                "remaining_budget_seconds": self.tracker.remaining,
                "budget_violation_seconds": max(self.tracker.ready_cost - self.total_budget, 0.0),
                "budget_governor_activations": governor_activations,
                "budget_ledger": self.tracker.as_records(),
            }
            (self.result_directory / "controller_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            return result
        finally:
            if self.runtime_manifest_path is not None:
                self.kube.delete_file(str(self.runtime_manifest_path))
            self.kube.scale(1)
