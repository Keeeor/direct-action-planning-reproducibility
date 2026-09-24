from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .action_mapper import ACTION_ORDER, ActionMapper


@dataclass(frozen=True)
class ControllerConfig:
    context: str
    namespace: str
    deployment: str
    profile: str
    checkpoint_path: Path
    system_model_path: Path
    result_directory: Path
    total_budget_seconds: float
    horizon_steps: int
    control_interval_seconds: float
    slo_seconds: float
    capacity_per_pod_rps: float
    action_mapper: ActionMapper
    base_replicas: int = 1

    @classmethod
    def from_mapping(cls, values: dict[str, Any], *, base_directory: Path) -> "ControllerConfig":
        kube = values["kubernetes"]
        control = values["controller"]
        paths = values["paths"]
        mapping = values.get("actions", dict(zip(ACTION_ORDER, (1, 2, 3, 5))))
        return cls(
            context=str(kube["context"]), namespace=str(kube["namespace"]),
            deployment=str(kube.get("deployment", "dap-worker")), profile=str(control["profile"]),
            checkpoint_path=(base_directory / str(paths["checkpoint"])).resolve(),
            system_model_path=(base_directory / str(paths["system_model"])).resolve(),
            result_directory=(base_directory / str(paths["result_directory"])).resolve(),
            total_budget_seconds=float(control["total_budget_seconds"]),
            horizon_steps=int(control["horizon_steps"]),
            control_interval_seconds=float(control["control_interval_seconds"]),
            slo_seconds=float(control["slo_seconds"]),
            capacity_per_pod_rps=float(control["capacity_per_pod_rps"]),
            action_mapper=ActionMapper({name: int(mapping[name]) for name in ACTION_ORDER}),
            base_replicas=int(control.get("base_replicas", 1)),
        )


def load_controller_config(path: str | Path) -> ControllerConfig:
    path = Path(path).resolve()
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ControllerConfig.from_mapping(values, base_directory=path.parent)
